# Copyright (c) 2026 IIIA-CSIC. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Widget rendering the dialogue_manager debug state in real time."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import time
from typing import Any

from python_qt_binding.QtCore import Qt, QTimer, Signal
from python_qt_binding.QtGui import QColor
from python_qt_binding.QtWidgets import (
    QHeaderView,
    QLabel,
    QSplitter,
    QTextBrowser,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String

from .render import (
    render_detail_header,
    render_history_html,
    render_label,
    render_status,
    render_updated,
)


DEBUG_TOPIC = '/dialogue_manager/debug_state'
HIGHLIGHT_DURATION_MS = 1500
HIGHLIGHT_COLOR = QColor(255, 230, 120)  # warm yellow


@dataclass
class _ItemState:
    """Tracks per-dialogue UI state across snapshots."""

    item: QTreeWidgetItem
    last_updated_at: float | None = None
    decay_until: float = 0.0
    archived: bool = False
    payload: dict[str, Any] = field(default_factory=dict)


class DialogueView(QWidget):
    """Two-pane Qt widget rendering the dialogue manager state."""

    # Marshals snapshots from rclpy callback threads onto the Qt main thread.
    _snapshot_received = Signal(dict)

    def __init__(self, node, parent=None):
        """Build the widget and subscribe to the debug topic."""
        super().__init__(parent)
        self._node = node
        self.setObjectName('DialogueView')
        self.setWindowTitle('Dialogues')

        # Maps "dialogue_id" -> _ItemState
        self._items: dict[str, _ItemState] = {}
        self._selected_id: str | None = None

        self._build_ui()

        # Subscribe with TRANSIENT_LOCAL to immediately get the latched
        # snapshot when this widget starts up.
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._sub = node.create_subscription(
            String, DEBUG_TOPIC, self._on_msg, qos,
        )
        self._snapshot_received.connect(self._apply_snapshot)

        # Drives the "fade highlight" animation on recently-updated boxes.
        self._decay_timer = QTimer(self)
        self._decay_timer.setInterval(150)
        self._decay_timer.timeout.connect(self._tick_decay)
        self._decay_timer.start()

    # ------------------------------------------------------------------ build

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)

        self._status_label = QLabel('Waiting for snapshot...')
        self._status_label.setObjectName('status_label')
        outer.addWidget(self._status_label)

        splitter = QSplitter(Qt.Horizontal, self)

        # ----- Left: dialogue tree (active + archived) -----
        self._tree = QTreeWidget()
        self._tree.setColumnCount(2)
        self._tree.setHeaderLabels(['Dialogue', 'Updated'])
        self._tree.header().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        self._tree.header().setSectionResizeMode(
            1, QHeaderView.ResizeToContents
        )
        self._tree.itemSelectionChanged.connect(self._on_select)
        splitter.addWidget(self._tree)

        # ----- Right: detail pane -----
        detail = QWidget()
        detail_layout = QVBoxLayout(detail)

        self._detail_header = QLabel('Select a dialogue to inspect')
        self._detail_header.setWordWrap(True)
        self._detail_header.setObjectName('detail_header')
        detail_layout.addWidget(self._detail_header)

        self._history_view = QTextBrowser()
        self._history_view.setOpenExternalLinks(False)
        detail_layout.addWidget(self._history_view, stretch=1)

        splitter.addWidget(detail)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        outer.addWidget(splitter, stretch=1)

        # Top-level tree sections (built once, repopulated below).
        self._active_section = QTreeWidgetItem(['Active dialogues', ''])
        self._active_section.setFlags(self._active_section.flags() & ~Qt.ItemIsSelectable)
        self._tree.addTopLevelItem(self._active_section)
        self._active_section.setExpanded(True)

        self._archived_section = QTreeWidgetItem(['Archived (by interlocutor)', ''])
        self._archived_section.setFlags(self._archived_section.flags() & ~Qt.ItemIsSelectable)
        self._tree.addTopLevelItem(self._archived_section)
        self._archived_section.setExpanded(True)

    # --------------------------------------------------------- subscription

    def _on_msg(self, msg: String) -> None:
        """Receive a snapshot from rclpy and marshal it onto the GUI thread."""
        try:
            snapshot = json.loads(msg.data)
        except Exception:
            return
        self._snapshot_received.emit(snapshot)

    def shutdown(self) -> None:
        """Tear down ROS subscriptions and timers."""
        if self._decay_timer is not None:
            self._decay_timer.stop()
        if self._sub is not None:
            self._node.destroy_subscription(self._sub)
            self._sub = None

    # ---------------------------------------------------------- snapshot

    def _apply_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Update the UI from a fresh debug-state snapshot."""
        self._update_status_label(snapshot)
        self._refresh_active(snapshot.get('dialogues', []))
        self._refresh_archived(snapshot.get('archived_by_interlocutor', {}))
        self._refresh_selected_detail()

    def _update_status_label(self, snapshot: dict[str, Any]) -> None:
        self._status_label.setText(render_status(snapshot))

    def _refresh_active(self, dialogues: list[dict[str, Any]]) -> None:
        live_ids = set()
        for snap in dialogues:
            did = snap['dialogue_id']
            live_ids.add(did)
            self._upsert_dialogue_item(
                did=did,
                parent=self._active_section,
                payload=snap,
                archived=False,
            )
        # Remove items that are no longer active.
        for did in list(self._items.keys()):
            state = self._items[did]
            if state.archived:
                continue
            if did not in live_ids:
                self._remove_item(did)

    def _refresh_archived(
        self, archived: dict[str, list[dict[str, Any]]]
    ) -> None:
        # Track which (interlocutor, dialogue_id) pairs are still alive.
        alive: set[str] = set()
        # Build interlocutor section nodes on the fly.
        existing_interlocutor_nodes = {}
        for i in range(self._archived_section.childCount()):
            child = self._archived_section.child(i)
            existing_interlocutor_nodes[child.text(0)] = child

        for interlocutor_key, snaps in sorted(archived.items()):
            node = existing_interlocutor_nodes.pop(interlocutor_key, None)
            if node is None:
                node = QTreeWidgetItem([interlocutor_key, ''])
                node.setFlags(node.flags() & ~Qt.ItemIsSelectable)
                self._archived_section.addChild(node)
                node.setExpanded(False)
            # Re-build children under this interlocutor.
            current_dids = set()
            for snap in snaps:
                did = snap['dialogue_id']
                # Disambiguate archived from active duplicate (shouldn't
                # happen but be defensive).
                key = f'archived:{interlocutor_key}:{did}'
                current_dids.add(key)
                alive.add(key)
                self._upsert_dialogue_item(
                    did=key,
                    parent=node,
                    payload=snap,
                    archived=True,
                )
            # Remove stale archived children under this interlocutor.
            for key in list(self._items.keys()):
                if not key.startswith(f'archived:{interlocutor_key}:'):
                    continue
                if key not in current_dids:
                    self._remove_item(key)

        # Remove interlocutor nodes that no longer have any archived data.
        for key, node in existing_interlocutor_nodes.items():
            self._archived_section.removeChild(node)
            # Also drop any tracked items under it.
            for k in list(self._items.keys()):
                if k.startswith(f'archived:{key}:'):
                    self._remove_item(k)

    # --------------------------------------------------------- item upsert

    def _upsert_dialogue_item(
        self,
        did: str,
        parent: QTreeWidgetItem,
        payload: dict[str, Any],
        archived: bool,
    ) -> None:
        state = self._items.get(did)
        label = render_label(payload)
        last_updated = payload.get('last_updated_at')

        if state is None:
            item = QTreeWidgetItem([label, ''])
            item.setData(0, Qt.UserRole, did)
            parent.addChild(item)
            state = _ItemState(item=item)
            self._items[did] = state
        else:
            state.item.setText(0, label)

        state.archived = archived
        state.payload = payload

        if (last_updated is not None
                and last_updated != state.last_updated_at):
            state.last_updated_at = last_updated
            state.decay_until = time.monotonic() + HIGHLIGHT_DURATION_MS / 1000.0
            self._paint_item(state, 1.0)
        else:
            self._paint_item(state, self._current_intensity(state))

        state.item.setText(1, render_updated(last_updated))

    def _remove_item(self, did: str) -> None:
        state = self._items.pop(did, None)
        if state is None:
            return
        parent = state.item.parent()
        if parent is not None:
            parent.removeChild(state.item)
        if self._selected_id == did:
            self._selected_id = None
            self._detail_header.setText('Select a dialogue to inspect')
            self._history_view.clear()

    # -------------------------------------------------------- rendering

    def _current_intensity(self, state: _ItemState) -> float:
        remaining = state.decay_until - time.monotonic()
        if remaining <= 0:
            return 0.0
        return max(0.0, min(1.0, remaining / (HIGHLIGHT_DURATION_MS / 1000.0)))

    def _paint_item(self, state: _ItemState, intensity: float) -> None:
        """Blend the row background toward HIGHLIGHT_COLOR by `intensity`."""
        if intensity <= 0.0:
            color = QColor(0, 0, 0, 0)  # transparent → default theme
        else:
            color = QColor(HIGHLIGHT_COLOR)
            color.setAlphaF(intensity)
        state.item.setBackground(0, color)
        state.item.setBackground(1, color)

    def _tick_decay(self) -> None:
        """Fade highlight backgrounds toward zero on each timer tick."""
        for state in self._items.values():
            if state.decay_until == 0.0:
                continue
            intensity = self._current_intensity(state)
            self._paint_item(state, intensity)
            if intensity == 0.0:
                state.decay_until = 0.0
            # Also refresh the "Updated" column so ages don't stay stale.
            state.item.setText(1, render_updated(state.last_updated_at))

    # ------------------------------------------------------------ selection

    def _on_select(self) -> None:
        items = self._tree.selectedItems()
        if not items:
            self._selected_id = None
            return
        did = items[0].data(0, Qt.UserRole)
        if not did:
            return
        self._selected_id = did
        self._refresh_selected_detail()

    def _refresh_selected_detail(self) -> None:
        if self._selected_id is None:
            return
        state = self._items.get(self._selected_id)
        if state is None:
            return
        snap = state.payload
        self._detail_header.setText(render_detail_header(snap, state.archived))
        self._history_view.setHtml(render_history_html(snap))
