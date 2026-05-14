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
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Debug-state publisher for the Dialogue Manager.

Publishes a JSON snapshot of the manager's internal state to a latched
`std_msgs/String` topic so an introspection tool (e.g. the `rqt_dialogues`
plugin) can render active dialogues, their history, and archived
conversations.

The topic is **on-change only**: nothing is published until a dialogue is
added, mutated, finalized, or archived. The latched (TRANSIENT_LOCAL) QoS
ensures late subscribers immediately get the most recent snapshot.

JSON schema (see also `doc/DEBUG_STATE.md`):

    {
      "schema_version": 1,
      "snapshot_at": <epoch seconds>,
      "active": <bool>,
      "chatbot": {
        "configured": <bool>,
        "available": <bool>,
        "waiting_for_response": <bool>
      },
      "current_expression_priority": <int>,
      "dialogues": [<dialogue>, ...],
      "archived_by_interlocutor": {
        "<interlocutor.key>": [<dialogue>, ...]
      }
    }

    <dialogue> := {
      "dialogue_id": "<uuid>",
      "role": "<role name>",
      "interlocutor": {"person_id": "...", "group_id": "..."},
      "state": "pending|active|waiting_response|completed",
      "priority": <int>,
      "started_at": <float|null>,
      "ended_at": <float|null>,
      "last_updated_at": <float|null>,
      "results": "<json-encoded string or empty>",
      "session_start_index": <int>,
      "summary": "<text>|null",
      "summary_generated_at": <float|null>,
      "history": [<utterance>, ...]
    }
"""

import json
from typing import Any, Callable

from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String

from .conversations_history import ConversationsHistoryStore
from .dialogue import Dialogue, DialogueManager


DEBUG_STATE_TOPIC = '~/debug_state'
SCHEMA_VERSION = 1


class DebugStatePublisher:
    """Publish JSON snapshots of dialogue-manager state on change."""

    def __init__(
        self,
        node: Node,
        dialogue_manager: DialogueManager,
        conversations_store: ConversationsHistoryStore,
        chatbot_status: Callable[[], dict] | None = None,
        active_status: Callable[[], bool] | None = None,
    ):
        """
        Initialize the debug-state publisher.

        Parameters
        ----------
        node
            Lifecycle node hosting the publisher (used for the clock and
            publisher creation).
        dialogue_manager
            Source of truth for active dialogues.
        conversations_store
            Source of truth for archived dialogues per interlocutor.
        chatbot_status
            Callable returning a dict with keys
            `configured`/`available`/`waiting_for_response`
            describing the chatbot's current state. Returns a safely-empty
            dict by default (no chatbot).
        active_status
            Callable returning True when the lifecycle node is in its
            active state. Defaults to `lambda: True`.

        """
        self._node = node
        self._dialogue_manager = dialogue_manager
        self._conversations_store = conversations_store
        self._chatbot_status = chatbot_status or (lambda: {
            'configured': False,
            'available': False,
            'waiting_for_response': False,
        })
        self._active_status = active_status or (lambda: True)

        self._pub: Publisher | None = None

        # Tracks the per-dialogue `last_updated_at` we'd attach to the next
        # snapshot, keyed by dialogue UUID string.
        self._last_updated_at: dict[str, float] = {}

    def create_publisher(self) -> None:
        """Create the latched debug-state publisher."""
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._pub = self._node.create_publisher(
            String, DEBUG_STATE_TOPIC, qos
        )
        self._node.get_logger().info(
            f'[DEBUG] Debug-state publisher created on {DEBUG_STATE_TOPIC}'
        )

    def destroy(self) -> None:
        """Destroy the publisher."""
        if self._pub is not None:
            self._node.destroy_publisher(self._pub)
            self._pub = None

    def notify(self, dialogue_id: str | None = None) -> None:
        """
        Publish a fresh snapshot, optionally bumping a dialogue's update timestamp.

        Call this from anywhere in the dialogue_manager code where state has
        meaningfully changed: add_dialogue, remove_dialogue, add_utterance,
        archive, state transitions, chatbot status changes, etc.

        Passing `dialogue_id` advances that dialogue's `last_updated_at` so the
        UI can highlight the box that just changed.
        """
        if self._pub is None:
            return
        now = self._now()
        if dialogue_id:
            self._last_updated_at[dialogue_id] = now
        try:
            payload = self._build_snapshot(now)
        except Exception as exc:  # pragma: no cover - safety net
            self._node.get_logger().warn(
                f'[DEBUG] Snapshot build failed: {exc}'
            )
            return
        self._pub.publish(String(data=json.dumps(payload)))

    # ------------------------------------------------------------------ build

    def _now(self) -> float:
        return self._node.get_clock().now().nanoseconds / 1e9

    def _build_snapshot(self, now: float) -> dict[str, Any]:
        return {
            'schema_version': SCHEMA_VERSION,
            'snapshot_at': now,
            'active': bool(self._active_status()),
            'chatbot': self._chatbot_status(),
            'current_expression_priority':
                self._dialogue_manager.current_max_priority,
            'dialogues': [
                self._dialogue_to_dict(d)
                for d in self._dialogue_manager.active_dialogues.values()
            ],
            'archived_by_interlocutor': self._archived(),
        }

    def _dialogue_to_dict(self, d: Dialogue) -> dict[str, Any]:
        return {
            'dialogue_id': str(d.dialogue_id),
            'role': d.role.name,
            'interlocutor': {
                'person_id': d.interlocutor.person_id,
                'group_id': d.interlocutor.group_id,
            },
            'state': d.state.value,
            'priority': d.priority,
            'started_at': d.started_at,
            'ended_at': d.ended_at,
            'last_updated_at': self._last_updated_at.get(str(d.dialogue_id)),
            'results': d.results,
            'session_start_index': d.session_start_index,
            'summary': d.summary,
            'summary_generated_at': d.summary_generated_at,
            'history': [
                {
                    'timestamp': u.timestamp,
                    'speaker_id': u.speaker_id,
                    'text': u.text,
                }
                for u in d.history
            ],
        }

    def _archived(self) -> dict[str, list[dict[str, Any]]]:
        # ConversationsHistoryStore intentionally exposes a per-interlocutor
        # view via history_for(); we walk its internal map directly because
        # we need every interlocutor's bucket.
        result: dict[str, list[dict[str, Any]]] = {}
        for key, bucket in self._conversations_store._by_interlocutor.items():
            result[key] = [self._dialogue_to_dict(d) for d in bucket]
        return result
