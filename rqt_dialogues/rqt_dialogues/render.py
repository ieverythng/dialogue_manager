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

"""Pure-function rendering for the rqt_dialogues plugin (Qt-free, testable)."""

import time
from typing import Any


def render_label(snap: dict[str, Any]) -> str:
    """Build the one-line label shown in the active-dialogue list."""
    role = snap.get('role', '?')
    interlocutor = snap.get('interlocutor') or {}
    person = interlocutor.get('person_id') or ''
    group = interlocutor.get('group_id') or ''
    who = person or (group and f'group:{group}') or '(anonymous)'
    state = snap.get('state', '?')
    dialogue_id = snap.get('dialogue_id', '')
    short_id = dialogue_id[:8] if dialogue_id else '?'
    utt_count = len(snap.get('history') or [])
    return f'[{role}] {who} · {state} · {short_id} · {utt_count} utt'


def render_updated(last_updated: float | None, now: float | None = None) -> str:
    """Build the right-column 'updated X ago' string."""
    if last_updated is None:
        return ''
    if now is None:
        now = time.time()
    age = max(0.0, now - float(last_updated))
    if age < 1.0:
        return 'just now'
    if age < 60.0:
        return f'{age:.0f}s ago'
    return f'{age / 60.0:.0f}m ago'


def render_detail_header(snap: dict[str, Any], archived: bool) -> str:
    """Build the rich-text header line shown in the detail pane."""
    interlocutor = snap.get('interlocutor') or {}
    person = interlocutor.get('person_id') or ''
    group = interlocutor.get('group_id') or ''
    who = person or (group and f'group:{group}') or '(anonymous)'
    bits = [
        f'<b>{"Archived" if archived else "Active"}</b>',
        f'role <code>{snap.get("role", "?")}</code>',
        f'interlocutor <code>{who}</code>',
        f'state <code>{snap.get("state", "?")}</code>',
        f'priority <code>{snap.get("priority", "?")}</code>',
        f'id <code>{snap.get("dialogue_id", "")[:8]}</code>',
    ]
    if snap.get('summary'):
        bits.append('<i>summary available</i>')
    return ' · '.join(bits)


def render_history_html(snap: dict[str, Any]) -> str:
    """Build the HTML body shown in the right-hand history view."""
    history = snap.get('history') or []
    lines: list[str] = []
    session_start = snap.get('session_start_index', 0) or 0
    if snap.get('summary'):
        lines.append(
            f'<div style="margin-bottom:8px;color:#666;"><b>Summary:</b> '
            f'{snap["summary"]}</div>'
        )
    if not history:
        lines.append('<p style="color:#888;"><i>No utterances yet.</i></p>')
    for i, utt in enumerate(history):
        ts = utt.get('timestamp')
        ts_str = ''
        if ts is not None:
            ts_str = time.strftime('%H:%M:%S', time.localtime(float(ts)))
        speaker = utt.get('speaker_id', '?')
        text = utt.get('text', '')
        pre_session = i < session_start
        style = (
            'color:#777;font-style:italic;'
            if pre_session else
            ''
        )
        lines.append(
            f'<div style="margin:2px 0;{style}">'
            f'<span style="color:#999;font-family:monospace;">'
            f'[{ts_str}]</span> '
            f'<b>{speaker}:</b> {text}'
            f'</div>'
        )
    return ''.join(lines)


def render_status(snapshot: dict[str, Any], now: float | None = None) -> str:
    """Build the top-of-widget status line summarizing chatbot + activity."""
    chatbot = snapshot.get('chatbot', {})
    status = []
    status.append('active' if snapshot.get('active') else 'inactive')
    if chatbot.get('configured'):
        status.append(
            'chatbot ready' if chatbot.get('available')
            else 'chatbot unavailable'
        )
        if chatbot.get('waiting_for_response'):
            status.append('waiting for response')
    else:
        status.append('no chatbot')
    priority = snapshot.get('current_expression_priority', -1)
    if priority >= 0:
        status.append(f'expression priority {priority}')
    snap_at = snapshot.get('snapshot_at')
    if snap_at is not None:
        if now is None:
            now = time.time()
        age = max(0.0, now - float(snap_at))
        status.append(f'snapshot {age:.1f}s old')
    return ' · '.join(status)
