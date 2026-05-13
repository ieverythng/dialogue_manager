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
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Unit tests for the pure-function rendering helpers."""

from rqt_dialogues.render import (
    render_detail_header,
    render_history_html,
    render_label,
    render_status,
    render_updated,
)


class TestLabelRendering:
    """The label shown in the dialogue list box."""

    def test_label_for_active_chat(self):
        label = render_label({
            'role': 'default',
            'interlocutor': {'person_id': 'alice', 'group_id': ''},
            'state': 'active',
            'dialogue_id': 'abcdef1234567890',
            'history': [{'speaker_id': 'alice', 'text': 'hi'}],
        })
        assert '[default]' in label
        assert 'alice' in label
        assert 'active' in label
        assert 'abcdef12' in label
        assert '1 utt' in label

    def test_label_for_anonymous(self):
        label = render_label({
            'role': 'r',
            'interlocutor': {'person_id': '', 'group_id': ''},
            'state': 'pending',
            'dialogue_id': '01234567',
            'history': [],
        })
        assert '(anonymous)' in label
        assert '0 utt' in label

    def test_label_for_group(self):
        label = render_label({
            'role': 'r',
            'interlocutor': {'person_id': '', 'group_id': 'tour'},
            'state': 'active',
            'dialogue_id': '01234567',
            'history': [],
        })
        assert 'group:tour' in label


class TestUpdatedRendering:
    """The 'updated X ago' string."""

    def test_no_update_returns_empty(self):
        assert render_updated(None) == ''

    def test_just_now(self):
        assert render_updated(100.0, now=100.5) == 'just now'

    def test_seconds(self):
        assert render_updated(100.0, now=110.0) == '10s ago'

    def test_minutes(self):
        assert render_updated(0.0, now=180.0) == '3m ago'


class TestHistoryRendering:
    """The HTML history rendered in the detail pane."""

    def test_empty_history(self):
        html = render_history_html({
            'history': [],
            'session_start_index': 0,
        })
        assert 'No utterances yet' in html

    def test_history_marks_pre_session_entries(self):
        html = render_history_html({
            'history': [
                {'timestamp': 0.0, 'speaker_id': '__summary__', 'text': 'past'},
                {'timestamp': 0.5, 'speaker_id': '__session_break__', 'text': ''},
                {'timestamp': 1.0, 'speaker_id': 'alice', 'text': 'hello'},
            ],
            'session_start_index': 2,
            'summary': None,
        })
        # First two entries (pre-session) are italic; the third is normal.
        assert html.count('font-style:italic') == 2
        assert 'hello' in html

    def test_history_includes_summary_when_present(self):
        html = render_history_html({
            'history': [{'timestamp': 1.0, 'speaker_id': 'alice', 'text': 'hi'}],
            'session_start_index': 0,
            'summary': 'they said hi',
        })
        assert 'they said hi' in html
        assert 'Summary' in html


class TestDetailHeader:
    """The detail-pane header line."""

    def test_archived_label(self):
        html = render_detail_header(
            {'role': 'past', 'interlocutor': {'person_id': 'alice'},
             'state': 'completed', 'priority': 50, 'dialogue_id': 'abcdef12'},
            archived=True,
        )
        assert 'Archived' in html
        assert 'alice' in html
        assert 'completed' in html

    def test_summary_flag(self):
        html = render_detail_header(
            {'role': 'r', 'interlocutor': {}, 'state': 'active',
             'priority': 128, 'dialogue_id': '01234567', 'summary': 'x'},
            archived=False,
        )
        assert 'summary available' in html


class TestStatus:
    """The top-of-widget status line."""

    def test_no_chatbot(self):
        out = render_status({
            'active': True,
            'chatbot': {'configured': False},
            'current_expression_priority': -1,
            'snapshot_at': 100.0,
        }, now=100.0)
        assert 'active' in out
        assert 'no chatbot' in out

    def test_chatbot_waiting(self):
        out = render_status({
            'active': True,
            'chatbot': {
                'configured': True,
                'available': True,
                'waiting_for_response': True,
            },
            'current_expression_priority': 128,
            'snapshot_at': 100.0,
        }, now=100.0)
        assert 'chatbot ready' in out
        assert 'waiting for response' in out
        assert 'expression priority 128' in out
