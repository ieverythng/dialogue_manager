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

"""Unit tests for the debug-state publisher."""

import json
from unittest.mock import MagicMock

from chatbot_msgs.msg import DialogueRole
from dialogue_manager.conversations_history import ConversationsHistoryStore
from dialogue_manager.debug_publisher import (
    DebugStatePublisher,
    SCHEMA_VERSION,
)
from dialogue_manager.dialogue import (
    Dialogue,
    DialogueManager,
    DialogueState,
    Interlocutor,
)


def _make_node(now: float = 1234.5) -> MagicMock:
    """Build a node whose clock returns `now` seconds."""
    node = MagicMock()
    node.get_clock.return_value.now.return_value.nanoseconds = int(now * 1e9)
    return node


def _make_publisher(
    dialogue_manager: DialogueManager | None = None,
    store: ConversationsHistoryStore | None = None,
    chatbot_status: dict | None = None,
    active: bool = True,
    now: float = 1234.5,
) -> tuple[DebugStatePublisher, MagicMock]:
    node = _make_node(now=now)
    pub = DebugStatePublisher(
        node=node,
        dialogue_manager=dialogue_manager or DialogueManager(),
        conversations_store=store or ConversationsHistoryStore(),
        chatbot_status=lambda: (chatbot_status or {
            'configured': False,
            'available': False,
            'waiting_for_response': False,
            'default_dialogue_id': None,
        }),
        active_status=lambda: active,
    )
    pub._pub = MagicMock()  # bypass create_publisher
    return pub, pub._pub


def _payload(pub_mock: MagicMock) -> dict:
    """Extract the last published JSON payload from a mock publisher."""
    assert pub_mock.publish.call_args is not None
    msg = pub_mock.publish.call_args[0][0]
    return json.loads(msg.data)


class TestDebugSnapshotShape:
    """Validate the top-level snapshot schema."""

    def test_schema_version_and_clock(self):
        pub, pub_mock = _make_publisher(now=42.0)
        pub.notify()
        payload = _payload(pub_mock)
        assert payload['schema_version'] == SCHEMA_VERSION
        assert payload['snapshot_at'] == 42.0
        assert payload['active'] is True

    def test_empty_state_lists_no_dialogues(self):
        pub, pub_mock = _make_publisher()
        pub.notify()
        payload = _payload(pub_mock)
        assert payload['dialogues'] == []
        assert payload['archived_by_interlocutor'] == {}

    def test_chatbot_status_propagates(self):
        pub, pub_mock = _make_publisher(
            chatbot_status={
                'configured': True,
                'available': True,
                'waiting_for_response': True,
                'default_dialogue_id': 'abc',
            }
        )
        pub.notify()
        chatbot = _payload(pub_mock)['chatbot']
        assert chatbot == {
            'configured': True,
            'available': True,
            'waiting_for_response': True,
            'default_dialogue_id': 'abc',
        }

    def test_inactive_node(self):
        pub, pub_mock = _make_publisher(active=False)
        pub.notify()
        assert _payload(pub_mock)['active'] is False


class TestActiveDialoguesSerialization:
    """Active dialogues are serialized fully (with history)."""

    def test_active_dialogue_includes_full_history(self):
        dm = DialogueManager()
        d = Dialogue(
            role=DialogueRole(name='default'),
            interlocutor=Interlocutor(person_id='alice'),
            state=DialogueState.ACTIVE,
        )
        d.add_utterance('alice', 'hi', 100.0)
        d.add_utterance('__myself__', 'hello', 101.0)
        dm.add_dialogue(d)

        pub, pub_mock = _make_publisher(dialogue_manager=dm)
        pub.notify()

        dialogues = _payload(pub_mock)['dialogues']
        assert len(dialogues) == 1
        snap = dialogues[0]
        assert snap['dialogue_id'] == str(d.dialogue_id)
        assert snap['role'] == 'default'
        assert snap['interlocutor'] == {'person_id': 'alice', 'group_id': ''}
        assert snap['state'] == 'active'
        assert len(snap['history']) == 2
        assert snap['history'][0]['speaker_id'] == 'alice'
        assert snap['history'][1]['speaker_id'] == '__myself__'

    def test_last_updated_at_set_when_notify_targets_dialogue(self):
        dm = DialogueManager()
        d = Dialogue(
            role=DialogueRole(name='test'),
            interlocutor=Interlocutor(person_id='alice'),
            state=DialogueState.ACTIVE,
        )
        dm.add_dialogue(d)
        pub, pub_mock = _make_publisher(dialogue_manager=dm, now=500.0)
        pub.notify(str(d.dialogue_id))

        snap = _payload(pub_mock)['dialogues'][0]
        assert snap['last_updated_at'] == 500.0


class TestArchivedDialoguesSerialization:
    """Archived dialogues appear under per-interlocutor keys."""

    def test_archived_grouped_by_interlocutor_key(self):
        store = ConversationsHistoryStore()
        prior = Dialogue(
            role=DialogueRole(name='past'),
            interlocutor=Interlocutor(person_id='alice'),
            state=DialogueState.COMPLETED,
        )
        prior.add_utterance('alice', 'hi', 1.0)
        prior.summary = 'past chat'
        store.archive(prior)

        pub, pub_mock = _make_publisher(store=store)
        pub.notify()

        archived = _payload(pub_mock)['archived_by_interlocutor']
        assert 'person:alice' in archived
        assert len(archived['person:alice']) == 1
        archived_snap = archived['person:alice'][0]
        assert archived_snap['summary'] == 'past chat'
        assert archived_snap['history'][0]['text'] == 'hi'


class TestDialogueManagerHooks:
    """Hook integration: add_utterance / add_dialogue trigger notify."""

    def test_add_dialogue_fires_callback(self):
        dm = DialogueManager()
        events = []
        dm.set_change_callback(lambda did: events.append(did))

        d = Dialogue(
            role=DialogueRole(name='test'),
            interlocutor=Interlocutor(person_id='alice'),
        )
        dm.add_dialogue(d)

        assert events == [str(d.dialogue_id)]

    def test_add_utterance_fires_callback(self):
        dm = DialogueManager()
        d = Dialogue(
            role=DialogueRole(name='test'),
            interlocutor=Interlocutor(person_id='alice'),
        )
        dm.add_dialogue(d)
        events = []
        dm.set_change_callback(lambda did: events.append(did))

        d.add_utterance('alice', 'hi', 1.0)

        assert events == [str(d.dialogue_id)]

    def test_remove_dialogue_fires_callback(self):
        dm = DialogueManager()
        d = Dialogue(
            role=DialogueRole(name='test'),
            interlocutor=Interlocutor(person_id='alice'),
        )
        dm.add_dialogue(d)
        events = []
        dm.set_change_callback(lambda did: events.append(did))

        dm.remove_dialogue(d.dialogue_id)

        assert events == [str(d.dialogue_id)]

    def test_post_remove_utterance_does_not_fire(self):
        """Once a dialogue is removed, its mutations no longer notify."""
        dm = DialogueManager()
        d = Dialogue(
            role=DialogueRole(name='test'),
            interlocutor=Interlocutor(person_id='alice'),
        )
        dm.add_dialogue(d)
        dm.remove_dialogue(d.dialogue_id)
        events = []
        dm.set_change_callback(lambda did: events.append(did))

        d.add_utterance('alice', 'hi', 1.0)

        assert events == []

    def test_notify_with_no_callback_is_noop(self):
        dm = DialogueManager()
        dm.notify_change(None)  # must not raise


class TestPublisherSafety:
    """Edge cases that must not crash."""

    def test_notify_before_create_publisher_is_noop(self):
        pub, _ = _make_publisher()
        pub._pub = None  # simulate not-yet-created
        pub.notify()  # must not raise
