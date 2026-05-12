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

"""Unit tests for speech_handler.py with mocked ROS2 dependencies."""

from unittest.mock import MagicMock
from uuid import uuid4

from chatbot_msgs.msg import DialogueRole
from dialogue_manager.conversations_history import ConversationsHistoryStore
from dialogue_manager.dialogue import (
    Dialogue,
    DialogueManager,
    DialogueState,
    Interlocutor,
)
from dialogue_manager.speech_handler import SpeechHandler
from hri_msgs.msg import IdsList, LiveSpeech


class TestSpeechHandlerInit:
    """Tests for SpeechHandler initialization."""

    def test_init_sets_properties(self):
        """Handler stores all provided dependencies."""
        mock_node = MagicMock()
        mock_dialogue_manager = MagicMock()
        mock_chatbot_client = MagicMock()
        mock_captions_pub = MagicMock()
        mock_intents_pub = MagicMock()

        handler = SpeechHandler(
            node=mock_node,
            dialogue_manager=mock_dialogue_manager,
            chatbot_client=mock_chatbot_client,
            conversations_store=ConversationsHistoryStore(),
            closed_captions_pub=mock_captions_pub,
            intents_pub=mock_intents_pub
        )

        assert handler._node is mock_node
        assert handler._dialogue_manager is mock_dialogue_manager
        assert handler._chatbot_client is mock_chatbot_client
        assert handler._closed_captions_pub is mock_captions_pub
        assert handler._intents_pub is mock_intents_pub

    def test_init_starts_with_empty_voices(self):
        """Handler starts with no tracked voices."""
        handler = SpeechHandler(
            node=MagicMock(),
            dialogue_manager=MagicMock(),
            chatbot_client=MagicMock(),
            conversations_store=ConversationsHistoryStore(),
            closed_captions_pub=MagicMock(),
            intents_pub=MagicMock()
        )

        assert len(handler._tracked_voices) == 0
        assert len(handler._voice_subscriptions) == 0

    def test_init_chatbot_enabled_by_default(self):
        """Handler starts with chatbot enabled."""
        handler = SpeechHandler(
            node=MagicMock(),
            dialogue_manager=MagicMock(),
            chatbot_client=MagicMock(),
            conversations_store=ConversationsHistoryStore(),
            closed_captions_pub=MagicMock(),
            intents_pub=MagicMock()
        )

        assert handler._chatbot_enabled is True


class TestSpeechHandlerChatbotEnabled:
    """Tests for chatbot enable/disable."""

    def test_set_chatbot_enabled_true(self):
        """set_chatbot_enabled sets to True."""
        handler = SpeechHandler(
            node=MagicMock(),
            dialogue_manager=MagicMock(),
            chatbot_client=MagicMock(),
            conversations_store=ConversationsHistoryStore(),
            closed_captions_pub=MagicMock(),
            intents_pub=MagicMock()
        )

        handler.set_chatbot_enabled(True)

        assert handler._chatbot_enabled is True

    def test_set_chatbot_enabled_false(self):
        """set_chatbot_enabled sets to False."""
        handler = SpeechHandler(
            node=MagicMock(),
            dialogue_manager=MagicMock(),
            chatbot_client=MagicMock(),
            conversations_store=ConversationsHistoryStore(),
            closed_captions_pub=MagicMock(),
            intents_pub=MagicMock()
        )

        handler.set_chatbot_enabled(False)

        assert handler._chatbot_enabled is False


class TestSpeechHandlerVoiceTracking:
    """Tests for voice subscription management."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_node = MagicMock()
        self.mock_dialogue_manager = MagicMock()
        self.mock_chatbot_client = MagicMock()

        self.handler = SpeechHandler(
            node=self.mock_node,
            dialogue_manager=self.mock_dialogue_manager,
            chatbot_client=self.mock_chatbot_client,
            conversations_store=ConversationsHistoryStore(),
            closed_captions_pub=MagicMock(),
            intents_pub=MagicMock()
        )

    def test_subscribe_to_voices_creates_subscription(self):
        """subscribe_to_voices creates voices subscription."""
        self.handler.subscribe_to_voices()

        self.mock_node.create_subscription.assert_called_once()
        assert self.handler._voices_sub is not None

    def test_on_voices_tracked_adds_new_voice(self):
        """New voices create subscriptions."""
        msg = IdsList()
        msg.ids = ['voice1', 'voice2']

        self.handler._on_voices_tracked(msg)

        assert 'voice1' in self.handler._tracked_voices
        assert 'voice2' in self.handler._tracked_voices
        assert self.mock_node.create_subscription.call_count == 2

    def test_on_voices_tracked_removes_old_voice(self):
        """Removed voices destroy subscriptions."""
        # First add voices
        msg1 = IdsList()
        msg1.ids = ['voice1', 'voice2']
        self.handler._on_voices_tracked(msg1)

        # Then remove one
        msg2 = IdsList()
        msg2.ids = ['voice1']
        self.handler._on_voices_tracked(msg2)

        assert 'voice1' in self.handler._tracked_voices
        assert 'voice2' not in self.handler._tracked_voices
        self.mock_node.destroy_subscription.assert_called_once()

    def test_unsubscribe_all_clears_subscriptions(self):
        """unsubscribe_all clears all voice subscriptions."""
        # Add voices
        msg = IdsList()
        msg.ids = ['voice1', 'voice2']
        self.handler._on_voices_tracked(msg)

        # Unsubscribe all
        self.handler.unsubscribe_all()

        assert len(self.handler._tracked_voices) == 0
        assert len(self.handler._voice_subscriptions) == 0


class TestSpeechHandlerOnSpeech:
    """Tests for speech handling logic."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_node = MagicMock()
        self.mock_dialogue_manager = DialogueManager()
        self.mock_chatbot_client = MagicMock()
        self.mock_chatbot_client.waiting_for_response = False
        self.mock_captions_pub = MagicMock()
        self.mock_intents_pub = MagicMock()

        self.handler = SpeechHandler(
            node=self.mock_node,
            dialogue_manager=self.mock_dialogue_manager,
            chatbot_client=self.mock_chatbot_client,
            conversations_store=ConversationsHistoryStore(),
            closed_captions_pub=self.mock_captions_pub,
            intents_pub=self.mock_intents_pub
        )

    def test_on_speech_ignores_non_final(self):
        """Non-final speech is ignored."""
        msg = LiveSpeech()
        msg.final = ''  # Empty = not final

        self.handler._on_speech('voice1', msg)

        # Nothing should be published
        self.mock_captions_pub.publish.assert_not_called()
        self.mock_intents_pub.publish.assert_not_called()

    def test_on_speech_ignores_while_waiting(self):
        """Speech is ignored while waiting for chatbot response."""
        self.mock_chatbot_client.waiting_for_response = True

        msg = LiveSpeech()
        msg.final = 'Hello'

        self.handler._on_speech('voice1', msg)

        # Chatbot should not receive input
        self.mock_chatbot_client.send_input.assert_not_called()

    def test_on_speech_publishes_caption(self):
        """Final speech publishes closed caption."""
        msg = LiveSpeech()
        msg.final = 'Hello world'
        msg.locale = 'en'

        self.handler._on_speech('voice1', msg)

        self.mock_captions_pub.publish.assert_called_once()
        caption = self.mock_captions_pub.publish.call_args[0][0]
        assert caption.text == 'Hello world'
        assert caption.speaker_id == 'voice1'

    def test_on_speech_chatbot_disabled_publishes_intent(self):
        """Speech publishes raw intent when chatbot disabled."""
        self.handler._chatbot_enabled = False

        msg = LiveSpeech()
        msg.final = 'Hello'

        self.handler._on_speech('voice1', msg)

        self.mock_intents_pub.publish.assert_called_once()

    def test_on_speech_uses_person_dialogue(self):
        """Speech routes to person's active dialogue."""
        # Create active dialogue for this person
        role = DialogueRole(name='test')
        chatbot_goal_id = uuid4()
        dialogue = Dialogue(
            role=role,
            interlocutor=Interlocutor(person_id='voice1'),
            state=DialogueState.ACTIVE,
            chatbot_goal_id=chatbot_goal_id
        )
        self.mock_dialogue_manager.add_dialogue(dialogue)

        msg = LiveSpeech()
        msg.final = 'Hello'

        self.handler._on_speech('voice1', msg)

        self.mock_chatbot_client.send_input.assert_called_once_with(
            dialogue.dialogue_id, 'voice1', 'Hello'
        )

    def test_on_speech_spawns_per_person_default_dialogue(self):
        """First speech from a new speaker auto-spawns a per-person Dialogue."""
        self.handler.set_default_chat('__default__')

        msg = LiveSpeech()
        msg.final = 'hiya'

        self.handler._on_speech('alice', msg)

        # A new per-person dialogue was created.
        spawned = self.mock_dialogue_manager.get_dialogue_for_interlocutor(
            Interlocutor(person_id='alice')
        )
        assert spawned is not None
        assert spawned.role.name == '__default__'
        assert spawned.interlocutor.person_id == 'alice'
        # The utterance lives in that dialogue's history.
        assert len(spawned.history) == 1
        assert spawned.history[0].text == 'hiya'
        # And the chatbot was asked to attach (mock records the call).
        self.mock_chatbot_client.attach_to_dialogue.assert_called_once()

    def test_on_speech_spawns_one_dialogue_per_person(self):
        """Different speakers each get their own __default__ dialogue."""
        self.handler.set_default_chat('__default__')

        msg_a = LiveSpeech()
        msg_a.final = 'hi from A'
        msg_b = LiveSpeech()
        msg_b.final = 'hi from B'

        self.handler._on_speech('alice', msg_a)
        self.handler._on_speech('bob', msg_b)

        d_a = self.mock_dialogue_manager.get_dialogue_for_interlocutor(
            Interlocutor(person_id='alice')
        )
        d_b = self.mock_dialogue_manager.get_dialogue_for_interlocutor(
            Interlocutor(person_id='bob')
        )
        assert d_a is not None and d_b is not None
        assert d_a.dialogue_id != d_b.dialogue_id
        assert [u.text for u in d_a.history] == ['hi from A']
        assert [u.text for u in d_b.history] == ['hi from B']

    def test_on_speech_reuses_existing_per_person_dialogue(self):
        """Subsequent utterances from the same speaker land in the same Dialogue."""
        self.handler.set_default_chat('__default__')

        msg1 = LiveSpeech()
        msg1.final = 'hi'
        msg2 = LiveSpeech()
        msg2.final = 'how are you'

        self.handler._on_speech('alice', msg1)
        self.handler._on_speech('alice', msg2)

        d = self.mock_dialogue_manager.get_dialogue_for_interlocutor(
            Interlocutor(person_id='alice')
        )
        assert d is not None
        assert [u.text for u in d.history] == ['hi', 'how are you']
        # Only one chatbot attach call (for the spawn).
        assert self.mock_chatbot_client.attach_to_dialogue.call_count == 1

    def test_on_speech_chatbot_less_does_not_attach(self):
        """No chatbot present → spawn dialogue but never call attach_to_dialogue."""
        self.handler._chatbot_client = None
        self.handler.set_default_chat('__default__')

        msg = LiveSpeech()
        msg.final = 'hello'

        self.handler._on_speech('alice', msg)

        d = self.mock_dialogue_manager.get_dialogue_for_interlocutor(
            Interlocutor(person_id='alice')
        )
        assert d is not None
        assert d.chatbot_goal_id is None
        assert len(d.history) == 1
        # RAW_USER_INPUT still published.
        self.mock_intents_pub.publish.assert_called_once()

    def test_on_speech_default_chat_disabled_does_not_spawn(self):
        """When default chat is off, unmatched speech is RAW_USER_INPUT only."""
        # Don't call set_default_chat → no auto-spawn.

        msg = LiveSpeech()
        msg.final = 'hello'

        self.handler._on_speech('alice', msg)

        d = self.mock_dialogue_manager.get_dialogue_for_interlocutor(
            Interlocutor(person_id='alice')
        )
        assert d is None
        self.mock_intents_pub.publish.assert_called_once()

    def test_on_speech_no_dialogue_publishes_intent(self):
        """Speech with no active dialogue publishes raw intent."""
        msg = LiveSpeech()
        msg.final = 'Hello'

        self.handler._on_speech('voice1', msg)

        self.mock_intents_pub.publish.assert_called_once()


class TestSpeechHandlerIntentPublishing:
    """Tests for intent publishing."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_intents_pub = MagicMock()

        self.handler = SpeechHandler(
            node=MagicMock(),
            dialogue_manager=DialogueManager(),
            chatbot_client=MagicMock(),
            conversations_store=ConversationsHistoryStore(),
            closed_captions_pub=MagicMock(),
            intents_pub=self.mock_intents_pub
        )

    def test_publish_raw_intent_sets_fields(self):
        """_publish_raw_intent sets all intent fields."""
        self.handler._publish_raw_intent('Hello', 'voice1', 'en')

        self.mock_intents_pub.publish.assert_called_once()
        intent = self.mock_intents_pub.publish.call_args[0][0]

        # Verify intent fields
        assert intent.source == 'voice1'
        assert intent.priority == 128
        assert intent.confidence == 1.0
