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

"""Unit tests for chatbot_client.py with mocked ROS2 dependencies."""

from unittest.mock import MagicMock
from uuid import UUID, uuid4

from chatbot_msgs.msg import DialogueRole, Utterance

from dialogue_manager.chatbot_client import ChatbotClient, uuid_to_msg
from dialogue_manager.dialogue import (
    Dialogue,
    DialogueManager,
    DialogueState,
    ROBOT_SPEAKER_ID,
    SYSTEM_SPEAKER_ID,
)


class TestUuidToMsg:
    """Tests for the uuid_to_msg helper function."""

    def test_uuid_to_msg_conversion(self):
        """Convert Python UUID to ROS UUID message."""
        test_uuid = uuid4()
        msg = uuid_to_msg(test_uuid)
        assert bytes(msg.uuid) == test_uuid.bytes

    def test_uuid_to_msg_preserves_value(self):
        """UUID round-trip preserves value."""
        test_uuid = uuid4()
        msg = uuid_to_msg(test_uuid)
        recovered = UUID(bytes=bytes(msg.uuid))
        assert recovered == test_uuid


def _make_client(dialogue_manager=None) -> ChatbotClient:
    """Construct a bare ChatbotClient with mocked ROS deps."""
    return ChatbotClient(
        node=MagicMock(),
        dialogue_manager=dialogue_manager or DialogueManager(),
        say_client=MagicMock(),
        intents_pub=MagicMock(),
        waiting_chatbot_pub=MagicMock(),
    )


class TestChatbotClientInit:
    """Tests for ChatbotClient initialization."""

    def test_init_sets_properties(self):
        """Client stores all provided dependencies."""
        mock_node = MagicMock()
        mock_dm = MagicMock()
        mock_say = MagicMock()
        mock_intents = MagicMock()
        mock_waiting = MagicMock()

        client = ChatbotClient(
            node=mock_node,
            dialogue_manager=mock_dm,
            say_client=mock_say,
            intents_pub=mock_intents,
            waiting_chatbot_pub=mock_waiting,
        )

        assert client._node is mock_node
        assert client._dialogue_manager is mock_dm
        assert client._say_client is mock_say
        assert client._intents_pub is mock_intents
        assert client._waiting_chatbot_pub is mock_waiting

    def test_init_waiting_for_response_false(self):
        """Client starts not waiting for response."""
        client = _make_client()
        assert client.waiting_for_response is False


class TestChatbotClientCreateClients:
    """Tests for creating chatbot service clients."""

    def test_create_clients_creates_two_service_clients(self):
        """create_clients creates prepare + interaction service clients."""
        mock_node = MagicMock()
        client = ChatbotClient(
            node=mock_node,
            dialogue_manager=MagicMock(),
            say_client=MagicMock(),
            intents_pub=MagicMock(),
            waiting_chatbot_pub=MagicMock(),
        )

        client.create_clients('chatbot')

        assert mock_node.create_client.call_count == 2
        assert client._prepare_client is not None
        assert client._interaction_client is not None


class TestChatbotClientPrepare:
    """Tests for prepare()."""

    def test_prepare_no_client_returns_false(self):
        """No prepare client -> prepare() refuses."""
        client = _make_client()
        dialogue = Dialogue(role=DialogueRole(name='r'))
        assert client.prepare(dialogue) is False

    def test_prepare_service_not_ready_returns_false(self):
        """If prepare service isn't ready, skip — non-fatal."""
        client = _make_client()
        client._prepare_client = MagicMock()
        client._prepare_client.service_is_ready.return_value = False
        dialogue = Dialogue(role=DialogueRole(name='r'))
        assert client.prepare(dialogue) is False
        client._prepare_client.call_async.assert_not_called()

    def test_prepare_sends_request_with_role_and_id(self):
        """A ready service receives a PrepareDialogue with role + id."""
        client = _make_client()
        mock_srv = MagicMock()
        mock_srv.service_is_ready.return_value = True
        client._prepare_client = mock_srv

        role = DialogueRole(name='__default__')
        dialogue = Dialogue(role=role)
        assert client.prepare(dialogue) is True

        request = mock_srv.call_async.call_args[0][0]
        assert request.role.name == '__default__'
        assert bytes(request.dialogue_id.uuid) == dialogue.dialogue_id.bytes


class TestChatbotClientInteract:
    """Tests for interact() — the main turn driver."""

    def _client_with_ready_interaction(self):
        manager = DialogueManager()
        client = _make_client(dialogue_manager=manager)
        mock_srv = MagicMock()
        mock_srv.call_async.return_value = MagicMock()
        client._interaction_client = mock_srv
        return manager, client, mock_srv

    def test_interact_no_client_returns_false(self):
        """No interaction client -> interact() refuses."""
        client = _make_client()
        dialogue = Dialogue(role=DialogueRole(name='r'))
        dialogue.add_utterance('alice', 'hi', 1.0)
        assert client.interact(dialogue) is False

    def test_interact_empty_session_returns_false(self):
        """An empty session_utterances list short-circuits."""
        _, client, _ = self._client_with_ready_interaction()
        dialogue = Dialogue(role=DialogueRole(name='r'))
        # No add_utterance call -> session_utterances is empty
        assert client.interact(dialogue) is False

    def test_interact_packs_history(self):
        """interact() snapshots dialogue.session_utterances into the request."""
        manager, client, mock_srv = self._client_with_ready_interaction()
        dialogue = Dialogue(role=DialogueRole(name='__default__'))
        manager.add_dialogue(dialogue)
        dialogue.add_utterance('alice', 'hi there', 1.0)
        dialogue.add_utterance(ROBOT_SPEAKER_ID, 'hello!', 1.1)
        dialogue.add_utterance(SYSTEM_SPEAKER_ID, 'world update', 1.2)
        dialogue.add_utterance('alice', 'how are you?', 1.3)

        assert client.interact(dialogue) is True

        request = mock_srv.call_async.call_args[0][0]
        assert request.role.name == '__default__'
        assert bytes(request.dialogue_id.uuid) == dialogue.dialogue_id.bytes
        assert len(request.history) == 4
        # speaker mapping
        assert request.history[0].speaker == 'alice'
        assert request.history[1].speaker == Utterance.ASSISTANT
        assert request.history[2].speaker == Utterance.SYSTEM
        assert request.history[3].speaker == 'alice'
        # text preserved
        assert request.history[1].text == 'hello!'

    def test_interact_marks_waiting_state(self):
        """interact() flips dialogue state to WAITING_RESPONSE and the flag."""
        manager, client, _ = self._client_with_ready_interaction()
        dialogue = Dialogue(role=DialogueRole(name='r'))
        manager.add_dialogue(dialogue)
        dialogue.add_utterance('alice', 'hi', 1.0)

        assert client.waiting_for_response is False
        client.interact(dialogue)
        assert client.waiting_for_response is True
        assert dialogue.state == DialogueState.WAITING_RESPONSE


class TestChatbotClientOnResponse:
    """Tests for _on_response — the post-call pipeline."""

    def _setup(self):
        manager = DialogueManager()
        client = _make_client(dialogue_manager=manager)
        dialogue = Dialogue(role=DialogueRole(name='r'))
        manager.add_dialogue(dialogue)
        dialogue.state = DialogueState.WAITING_RESPONSE
        return manager, client, dialogue

    def _response(self, **kwargs):
        defaults = {
            'response': '',
            'intents': [],
            'dialogue_terminal': False,
            'results': '',
            'error_msg': '',
        }
        defaults.update(kwargs)
        r = MagicMock()
        r.response = defaults['response']
        r.intents = defaults['intents']
        r.dialogue_terminal = defaults['dialogue_terminal']
        r.results = defaults['results']
        r.error_msg = defaults['error_msg']
        return r

    def test_response_clears_waiting_state(self):
        """A successful response clears the waiting-for-response flag."""
        manager, client, dialogue = self._setup()
        client._waiting_for_response = True
        future = MagicMock()
        future.result.return_value = self._response(response='hi')

        client._on_response(future, dialogue.dialogue_id)
        assert client.waiting_for_response is False
        assert dialogue.state == DialogueState.ACTIVE

    def test_response_records_robot_utterance(self):
        """A non-empty response appends a ROBOT_SPEAKER_ID utterance."""
        _, client, dialogue = self._setup()
        future = MagicMock()
        future.result.return_value = self._response(response='hello user')

        client._on_response(future, dialogue.dialogue_id)
        assert dialogue.history[-1].speaker_id == ROBOT_SPEAKER_ID
        assert dialogue.history[-1].text == 'hello user'

    def test_terminal_sets_completed_and_captures_results(self):
        """dialogue_terminal=True flips state to COMPLETED and stores results."""
        _, client, dialogue = self._setup()
        future = MagicMock()
        future.result.return_value = self._response(
            response='thanks!',
            dialogue_terminal=True,
            results='{"age": 42}',
        )

        client._on_response(future, dialogue.dialogue_id)
        assert dialogue.state == DialogueState.COMPLETED
        assert dialogue.results == '{"age": 42}'

    def test_error_msg_is_logged_but_does_not_crash(self):
        """An error_msg in the response is handled without raising."""
        _, client, dialogue = self._setup()
        future = MagicMock()
        future.result.return_value = self._response(error_msg='LLM down')

        # Should return cleanly; the test just asserts no exception.
        client._on_response(future, dialogue.dialogue_id)


class TestChatbotClientDestroy:
    """Tests for destroy method."""

    def test_destroy_cleans_up_clients(self):
        """Destroy tears down both service clients."""
        mock_node = MagicMock()
        client = ChatbotClient(
            node=mock_node,
            dialogue_manager=MagicMock(),
            say_client=MagicMock(),
            intents_pub=MagicMock(),
            waiting_chatbot_pub=MagicMock(),
        )

        client._prepare_client = MagicMock()
        client._interaction_client = MagicMock()
        client.destroy()

        assert mock_node.destroy_client.call_count == 2
        assert client._prepare_client is None
        assert client._interaction_client is None


class TestChatbotClientIsAvailable:
    """Tests for is_available method."""

    def test_is_available_no_client(self):
        """is_available returns False when no client exists."""
        client = _make_client()
        assert client.is_available() is False

    def test_is_available_server_ready(self):
        """is_available returns True when interaction service is reachable."""
        client = _make_client()
        mock_srv = MagicMock()
        mock_srv.wait_for_service.return_value = True
        client._interaction_client = mock_srv
        assert client.is_available() is True

    def test_is_available_server_not_ready(self):
        """is_available returns False when interaction service times out."""
        client = _make_client()
        mock_srv = MagicMock()
        mock_srv.wait_for_service.return_value = False
        client._interaction_client = mock_srv
        assert client.is_available() is False
