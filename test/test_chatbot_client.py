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

from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

from chatbot_msgs.msg import DialogueRole

from dialogue_manager.chatbot_client import ChatbotClient, uuid_to_msg
from dialogue_manager.dialogue import Dialogue, DialogueManager, DialogueState


class TestUuidToMsg:
    """Tests for the uuid_to_msg helper function."""

    def test_uuid_to_msg_conversion(self):
        """Convert Python UUID to ROS UUID message."""
        test_uuid = uuid4()
        msg = uuid_to_msg(test_uuid)

        # The message should have the same bytes
        assert bytes(msg.uuid) == test_uuid.bytes

    def test_uuid_to_msg_preserves_value(self):
        """UUID round-trip preserves value."""
        test_uuid = uuid4()
        msg = uuid_to_msg(test_uuid)
        recovered = UUID(bytes=bytes(msg.uuid))

        assert recovered == test_uuid


class TestChatbotClientInit:
    """Tests for ChatbotClient initialization."""

    def test_init_sets_properties(self):
        """Client stores all provided dependencies."""
        mock_node = MagicMock()
        mock_dialogue_manager = MagicMock()
        mock_tts_client = MagicMock()
        mock_intents_pub = MagicMock()
        mock_waiting_pub = MagicMock()

        client = ChatbotClient(
            node=mock_node,
            dialogue_manager=mock_dialogue_manager,
            tts_client=mock_tts_client,
            intents_pub=mock_intents_pub,
            waiting_chatbot_pub=mock_waiting_pub
        )

        assert client._node is mock_node
        assert client._dialogue_manager is mock_dialogue_manager
        assert client._tts_client is mock_tts_client
        assert client._intents_pub is mock_intents_pub
        assert client._waiting_chatbot_pub is mock_waiting_pub

    def test_init_waiting_for_response_false(self):
        """Client starts not waiting for response."""
        client = ChatbotClient(
            node=MagicMock(),
            dialogue_manager=MagicMock(),
            tts_client=MagicMock(),
            intents_pub=MagicMock(),
            waiting_chatbot_pub=MagicMock()
        )

        assert client.waiting_for_response is False

    def test_init_default_dialogue_id_none(self):
        """Client starts with no default dialogue."""
        client = ChatbotClient(
            node=MagicMock(),
            dialogue_manager=MagicMock(),
            tts_client=MagicMock(),
            intents_pub=MagicMock(),
            waiting_chatbot_pub=MagicMock()
        )

        assert client.default_dialogue_id is None


class TestChatbotClientCreateClients:
    """Tests for creating chatbot clients."""

    def test_create_clients_creates_action_client(self):
        """create_clients creates dialogue action client."""
        mock_node = MagicMock()
        client = ChatbotClient(
            node=mock_node,
            dialogue_manager=MagicMock(),
            tts_client=MagicMock(),
            intents_pub=MagicMock(),
            waiting_chatbot_pub=MagicMock()
        )

        with patch('dialogue_manager.chatbot_client.ActionClient') as mock_ac:
            client.create_clients('chatbot')

        # Should create action client
        mock_ac.assert_called_once()
        assert client._dialogue_client is not None

    def test_create_clients_creates_service_client(self):
        """create_clients creates interaction service client."""
        mock_node = MagicMock()
        client = ChatbotClient(
            node=mock_node,
            dialogue_manager=MagicMock(),
            tts_client=MagicMock(),
            intents_pub=MagicMock(),
            waiting_chatbot_pub=MagicMock()
        )

        with patch('dialogue_manager.chatbot_client.ActionClient'):
            client.create_clients('chatbot')

        mock_node.create_client.assert_called_once()
        assert client._interaction_client is not None


class TestChatbotClientSendInput:
    """Tests for send_input method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_node = MagicMock()
        self.mock_dialogue_manager = DialogueManager()
        self.mock_tts_client = MagicMock()
        self.mock_intents_pub = MagicMock()
        self.mock_waiting_pub = MagicMock()

        self.client = ChatbotClient(
            node=self.mock_node,
            dialogue_manager=self.mock_dialogue_manager,
            tts_client=self.mock_tts_client,
            intents_pub=self.mock_intents_pub,
            waiting_chatbot_pub=self.mock_waiting_pub
        )

    def test_send_input_no_client_returns_false(self):
        """send_input returns False when no interaction client."""
        role = DialogueRole(name='test')
        dialogue = Dialogue(role=role)
        self.mock_dialogue_manager.add_dialogue(dialogue)

        result = self.client.send_input(dialogue.dialogue_id, 'user1', 'hello')

        assert result is False

    def test_send_input_unknown_dialogue_returns_false(self):
        """send_input returns False for unknown dialogue ID."""
        self.client._interaction_client = MagicMock()
        unknown_id = uuid4()

        result = self.client.send_input(unknown_id, 'user1', 'hello')

        assert result is False

    def test_send_input_no_chatbot_goal_id_returns_false(self):
        """send_input returns False when dialogue has no chatbot goal ID."""
        self.client._interaction_client = MagicMock()

        role = DialogueRole(name='test')
        dialogue = Dialogue(role=role)  # No chatbot_goal_id set
        self.mock_dialogue_manager.add_dialogue(dialogue)

        result = self.client.send_input(dialogue.dialogue_id, 'user1', 'hello')

        assert result is False

    def test_send_input_success(self):
        """send_input succeeds with valid dialogue and chatbot goal ID."""
        mock_interaction_client = MagicMock()
        mock_interaction_client.call_async.return_value = MagicMock()
        self.client._interaction_client = mock_interaction_client

        role = DialogueRole(name='test')
        chatbot_goal_id = uuid4()
        dialogue = Dialogue(role=role, chatbot_goal_id=chatbot_goal_id)
        self.mock_dialogue_manager.add_dialogue(dialogue)

        result = self.client.send_input(dialogue.dialogue_id, 'user1', 'hello')

        assert result is True
        assert self.client.waiting_for_response is True
        mock_interaction_client.call_async.assert_called_once()

    def test_send_input_sets_dialogue_state(self):
        """send_input sets dialogue state to WAITING_RESPONSE."""
        mock_interaction_client = MagicMock()
        mock_interaction_client.call_async.return_value = MagicMock()
        self.client._interaction_client = mock_interaction_client

        role = DialogueRole(name='test')
        chatbot_goal_id = uuid4()
        dialogue = Dialogue(role=role, chatbot_goal_id=chatbot_goal_id)
        self.mock_dialogue_manager.add_dialogue(dialogue)

        self.client.send_input(dialogue.dialogue_id, 'user1', 'hello')

        assert dialogue.state == DialogueState.WAITING_RESPONSE


class TestChatbotClientDestroy:
    """Tests for destroy method."""

    def test_destroy_cleans_up_clients(self):
        """Destroy cleans up action and service clients."""
        mock_node = MagicMock()
        client = ChatbotClient(
            node=mock_node,
            dialogue_manager=MagicMock(),
            tts_client=MagicMock(),
            intents_pub=MagicMock(),
            waiting_chatbot_pub=MagicMock()
        )

        mock_dialogue_client = MagicMock()
        mock_interaction_client = MagicMock()
        client._dialogue_client = mock_dialogue_client
        client._interaction_client = mock_interaction_client

        client.destroy()

        mock_dialogue_client.destroy.assert_called_once()
        mock_node.destroy_client.assert_called_once_with(mock_interaction_client)


class TestChatbotClientIsAvailable:
    """Tests for is_available method."""

    def test_is_available_no_client(self):
        """is_available returns False when no client exists."""
        client = ChatbotClient(
            node=MagicMock(),
            dialogue_manager=MagicMock(),
            tts_client=MagicMock(),
            intents_pub=MagicMock(),
            waiting_chatbot_pub=MagicMock()
        )

        assert client.is_available() is False

    def test_is_available_server_ready(self):
        """is_available returns True when server is ready."""
        client = ChatbotClient(
            node=MagicMock(),
            dialogue_manager=MagicMock(),
            tts_client=MagicMock(),
            intents_pub=MagicMock(),
            waiting_chatbot_pub=MagicMock()
        )

        mock_dialogue_client = MagicMock()
        mock_dialogue_client.wait_for_server.return_value = True
        client._dialogue_client = mock_dialogue_client

        assert client.is_available() is True

    def test_is_available_server_not_ready(self):
        """is_available returns False when server times out."""
        client = ChatbotClient(
            node=MagicMock(),
            dialogue_manager=MagicMock(),
            tts_client=MagicMock(),
            intents_pub=MagicMock(),
            waiting_chatbot_pub=MagicMock()
        )

        mock_dialogue_client = MagicMock()
        mock_dialogue_client.wait_for_server.return_value = False
        client._dialogue_client = mock_dialogue_client

        assert client.is_available() is False
