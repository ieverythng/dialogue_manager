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

"""Unit tests for say_client.py with mocked ROS2 dependencies."""

from unittest.mock import MagicMock, patch

from dialogue_manager.dialogue import DialogueManager
from dialogue_manager.say_client import CALLER_NAME, SayClient


class TestSayClientInit:
    """Tests for SayClient initialization."""

    def test_init_sets_properties(self):
        """Client stores all provided dependencies."""
        mock_node = MagicMock()
        mock_dialogue_manager = MagicMock()
        mock_captions_pub = MagicMock()
        mock_speech_pub = MagicMock()

        client = SayClient(
            node=mock_node,
            dialogue_manager=mock_dialogue_manager,
            closed_captions_pub=mock_captions_pub,
            robot_speech_pub=mock_speech_pub
        )

        assert client._node is mock_node
        assert client._dialogue_manager is mock_dialogue_manager
        assert client._closed_captions_pub is mock_captions_pub
        assert client._robot_speech_pub is mock_speech_pub

    def test_init_no_say_client_initially(self):
        """Client starts without Say action client."""
        client = SayClient(
            node=MagicMock(),
            dialogue_manager=MagicMock(),
            closed_captions_pub=MagicMock(),
            robot_speech_pub=MagicMock()
        )

        assert client._say_client is None


class TestSayClientCreateClient:
    """Tests for create_client method."""

    def test_create_client_creates_action_client(self):
        """create_client creates Say action client."""
        mock_node = MagicMock()
        client = SayClient(
            node=mock_node,
            dialogue_manager=MagicMock(),
            closed_captions_pub=MagicMock(),
            robot_speech_pub=MagicMock()
        )

        with patch('dialogue_manager.say_client.ActionClient') as mock_ac:
            client.create_client('/custom_say')

        mock_ac.assert_called_once()
        assert client._say_client is not None


class TestSayClientIsAvailable:
    """Tests for is_available method."""

    def test_is_available_no_client(self):
        """is_available returns False when no client exists."""
        client = SayClient(
            node=MagicMock(),
            dialogue_manager=MagicMock(),
            closed_captions_pub=MagicMock(),
            robot_speech_pub=MagicMock()
        )

        assert client.is_available() is False

    def test_is_available_server_ready(self):
        """is_available returns True when server is ready."""
        client = SayClient(
            node=MagicMock(),
            dialogue_manager=MagicMock(),
            closed_captions_pub=MagicMock(),
            robot_speech_pub=MagicMock()
        )

        mock_say_client = MagicMock()
        mock_say_client.wait_for_server.return_value = True
        client._say_client = mock_say_client

        assert client.is_available() is True

    def test_is_available_server_not_ready(self):
        """is_available returns False when server times out."""
        client = SayClient(
            node=MagicMock(),
            dialogue_manager=MagicMock(),
            closed_captions_pub=MagicMock(),
            robot_speech_pub=MagicMock()
        )

        mock_say_client = MagicMock()
        mock_say_client.wait_for_server.return_value = False
        client._say_client = mock_say_client

        assert client.is_available() is False


class TestSayClientSpeak:
    """Tests for speak method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_node = MagicMock()
        self.mock_dialogue_manager = DialogueManager()
        self.mock_captions_pub = MagicMock()
        self.mock_speech_pub = MagicMock()

        self.client = SayClient(
            node=self.mock_node,
            dialogue_manager=self.mock_dialogue_manager,
            closed_captions_pub=self.mock_captions_pub,
            robot_speech_pub=self.mock_speech_pub
        )

    def test_speak_no_client_returns_false(self):
        """Speak returns False when no Say client."""
        result = self.client.speak('Hello world')
        assert result is False

    def test_speak_server_not_available_returns_false(self):
        """Speak returns False when server not available."""
        mock_say_client = MagicMock()
        mock_say_client.wait_for_server.return_value = False
        self.client._say_client = mock_say_client

        result = self.client.speak('Hello world')

        assert result is False

    def test_speak_success_returns_true(self):
        """Speak returns True on success."""
        mock_say_client = MagicMock()
        mock_say_client.wait_for_server.return_value = True
        mock_say_client.send_goal_async.return_value = MagicMock()
        self.client._say_client = mock_say_client

        result = self.client.speak('Hello world')

        assert result is True

    def test_speak_rejects_duplicate_normalized_utterance(self):
        """The speaking owner dispatches the same utterance at most once."""
        mock_say_client = MagicMock()
        mock_say_client.wait_for_server.return_value = True
        mock_say_client.send_goal_async.return_value = MagicMock()
        self.client._say_client = mock_say_client

        assert self.client.speak('Hello   world') is True
        assert self.client.speak(' hello world ') is False
        mock_say_client.send_goal_async.assert_called_once()

    def test_speak_sets_expression_priority(self):
        """Speak sets expression priority in dialogue manager."""
        mock_say_client = MagicMock()
        mock_say_client.wait_for_server.return_value = True
        mock_say_client.send_goal_async.return_value = MagicMock()
        self.client._say_client = mock_say_client

        self.client.speak('Hello', priority=175)

        assert self.mock_dialogue_manager.current_max_priority == 175

    def test_speak_publishes_closed_caption(self):
        """Speak publishes closed caption."""
        mock_say_client = MagicMock()
        mock_say_client.wait_for_server.return_value = True
        mock_say_client.send_goal_async.return_value = MagicMock()
        self.client._say_client = mock_say_client

        self.client.speak('Hello world')

        self.mock_captions_pub.publish.assert_called_once()
        caption = self.mock_captions_pub.publish.call_args[0][0]
        assert caption.text == 'Hello world'

    def test_speak_sends_goal_with_meta_and_input(self):
        """Speak builds Say.Goal with meta, input, and forwards person/group."""
        mock_say_client = MagicMock()
        mock_say_client.wait_for_server.return_value = True
        mock_future = MagicMock()
        mock_say_client.send_goal_async.return_value = mock_future
        self.client._say_client = mock_say_client

        self.client.speak(
            'Hello world',
            priority=200,
            person_id='alice',
            group_id='',
        )

        mock_say_client.send_goal_async.assert_called_once()
        goal = mock_say_client.send_goal_async.call_args[0][0]
        assert goal.input == 'Hello world'
        assert goal.meta.priority == 200
        assert goal.meta.caller == CALLER_NAME
        assert goal.person_id == 'alice'
        assert goal.group_id == ''


class TestSayClientDestroy:
    """Tests for destroy method."""

    def test_destroy_cleans_up_client(self):
        """Destroy cleans up Say action client."""
        client = SayClient(
            node=MagicMock(),
            dialogue_manager=MagicMock(),
            closed_captions_pub=MagicMock(),
            robot_speech_pub=MagicMock()
        )

        mock_say_client = MagicMock()
        client._say_client = mock_say_client

        client.destroy()

        mock_say_client.destroy.assert_called_once()
        assert client._say_client is None

    def test_destroy_no_client_no_error(self):
        """Destroy with no client does not error."""
        client = SayClient(
            node=MagicMock(),
            dialogue_manager=MagicMock(),
            closed_captions_pub=MagicMock(),
            robot_speech_pub=MagicMock()
        )

        # Should not raise
        client.destroy()


class TestSayClientCallbacks:
    """Tests for Say callback methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_node = MagicMock()
        self.mock_dialogue_manager = DialogueManager()
        self.mock_speech_pub = MagicMock()

        self.client = SayClient(
            node=self.mock_node,
            dialogue_manager=self.mock_dialogue_manager,
            closed_captions_pub=MagicMock(),
            robot_speech_pub=self.mock_speech_pub
        )

    def test_on_feedback_publishes_word_from_data_str(self):
        """_on_feedback publishes Feedback.data_str to robot_speech."""
        mock_feedback = MagicMock()
        mock_feedback.feedback.feedback.data_str = 'hello'

        self.client._on_feedback(mock_feedback)

        self.mock_speech_pub.publish.assert_called_once()
        msg = self.mock_speech_pub.publish.call_args[0][0]
        assert msg.data == 'hello'

    def test_on_feedback_empty_data_str_does_not_publish(self):
        """_on_feedback with empty data_str publishes nothing."""
        mock_feedback = MagicMock()
        mock_feedback.feedback.feedback.data_str = ''

        self.client._on_feedback(mock_feedback)

        self.mock_speech_pub.publish.assert_not_called()

    def test_on_result_clears_expression_priority(self):
        """_on_result clears expression priority."""
        self.mock_dialogue_manager.set_expression_priority(100)

        mock_future = MagicMock()
        mock_future.result.return_value.result.result.error_msg = ''

        self.client._on_result(mock_future)

        assert self.mock_dialogue_manager.current_max_priority == -1

    def test_on_result_invokes_complete_callback(self):
        """_on_result invokes completion callback if set."""
        callback_called = []

        def on_complete():
            callback_called.append(True)

        self.client._on_complete_callback = on_complete

        mock_future = MagicMock()
        mock_future.result.return_value.result.result.error_msg = ''

        self.client._on_result(mock_future)

        assert len(callback_called) == 1

    def test_invoke_complete_callback_clears_callback(self):
        """_invoke_complete_callback clears the callback after invoking."""
        self.client._on_complete_callback = MagicMock()

        self.client._invoke_complete_callback()

        assert self.client._on_complete_callback is None
