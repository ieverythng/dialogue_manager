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

"""Unit tests for skill_servers.py with mocked ROS2 dependencies."""

from unittest.mock import MagicMock, patch

from dialogue_manager.dialogue import DialogueManager
from dialogue_manager.skill_servers import SkillServers


class TestSkillServersInit:
    """Tests for SkillServers initialization."""

    def test_init_sets_properties(self):
        """Servers store all provided dependencies."""
        mock_node = MagicMock()
        mock_dialogue_manager = MagicMock()
        mock_chatbot_client = MagicMock()
        mock_tts_client = MagicMock()
        mock_captions_pub = MagicMock()

        servers = SkillServers(
            node=mock_node,
            dialogue_manager=mock_dialogue_manager,
            chatbot_client=mock_chatbot_client,
            tts_client=mock_tts_client,
            closed_captions_pub=mock_captions_pub
        )

        assert servers._node is mock_node
        assert servers._dialogue_manager is mock_dialogue_manager
        assert servers._chatbot_client is mock_chatbot_client
        assert servers._tts_client is mock_tts_client
        assert servers._closed_captions_pub is mock_captions_pub

    def test_init_starts_inactive(self):
        """Servers start in inactive state."""
        servers = SkillServers(
            node=MagicMock(),
            dialogue_manager=MagicMock(),
            chatbot_client=MagicMock(),
            tts_client=MagicMock(),
            closed_captions_pub=MagicMock()
        )

        assert servers._is_active is False


class TestSkillServersSetActive:
    """Tests for set_active method."""

    def test_set_active_true(self):
        """set_active sets to True."""
        servers = SkillServers(
            node=MagicMock(),
            dialogue_manager=MagicMock(),
            chatbot_client=MagicMock(),
            tts_client=MagicMock(),
            closed_captions_pub=MagicMock()
        )

        servers.set_active(True)

        assert servers._is_active is True

    def test_set_active_false(self):
        """set_active sets to False."""
        servers = SkillServers(
            node=MagicMock(),
            dialogue_manager=MagicMock(),
            chatbot_client=MagicMock(),
            tts_client=MagicMock(),
            closed_captions_pub=MagicMock()
        )
        servers._is_active = True

        servers.set_active(False)

        assert servers._is_active is False


class TestSkillServersCreateServers:
    """Tests for create_servers method."""

    def test_create_servers_creates_all(self):
        """create_servers creates Chat, Ask, and Say action servers."""
        mock_node = MagicMock()
        servers = SkillServers(
            node=mock_node,
            dialogue_manager=MagicMock(),
            chatbot_client=MagicMock(),
            tts_client=MagicMock(),
            closed_captions_pub=MagicMock()
        )

        with patch('dialogue_manager.skill_servers.ActionServer') as mock_as:
            servers.create_servers()

        # Should create 3 action servers
        assert mock_as.call_count == 3


class TestSkillServersDestroy:
    """Tests for destroy method."""

    def test_destroy_cleans_up_servers(self):
        """Destroy cleans up all action servers."""
        servers = SkillServers(
            node=MagicMock(),
            dialogue_manager=MagicMock(),
            chatbot_client=MagicMock(),
            tts_client=MagicMock(),
            closed_captions_pub=MagicMock()
        )

        mock_chat_server = MagicMock()
        mock_ask_server = MagicMock()
        mock_say_server = MagicMock()

        servers._chat_server = mock_chat_server
        servers._ask_server = mock_ask_server
        servers._say_server = mock_say_server

        servers.destroy()

        mock_chat_server.destroy.assert_called_once()
        mock_ask_server.destroy.assert_called_once()
        mock_say_server.destroy.assert_called_once()


class TestSkillServersGoalCallbacks:
    """Tests for goal acceptance callbacks."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_node = MagicMock()
        self.mock_dialogue_manager = DialogueManager()
        self.mock_chatbot_client = MagicMock()
        self.mock_tts_client = MagicMock()

        self.servers = SkillServers(
            node=self.mock_node,
            dialogue_manager=self.mock_dialogue_manager,
            chatbot_client=self.mock_chatbot_client,
            tts_client=self.mock_tts_client,
            closed_captions_pub=MagicMock()
        )

    def test_chat_goal_rejected_when_inactive(self):
        """Chat goal rejected when server is inactive."""
        mock_request = MagicMock()
        mock_request.meta.priority = 128

        from rclpy.action import GoalResponse
        result = self.servers._chat_goal_callback(mock_request)

        assert result == GoalResponse.REJECT

    def test_chat_goal_accepted_when_active_with_higher_priority(self):
        """Chat goal accepted when active with higher priority."""
        self.servers._is_active = True

        mock_request = MagicMock()
        mock_request.meta.priority = 128

        from rclpy.action import GoalResponse
        result = self.servers._chat_goal_callback(mock_request)

        assert result == GoalResponse.ACCEPT

    def test_ask_goal_rejected_when_inactive(self):
        """Ask goal rejected when server is inactive."""
        mock_request = MagicMock()
        mock_request.meta.priority = 128

        from rclpy.action import GoalResponse
        result = self.servers._ask_goal_callback(mock_request)

        assert result == GoalResponse.REJECT

    def test_ask_goal_accepted_when_active_with_higher_priority(self):
        """Ask goal accepted when active with higher priority."""
        self.servers._is_active = True

        mock_request = MagicMock()
        mock_request.meta.priority = 128

        from rclpy.action import GoalResponse
        result = self.servers._ask_goal_callback(mock_request)

        assert result == GoalResponse.ACCEPT

    def test_say_goal_rejected_when_inactive(self):
        """Say goal rejected when server is inactive."""
        mock_request = MagicMock()
        mock_request.meta.priority = 128

        from rclpy.action import GoalResponse
        result = self.servers._say_goal_callback(mock_request)

        assert result == GoalResponse.REJECT

    def test_say_goal_accepted_when_active_with_higher_priority(self):
        """Say goal accepted when active with higher priority."""
        self.servers._is_active = True

        mock_request = MagicMock()
        mock_request.meta.priority = 128

        from rclpy.action import GoalResponse
        result = self.servers._say_goal_callback(mock_request)

        assert result == GoalResponse.ACCEPT


class TestSkillServersPriorityRejection:
    """Tests for priority-based goal rejection."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_dialogue_manager = DialogueManager()

        self.servers = SkillServers(
            node=MagicMock(),
            dialogue_manager=self.mock_dialogue_manager,
            chatbot_client=MagicMock(),
            tts_client=MagicMock(),
            closed_captions_pub=MagicMock()
        )
        self.servers._is_active = True

    def test_chat_goal_rejected_lower_priority(self):
        """Chat goal rejected when priority is too low."""
        # Set current expression priority
        self.mock_dialogue_manager.set_expression_priority(200)

        mock_request = MagicMock()
        mock_request.meta.priority = 100  # Lower than 200

        from rclpy.action import GoalResponse
        result = self.servers._chat_goal_callback(mock_request)

        assert result == GoalResponse.REJECT

    def test_ask_goal_rejected_lower_priority(self):
        """Ask goal rejected when priority is too low."""
        self.mock_dialogue_manager.set_expression_priority(200)

        mock_request = MagicMock()
        mock_request.meta.priority = 100

        from rclpy.action import GoalResponse
        result = self.servers._ask_goal_callback(mock_request)

        assert result == GoalResponse.REJECT

    def test_say_goal_rejected_lower_priority(self):
        """Say goal rejected when priority is too low."""
        self.mock_dialogue_manager.set_expression_priority(200)

        mock_request = MagicMock()
        mock_request.meta.priority = 100

        from rclpy.action import GoalResponse
        result = self.servers._say_goal_callback(mock_request)

        assert result == GoalResponse.REJECT


class TestSkillServersCancelCallback:
    """Tests for cancel callback."""

    def test_cancel_callback_accepts(self):
        """Cancel callback accepts cancellation requests."""
        servers = SkillServers(
            node=MagicMock(),
            dialogue_manager=MagicMock(),
            chatbot_client=MagicMock(),
            tts_client=MagicMock(),
            closed_captions_pub=MagicMock()
        )

        mock_goal_handle = MagicMock()

        from rclpy.action import CancelResponse
        result = servers._cancel_callback(mock_goal_handle)

        assert result == CancelResponse.ACCEPT
