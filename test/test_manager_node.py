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

"""Integration tests for manager_node.py lifecycle node."""

from dialogue_manager.manager_node import DialogueManagerNode
import pytest
import rclpy
from rclpy.lifecycle import TransitionCallbackReturn


@pytest.fixture(scope='module')
def rclpy_context():
    """Initialize rclpy for the test module."""
    rclpy.init()
    yield
    rclpy.shutdown()


class TestDialogueManagerNodeInit:
    """Tests for DialogueManagerNode initialization."""

    def test_node_creation(self, rclpy_context):
        """Node can be created."""
        node = DialogueManagerNode()

        assert node.get_name() == 'dialogue_manager'

        node.destroy_node()

    def test_parameters_declared(self, rclpy_context):
        """All parameters are declared on creation."""
        node = DialogueManagerNode()

        # Check core parameters exist
        assert node.has_parameter('chatbot')
        assert node.has_parameter('enable_default_chat')
        assert node.has_parameter('default_chat_role')
        assert node.has_parameter('default_chat_configuration')

        node.destroy_node()

    def test_default_parameter_values(self, rclpy_context):
        """Default parameter values are correct."""
        node = DialogueManagerNode()

        assert node.get_parameter('chatbot').value == 'chatbot'
        assert node.get_parameter('enable_default_chat').value is False

        node.destroy_node()


class TestDialogueManagerNodeConfigure:
    """Tests for on_configure lifecycle callback."""

    def test_configure_returns_success(self, rclpy_context):
        """on_configure returns SUCCESS."""
        node = DialogueManagerNode()

        from lifecycle_msgs.msg import State
        state = State()

        result = node.on_configure(state)

        assert result == TransitionCallbackReturn.SUCCESS

        node.destroy_node()

    def test_configure_creates_publishers(self, rclpy_context):
        """on_configure creates all publishers."""
        node = DialogueManagerNode()

        from lifecycle_msgs.msg import State
        state = State()
        node.on_configure(state)

        # Check key publishers exist
        assert node._diag_pub is not None
        assert node._robot_speech_pub is not None
        assert node._closed_captions_pub is not None
        assert node._intents_pub is not None
        assert node._waiting_chatbot_pub is not None

        node.destroy_node()

    def test_configure_creates_components(self, rclpy_context):
        """on_configure creates all component handlers."""
        node = DialogueManagerNode()

        from lifecycle_msgs.msg import State
        state = State()
        node.on_configure(state)

        assert node._dialogue_manager is not None
        assert node._tts_client is not None
        assert node._chatbot_client is not None
        assert node._speech_handler is not None
        assert node._skill_servers is not None

        node.destroy_node()


class TestDialogueManagerNodeActivate:
    """Tests for on_activate lifecycle callback."""

    def test_activate_returns_success(self, rclpy_context):
        """on_activate returns SUCCESS."""
        node = DialogueManagerNode()

        from lifecycle_msgs.msg import State
        state = State()

        # Must configure first
        node.on_configure(state)
        result = node.on_activate(state)

        assert result == TransitionCallbackReturn.SUCCESS

        node.destroy_node()

    def test_activate_enables_servers(self, rclpy_context):
        """on_activate enables skill servers."""
        node = DialogueManagerNode()

        from lifecycle_msgs.msg import State
        state = State()
        node.on_configure(state)
        node.on_activate(state)

        assert node._skill_servers._is_active is True

        node.destroy_node()


class TestDialogueManagerNodeDeactivate:
    """Tests for on_deactivate lifecycle callback."""

    def test_deactivate_returns_success(self, rclpy_context):
        """on_deactivate returns SUCCESS."""
        node = DialogueManagerNode()

        from lifecycle_msgs.msg import State
        state = State()
        node.on_configure(state)
        node.on_activate(state)
        result = node.on_deactivate(state)

        assert result == TransitionCallbackReturn.SUCCESS

        node.destroy_node()

    def test_deactivate_disables_servers(self, rclpy_context):
        """on_deactivate disables skill servers."""
        node = DialogueManagerNode()

        from lifecycle_msgs.msg import State
        state = State()
        node.on_configure(state)
        node.on_activate(state)
        node.on_deactivate(state)

        assert node._skill_servers._is_active is False

        node.destroy_node()


class TestDialogueManagerNodeShutdown:
    """Tests for on_shutdown lifecycle callback."""

    def test_shutdown_returns_success(self, rclpy_context):
        """on_shutdown returns SUCCESS."""
        node = DialogueManagerNode()

        from lifecycle_msgs.msg import State
        state = State()
        node.on_configure(state)
        result = node.on_shutdown(state)

        assert result == TransitionCallbackReturn.SUCCESS

        node.destroy_node()


class TestDialogueManagerNodeFullLifecycle:
    """Tests for complete lifecycle transitions."""

    def test_full_lifecycle(self, rclpy_context):
        """Complete lifecycle: configure -> activate -> deactivate -> shutdown."""
        node = DialogueManagerNode()

        from lifecycle_msgs.msg import State
        state = State()

        # Unconfigured -> Inactive
        result = node.on_configure(state)
        assert result == TransitionCallbackReturn.SUCCESS

        # Inactive -> Active
        result = node.on_activate(state)
        assert result == TransitionCallbackReturn.SUCCESS

        # Active -> Inactive
        result = node.on_deactivate(state)
        assert result == TransitionCallbackReturn.SUCCESS

        # Inactive -> Finalized
        result = node.on_shutdown(state)
        assert result == TransitionCallbackReturn.SUCCESS

        node.destroy_node()

    def test_reactivation(self, rclpy_context):
        """Node can be reactivated after deactivation."""
        node = DialogueManagerNode()

        from lifecycle_msgs.msg import State
        state = State()

        node.on_configure(state)
        node.on_activate(state)
        node.on_deactivate(state)

        # Re-activate
        result = node.on_activate(state)
        assert result == TransitionCallbackReturn.SUCCESS
        assert node._skill_servers._is_active is True

        node.destroy_node()
