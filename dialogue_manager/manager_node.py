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

"""Main Dialogue Manager ROS2 node."""

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from hri_actions_msgs.msg import ClosedCaption, Intent
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.lifecycle import LifecycleNode, State, TransitionCallbackReturn
from std_msgs.msg import Bool, String

from .chatbot_client import ChatbotClient
from .dialogue import DialogueManager
from .skill_servers import SkillServers
from .speech_handler import SpeechHandler
from .tts_client import TTSClient


class DialogueManagerNode(LifecycleNode):
    """
    Dialogue Manager ROS2 Lifecycle Node.

    Implements the chat, ask, and say skills for human-robot dialogue.
    Orchestrates chatbot, TTS, and speech input handling through
    composition of specialized handler classes.
    """

    def __init__(self) -> None:
        """Construct the node."""
        super().__init__('dialogue_manager')

        # Callback group for concurrent action handling
        self._callback_group = ReentrantCallbackGroup()

        # Core dialogue tracking
        self._dialogue_manager = DialogueManager()

        # Handlers (created in on_configure)
        self._tts_client: TTSClient | None = None
        self._chatbot_client: ChatbotClient | None = None
        self._speech_handler: SpeechHandler | None = None
        self._skill_servers: SkillServers | None = None

        # Publishers (created in on_configure)
        self._closed_captions_pub = None
        self._robot_speech_pub = None
        self._waiting_chatbot_pub = None
        self._intents_pub = None
        self._diag_pub = None

        # Timers
        self._diag_timer = None

        self._declare_parameters()
        self.get_logger().info('Dialogue Manager node created, awaiting configuration.')

    def _declare_parameters(self) -> None:
        """Declare all ROS parameters."""
        self.declare_parameter(
            'chatbot', 'chatbot',
            ParameterDescriptor(description='Chatbot node FQN for action/service prefix')
        )
        self.declare_parameter(
            'enable_default_chat', False,
            ParameterDescriptor(description='Enable default chat while active')
        )
        self.declare_parameter(
            'default_chat_role', '__default__',
            ParameterDescriptor(description='Role for default chat')
        )
        self.declare_parameter(
            'default_chat_configuration', '',
            ParameterDescriptor(description='Configuration for default chat')
        )
        self.declare_parameter(
            'chatbot_startup_timeout', 30.0,
            ParameterDescriptor(description='Max wait for chatbot startup (s)')
        )
        self.declare_parameter(
            'chatbot_response_timeout', 5.0,
            ParameterDescriptor(description='Max wait for chatbot response (s)')
        )
        self.declare_parameter(
            'multi_modal_expression_timeout', 60.0,
            ParameterDescriptor(description='Max duration for expression (s)')
        )
        self.declare_parameter(
            'markup_action_timeout', 10.0,
            ParameterDescriptor(description='Default max time for markup action (s)')
        )
        self.declare_parameter(
            'markup_libraries', ['config/00-default_markup_libraries.json'],
            ParameterDescriptor(description='Paths to markup action definitions')
        )
        self.declare_parameter(
            'disabled_markup_actions', ['motion'],
            ParameterDescriptor(description='Markup actions to skip')
        )

    # =========================================================================
    # Lifecycle callbacks
    # =========================================================================

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        """Configure the node: create publishers and handlers."""
        self.get_logger().info('Configuring Dialogue Manager...')

        # Create publishers
        self._closed_captions_pub = self.create_publisher(
            ClosedCaption, '~/closed_captions', 10
        )
        self._robot_speech_pub = self.create_publisher(
            String, '~/robot_speech', 10
        )
        self._waiting_chatbot_pub = self.create_publisher(
            Bool, '~/currently_waiting_for_chatbot_response', 10
        )
        self._intents_pub = self.create_publisher(Intent, '/intents', 10)
        self._diag_pub = self.create_publisher(DiagnosticArray, '/diagnostics', 10)

        # Create TTS client
        self._tts_client = TTSClient(
            node=self,
            dialogue_manager=self._dialogue_manager,
            closed_captions_pub=self._closed_captions_pub,
            robot_speech_pub=self._robot_speech_pub,
            callback_group=self._callback_group
        )
        self._tts_client.create_client()
        self.get_logger().debug('[CONFIGURE] TTS client created')

        # Create chatbot client (if configured)
        chatbot = self.get_parameter('chatbot').get_parameter_value().string_value
        if chatbot:
            self._chatbot_client = ChatbotClient(
                node=self,
                dialogue_manager=self._dialogue_manager,
                tts_client=self._tts_client,
                intents_pub=self._intents_pub,
                waiting_chatbot_pub=self._waiting_chatbot_pub,
                callback_group=self._callback_group
            )
            self._chatbot_client.create_clients(chatbot)
            self.get_logger().debug(f'[CONFIGURE] Chatbot client created for "{chatbot}"')
        else:
            self.get_logger().info('[CONFIGURE] No chatbot configured')

        # Create speech handler
        self._speech_handler = SpeechHandler(
            node=self,
            dialogue_manager=self._dialogue_manager,
            chatbot_client=self._chatbot_client,
            closed_captions_pub=self._closed_captions_pub,
            intents_pub=self._intents_pub
        )
        self._speech_handler.set_chatbot_enabled(chatbot != '')
        self.get_logger().debug('[CONFIGURE] Speech handler created')

        # Create skill servers
        self._skill_servers = SkillServers(
            node=self,
            dialogue_manager=self._dialogue_manager,
            chatbot_client=self._chatbot_client,
            tts_client=self._tts_client,
            closed_captions_pub=self._closed_captions_pub,
            callback_group=self._callback_group
        )
        self._skill_servers.create_servers()
        self.get_logger().debug('[CONFIGURE] Skill servers created')

        # Diagnostics timer
        self._diag_timer = self.create_timer(1.0, self._publish_diagnostics)

        self.get_logger().info('Dialogue Manager configured.')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        """Activate the node: subscribe to speech, optionally start default chat."""
        self.get_logger().info('Activating Dialogue Manager...')

        # Enable skill servers
        self._skill_servers.set_active(True)

        # Subscribe to voices
        self._speech_handler.subscribe_to_voices()

        # Start default chat if enabled
        if self.get_parameter('enable_default_chat').get_parameter_value().bool_value:
            if self._chatbot_client:
                role = self.get_parameter('default_chat_role').get_parameter_value().string_value
                config = self.get_parameter(
                    'default_chat_configuration'
                ).get_parameter_value().string_value
                self._chatbot_client.start_default_chat(role, config)
            else:
                self.get_logger().warn('[ACTIVATE] Default chat enabled but no chatbot configured')

        self.get_logger().info('Dialogue Manager activated.')
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        """Deactivate the node: unsubscribe, disable servers."""
        self.get_logger().info('Deactivating Dialogue Manager...')

        # Disable skill servers
        self._skill_servers.set_active(False)

        # Unsubscribe from voices
        self._speech_handler.unsubscribe_all()

        # Cancel active dialogues
        self._dialogue_manager.clear_all()

        self.get_logger().info('Dialogue Manager deactivated.')
        return super().on_deactivate(state)

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        """Shutdown the node: destroy all handlers."""
        self.get_logger().info('Shutting down Dialogue Manager...')

        if self._diag_timer:
            self.destroy_timer(self._diag_timer)

        if self._skill_servers:
            self._skill_servers.destroy()
        if self._tts_client:
            self._tts_client.destroy()
        if self._chatbot_client:
            self._chatbot_client.destroy()

        self.get_logger().info('Dialogue Manager shutdown complete.')
        return TransitionCallbackReturn.SUCCESS

    # =========================================================================
    # Diagnostics
    # =========================================================================

    def _publish_diagnostics(self) -> None:
        """Publish diagnostic information."""
        arr = DiagnosticArray()
        status = DiagnosticStatus(
            level=DiagnosticStatus.OK,
            name='/dialogue_manager',
            message='Dialogue Manager running',
            values=[
                KeyValue(
                    key='Active dialogues',
                    value=str(len(self._dialogue_manager.active_dialogues))
                ),
                KeyValue(key='Chatbot configured', value=str(self._chatbot_client is not None)),
            ]
        )

        if self._chatbot_client:
            status.values.append(
                KeyValue(
                    key='Waiting for chatbot',
                    value=str(self._chatbot_client.waiting_for_response)
                )
            )

        arr.header.stamp = self.get_clock().now().to_msg()
        arr.status = [status]
        self._diag_pub.publish(arr)
