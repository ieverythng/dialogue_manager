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

"""Dialogue Manager ROS2 Node - implements chat, ask, and say skills."""

import json
import re
from uuid import UUID

from rclpy.lifecycle import Node, State, TransitionCallbackReturn
from rclpy.action import ActionServer, ActionClient, GoalResponse, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, DurabilityPolicy
from rcl_interfaces.msg import ParameterDescriptor

from unique_identifier_msgs.msg import UUID as UUIDMsg

# Message imports
from std_msgs.msg import String, Bool
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from hri_msgs.msg import IdsList, LiveSpeech
from hri_actions_msgs.msg import Intent, ClosedCaption

# Action imports
from communication_skills.action import Chat, Ask, Say
from chatbot_msgs.action import Dialogue as DialogueAction
from chatbot_msgs.msg import DialogueRole
from chatbot_msgs.srv import DialogueInteraction
from tts_msgs.action import TTS

from .dialogue import Dialogue, DialogueManager, DialogueState


def uuid_to_msg(uuid: UUID) -> UUIDMsg:
    """Convert a Python UUID to a ROS UUID message."""
    msg = UUIDMsg()
    msg.uuid = list(uuid.bytes)
    return msg


def msg_to_uuid(msg: UUIDMsg) -> UUID:
    """Convert a ROS UUID message to a Python UUID."""
    return UUID(bytes=bytes(msg.uuid))


class DialogueManagerNode(Node):
    """
    Dialogue Manager ROS2 Lifecycle Node.

    Implements the chat, ask, and say skills for human-robot dialogue.
    Acts as a bridge between the mission controller, chatbot engine, and TTS.
    """

    def __init__(self) -> None:
        """Construct the node."""
        super().__init__('dialogue_manager')

        # Callback group for concurrent action handling
        self._action_cb_group = ReentrantCallbackGroup()

        # Dialogue tracking
        self._dialogue_manager = DialogueManager()
        self._default_dialogue_id: UUID | None = None

        # Tracked voices (for dynamic subscription)
        self._tracked_voices: set[str] = set()
        self._voice_subscriptions: dict[str, object] = {}

        # State flags
        self._waiting_for_chatbot = False

        # ROS interfaces (created in on_configure)
        self._chat_server = None
        self._ask_server = None
        self._say_server = None
        self._dialogue_client = None
        self._tts_client = None
        self._dialogue_interaction_client = None

        # Publishers
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
        """Configure the node: create publishers, subscribers, action servers/clients."""
        self.get_logger().info('Configuring Dialogue Manager...')

        chatbot = self.get_parameter('chatbot').get_parameter_value().string_value

        # Create action servers (exist in configured state, reject goals until active)
        self._chat_server = ActionServer(
            self, Chat, '/skill/chat',
            goal_callback=self._chat_goal_callback,
            cancel_callback=self._cancel_callback,
            execute_callback=self._execute_chat,
            callback_group=self._action_cb_group
        )
        self._ask_server = ActionServer(
            self, Ask, '/skill/ask',
            goal_callback=self._ask_goal_callback,
            cancel_callback=self._cancel_callback,
            execute_callback=self._execute_ask,
            callback_group=self._action_cb_group
        )
        self._say_server = ActionServer(
            self, Say, '/skill/say',
            goal_callback=self._say_goal_callback,
            cancel_callback=self._cancel_callback,
            execute_callback=self._execute_say,
            callback_group=self._action_cb_group
        )

        # Create action clients
        if chatbot:
            self._dialogue_client = ActionClient(
                self, DialogueAction, f'{chatbot}/start_dialogue',
                callback_group=self._action_cb_group
            )
            self._dialogue_interaction_client = self.create_client(
                DialogueInteraction, f'{chatbot}/dialogue_interaction',
                callback_group=self._action_cb_group
            )

        self._tts_client = ActionClient(
            self, TTS, 'tts_engine/tts',
            callback_group=self._action_cb_group
        )

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
        self._intents_pub = self.create_publisher(
            Intent, '/intents', 10
        )
        self._diag_pub = self.create_publisher(
            DiagnosticArray, '/diagnostics', 1
        )

        # Start diagnostics (published in configured state too)
        self._diag_timer = self.create_timer(1.0, self._publish_diagnostics)

        self.get_logger().info('Dialogue Manager configured.')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        """Activate the node: subscribe to speech topics, optionally start default chat."""
        self.get_logger().info('Activating Dialogue Manager...')

        # Subscribe to tracked voices
        qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._voices_sub = self.create_subscription(
            IdsList, '/humans/voices/tracked',
            self._on_voices_tracked, qos
        )

        # Start default chat if enabled
        enable_default = self.get_parameter('enable_default_chat').get_parameter_value().bool_value
        if enable_default:
            self._start_default_chat()

        self.get_logger().info('Dialogue Manager active.')
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        """Deactivate the node: unsubscribe, cancel active dialogues."""
        self.get_logger().info('Deactivating Dialogue Manager...')

        # Clear voice subscriptions
        for sub in self._voice_subscriptions.values():
            self.destroy_subscription(sub)
        self._voice_subscriptions.clear()
        self._tracked_voices.clear()

        if hasattr(self, '_voices_sub'):
            self.destroy_subscription(self._voices_sub)

        self.get_logger().info('Dialogue Manager deactivated.')
        return super().on_deactivate(state)

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        """Shutdown the node: destroy all interfaces."""
        self.get_logger().info('Shutting down Dialogue Manager...')

        if self._diag_timer:
            self.destroy_timer(self._diag_timer)
        if self._diag_pub:
            self.destroy_publisher(self._diag_pub)

        self.get_logger().info('Dialogue Manager shutdown complete.')
        return TransitionCallbackReturn.SUCCESS

    # =========================================================================
    # Voice tracking
    # =========================================================================

    def _on_voices_tracked(self, msg: IdsList) -> None:
        """Handle updates to tracked voices list."""
        current_voices = set(msg.ids)

        # Subscribe to new voices
        for voice_id in current_voices - self._tracked_voices:
            topic = f'/humans/voices/{voice_id}/speech'
            self.get_logger().info(f'Subscribing to {topic}')
            sub = self.create_subscription(
                LiveSpeech, topic,
                lambda m, vid=voice_id: self._on_speech(vid, m),
                10
            )
            self._voice_subscriptions[voice_id] = sub

        # Unsubscribe from removed voices
        for voice_id in self._tracked_voices - current_voices:
            if voice_id in self._voice_subscriptions:
                self.destroy_subscription(self._voice_subscriptions.pop(voice_id))

        self._tracked_voices = current_voices

    def _on_speech(self, voice_id: str, msg: LiveSpeech) -> None:
        """Handle incoming speech from a user."""
        if not msg.final:
            return  # Only process final speech

        if self._waiting_for_chatbot:
            self.get_logger().debug('Ignoring speech while waiting for chatbot')
            return

        # Publish closed caption for user speech
        caption = ClosedCaption()
        caption.speaker_id = voice_id
        caption.text = msg.final
        caption.locale = msg.locale if msg.locale else ''
        self._closed_captions_pub.publish(caption)

        # Get chatbot param
        chatbot = self.get_parameter('chatbot').get_parameter_value().string_value

        if not chatbot or not self._dialogue_interaction_client:
            # No chatbot - publish as RAW_USER_INPUT intent
            self._publish_raw_intent(msg.final, voice_id, msg.locale)
            return

        # Find dialogue for this voice/person
        # TODO: Map voice_id to person_id via ROS4HRI
        dialogue = self._dialogue_manager.get_dialogue_for_person(voice_id)

        if dialogue:
            # Forward to chatbot
            self._handle_dialogue_input(dialogue.dialogue_id, voice_id, msg.final)
        elif self._default_dialogue_id:
            # Use default dialogue
            self._handle_dialogue_input(self._default_dialogue_id, voice_id, msg.final)
        else:
            # No active dialogue - publish as raw intent
            self._publish_raw_intent(msg.final, voice_id, msg.locale)

    def _publish_raw_intent(self, text: str, source: str, locale: str = '') -> None:
        """Publish a RAW_USER_INPUT intent."""
        intent = Intent()
        intent.intent = Intent.RAW_USER_INPUT
        intent.data = json.dumps({'input': text, 'locale': locale})
        intent.source = source if source else Intent.UNKNOWN_AGENT
        intent.modality = Intent.MODALITY_SPEECH
        intent.priority = 128
        intent.confidence = 1.0
        self._intents_pub.publish(intent)

    # =========================================================================
    # Chatbot interaction
    # =========================================================================

    def _start_default_chat(self) -> None:
        """Start the default chat dialogue."""
        role_name = self.get_parameter('default_chat_role').get_parameter_value().string_value
        role_config = self.get_parameter('default_chat_configuration').get_parameter_value().string_value

        role = DialogueRole()
        role.name = role_name
        role.configuration = role_config if role_config else '{}'

        if self._dialogue_client and self._dialogue_client.wait_for_server(timeout_sec=1.0):
            goal = DialogueAction.Goal()
            goal.role = role
            future = self._dialogue_client.send_goal_async(goal)
            future.add_done_callback(self._on_default_dialogue_started)
        else:
            self.get_logger().warn('Chatbot not available for default chat')

    def _on_default_dialogue_started(self, future) -> None:
        """Handle default dialogue start result."""
        goal_handle = future.result()
        if goal_handle and goal_handle.accepted:
            self.get_logger().info('Default chat dialogue started')
            dialogue = Dialogue(
                role=DialogueRole(name='__default__'),
                priority=0,  # Default chat has lowest priority
                state=DialogueState.ACTIVE
            )
            self._default_dialogue_id = dialogue.dialogue_id
            self._dialogue_manager.add_dialogue(dialogue)
        else:
            self.get_logger().warn('Failed to start default dialogue')

    def _handle_dialogue_input(self, dialogue_id: UUID, user_id: str, text: str) -> None:
        """Send user input to chatbot and handle response."""
        if not self._dialogue_interaction_client:
            return

        dialogue = self._dialogue_manager.get_dialogue(dialogue_id)
        if not dialogue:
            return

        # Set waiting flag
        self._waiting_for_chatbot = True
        self._waiting_chatbot_pub.publish(Bool(data=True))
        dialogue.state = DialogueState.WAITING_RESPONSE

        # Build request
        request = DialogueInteraction.Request()
        request.dialogue_id = uuid_to_msg(dialogue_id)
        request.user_id = user_id
        request.input = text
        request.response_expected = True

        # Call service asynchronously
        future = self._dialogue_interaction_client.call_async(request)
        future.add_done_callback(
            lambda f: self._on_dialogue_response(f, dialogue_id)
        )

    def _on_dialogue_response(self, future, dialogue_id: UUID) -> None:
        """Handle chatbot response."""
        self._waiting_for_chatbot = False
        self._waiting_chatbot_pub.publish(Bool(data=False))

        dialogue = self._dialogue_manager.get_dialogue(dialogue_id)
        if dialogue:
            dialogue.state = DialogueState.ACTIVE

        try:
            response = future.result()
        except Exception as e:
            self.get_logger().error(f'Chatbot response failed: {e}')
            return

        if response.error_msg:
            self.get_logger().warn(f'Chatbot error: {response.error_msg}')
            return

        # Publish intents
        for intent in response.intents:
            self._intents_pub.publish(intent)

        # Speak response if present
        if response.response:
            self._speak_text(response.response)

    def _speak_text(self, text: str, priority: int = 128) -> None:
        """Send text to TTS engine."""
        if not self._tts_client:
            return

        self._dialogue_manager.set_expression_priority(priority)

        goal = TTS.Goal()
        goal.input = text

        if self._tts_client.wait_for_server(timeout_sec=1.0):
            send_future = self._tts_client.send_goal_async(
                goal, feedback_callback=self._on_tts_feedback
            )
            send_future.add_done_callback(self._on_tts_started)

            # Publish closed caption
            caption = ClosedCaption()
            caption.speaker_id = ClosedCaption.SPEAKER_ID_SYSTEM
            caption.text = text
            self._closed_captions_pub.publish(caption)
        else:
            self.get_logger().warn('TTS not available')
            self._dialogue_manager.clear_expression_priority()

    def _on_tts_started(self, future) -> None:
        """Handle TTS goal acceptance."""
        goal_handle = future.result()
        if not goal_handle or not goal_handle.accepted:
            self.get_logger().warn('TTS goal rejected')
            self._dialogue_manager.clear_expression_priority()
            return

        # Wait for result
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_tts_done)

    def _on_tts_feedback(self, feedback_msg) -> None:
        """Forward TTS feedback to robot_speech topic."""
        word = feedback_msg.feedback.word
        if word:
            self._robot_speech_pub.publish(String(data=word))

    def _on_tts_done(self, future) -> None:
        """Handle TTS completion."""
        self._dialogue_manager.clear_expression_priority()
        try:
            result = future.result()
            if result.result.error_msg:
                self.get_logger().warn(f'TTS error: {result.result.error_msg}')
        except Exception as e:
            self.get_logger().error(f'TTS failed: {e}')

    # =========================================================================
    # Goal callbacks (for action servers)
    # =========================================================================

    def _chat_goal_callback(self, goal_request) -> GoalResponse:
        """Accept or reject Chat goals based on state and priority."""
        if self._state_machine.current_state[1] != 'active':
            self.get_logger().warn('Chat rejected: node not active')
            return GoalResponse.REJECT

        priority = goal_request.meta.priority
        if not self._dialogue_manager.can_accept_priority(priority):
            self.get_logger().warn(f'Chat rejected: priority {priority} too low')
            return GoalResponse.REJECT

        self.get_logger().info('Chat goal accepted')
        return GoalResponse.ACCEPT

    def _ask_goal_callback(self, goal_request) -> GoalResponse:
        """Accept or reject Ask goals based on state and priority."""
        if self._state_machine.current_state[1] != 'active':
            self.get_logger().warn('Ask rejected: node not active')
            return GoalResponse.REJECT

        priority = goal_request.meta.priority
        if not self._dialogue_manager.can_accept_priority(priority):
            self.get_logger().warn(f'Ask rejected: priority {priority} too low')
            return GoalResponse.REJECT

        self.get_logger().info('Ask goal accepted')
        return GoalResponse.ACCEPT

    def _say_goal_callback(self, goal_request) -> GoalResponse:
        """Accept or reject Say goals based on state and priority."""
        if self._state_machine.current_state[1] != 'active':
            self.get_logger().warn('Say rejected: node not active')
            return GoalResponse.REJECT

        priority = goal_request.meta.priority
        if not self._dialogue_manager.can_accept_priority(priority):
            self.get_logger().warn(f'Say rejected: priority {priority} too low')
            return GoalResponse.REJECT

        self.get_logger().info('Say goal accepted')
        return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle) -> CancelResponse:
        """Accept cancellation requests."""
        self.get_logger().info('Cancellation requested')
        return CancelResponse.ACCEPT

    # =========================================================================
    # Action execution
    # =========================================================================

    async def _execute_chat(self, goal_handle) -> Chat.Result:
        """Execute Chat action."""
        request = goal_handle.request
        self.get_logger().info(f'Executing Chat with role: {request.role.name}')

        result = Chat.Result()

        # Check chatbot availability
        chatbot = self.get_parameter('chatbot').get_parameter_value().string_value
        if not chatbot or not self._dialogue_client:
            result.result.error_code = 134  # ENOTSUP
            result.result.error_msg = 'Chatbot not configured'
            goal_handle.abort()
            return result

        # Create dialogue
        dialogue = Dialogue(
            role=request.role,
            person_id=request.person_id,
            group_id=request.group_id,
            priority=request.meta.priority,
            state=DialogueState.PENDING,
            goal_handle=goal_handle
        )
        self._dialogue_manager.add_dialogue(dialogue)

        # Start dialogue with chatbot
        dialogue_goal = DialogueAction.Goal()
        dialogue_goal.role = request.role

        if not self._dialogue_client.wait_for_server(timeout_sec=5.0):
            result.result.error_code = 134
            result.result.error_msg = 'Chatbot not available'
            self._dialogue_manager.remove_dialogue(dialogue.dialogue_id)
            goal_handle.abort()
            return result

        send_future = self._dialogue_client.send_goal_async(dialogue_goal)
        chatbot_handle = await send_future

        if not chatbot_handle or not chatbot_handle.accepted:
            result.result.error_code = 134
            result.result.error_msg = 'Chatbot rejected dialogue'
            self._dialogue_manager.remove_dialogue(dialogue.dialogue_id)
            goal_handle.abort()
            return result

        dialogue.state = DialogueState.ACTIVE

        # If initiate=true, generate initial utterance
        if request.initiate:
            if request.initial_input:
                self._speak_text(request.initial_input, request.meta.priority)
            else:
                # Ask chatbot to generate greeting
                self._handle_dialogue_input(
                    dialogue.dialogue_id,
                    DialogueInteraction.Request.ASSISTANT_USER_ID,
                    ''  # Empty input triggers generation
                )

        # Wait for dialogue to complete (cancelled or successful)
        # The dialogue continues until cancelled by the caller
        while not goal_handle.is_cancel_requested:
            await self.get_clock().sleep_for_sec(0.1)
            if dialogue.state == DialogueState.COMPLETED:
                break

        # Cancel chatbot dialogue
        chatbot_handle.cancel_goal_async()

        # Clean up
        self._dialogue_manager.remove_dialogue(dialogue.dialogue_id)

        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
            result.result.error_code = 125  # ECANCELED
            return result

        goal_handle.succeed()
        return result

    async def _execute_ask(self, goal_handle) -> Ask.Result:
        """Execute Ask action (specialized Chat)."""
        request = goal_handle.request
        self.get_logger().info(f'Executing Ask: {request.question}')

        result = Ask.Result()

        # Build ASK_ROLE configuration
        role = DialogueRole()
        role.name = DialogueRole.ASK_ROLE
        role.configuration = json.dumps({
            'question': request.question,
            'result_schema_properties': json.loads(request.answers_schema) if request.answers_schema else {}
        })

        # Check chatbot availability
        chatbot = self.get_parameter('chatbot').get_parameter_value().string_value
        if not chatbot or not self._dialogue_client:
            result.result.error_code = 134
            result.result.error_msg = 'Chatbot not configured'
            goal_handle.abort()
            return result

        # Create dialogue
        dialogue = Dialogue(
            role=role,
            person_id=request.person_id,
            group_id=request.group_id,
            priority=request.meta.priority,
            state=DialogueState.PENDING,
            goal_handle=goal_handle
        )
        self._dialogue_manager.add_dialogue(dialogue)

        # Start dialogue
        dialogue_goal = DialogueAction.Goal()
        dialogue_goal.role = role

        if not self._dialogue_client.wait_for_server(timeout_sec=5.0):
            result.result.error_code = 134
            result.result.error_msg = 'Chatbot not available'
            self._dialogue_manager.remove_dialogue(dialogue.dialogue_id)
            goal_handle.abort()
            return result

        send_future = self._dialogue_client.send_goal_async(dialogue_goal)
        chatbot_handle = await send_future

        if not chatbot_handle or not chatbot_handle.accepted:
            result.result.error_code = 134
            result.result.error_msg = 'Chatbot rejected dialogue'
            self._dialogue_manager.remove_dialogue(dialogue.dialogue_id)
            goal_handle.abort()
            return result

        dialogue.state = DialogueState.ACTIVE

        # Speak the question (initiate is always true for Ask)
        self._speak_text(request.question, request.meta.priority)

        # Wait for dialogue completion or cancellation
        # In a full implementation, we'd track when the chatbot returns results
        while not goal_handle.is_cancel_requested:
            await self.get_clock().sleep_for_sec(0.1)
            if dialogue.state == DialogueState.COMPLETED:
                break

        # Get results from chatbot
        chatbot_result_future = chatbot_handle.get_result_async()
        try:
            chatbot_result = await chatbot_result_future
            result.answers = chatbot_result.result.results
        except Exception as e:
            self.get_logger().warn(f'Failed to get chatbot results: {e}')

        # Clean up
        self._dialogue_manager.remove_dialogue(dialogue.dialogue_id)

        if goal_handle.is_cancel_requested:
            chatbot_handle.cancel_goal_async()
            goal_handle.canceled()
            result.result.error_code = 125
            return result

        goal_handle.succeed()
        return result

    async def _execute_say(self, goal_handle) -> Say.Result:
        """Execute Say action."""
        request = goal_handle.request
        self.get_logger().info(f'Executing Say: {request.input[:50]}...')

        result = Say.Result()

        # Set expression priority
        priority = request.meta.priority
        self._dialogue_manager.set_expression_priority(priority)

        # Send to TTS
        if not self._tts_client or not self._tts_client.wait_for_server(timeout_sec=1.0):
            result.result.error_code = 134
            result.result.error_msg = 'TTS not available'
            self._dialogue_manager.clear_expression_priority()
            goal_handle.abort()
            return result

        tts_goal = TTS.Goal()
        tts_goal.input = request.input

        # Publish closed caption
        caption = ClosedCaption()
        caption.speaker_id = ClosedCaption.SPEAKER_ID_SYSTEM
        caption.text = request.input
        self._closed_captions_pub.publish(caption)

        send_future = self._tts_client.send_goal_async(
            tts_goal, feedback_callback=self._on_tts_feedback
        )
        tts_handle = await send_future

        if not tts_handle or not tts_handle.accepted:
            result.result.error_code = 134
            result.result.error_msg = 'TTS rejected goal'
            self._dialogue_manager.clear_expression_priority()
            goal_handle.abort()
            return result

        # Wait for TTS completion
        tts_result_future = tts_handle.get_result_async()

        while not goal_handle.is_cancel_requested:
            # Check if TTS is done
            if tts_result_future.done():
                break
            await self.get_clock().sleep_for_sec(0.05)

        self._dialogue_manager.clear_expression_priority()

        if goal_handle.is_cancel_requested:
            tts_handle.cancel_goal_async()
            goal_handle.canceled()
            result.result.error_code = 125
            return result

        try:
            tts_result = tts_result_future.result()
            if tts_result.result.error_msg:
                result.result.error_msg = tts_result.result.error_msg
        except Exception as e:
            self.get_logger().warn(f'TTS error: {e}')

        goal_handle.succeed()
        return result

    # =========================================================================
    # Diagnostics
    # =========================================================================

    def _publish_diagnostics(self) -> None:
        """Publish diagnostic information."""
        arr = DiagnosticArray()
        msg = DiagnosticStatus(
            level=DiagnosticStatus.OK,
            name='/dialogue_manager',
            message='Dialogue Manager running',
            values=[
                KeyValue(key='State', value=self._state_machine.current_state[1]),
                KeyValue(key='Active dialogues', value=str(len(self._dialogue_manager.active_dialogues))),
                KeyValue(key='Tracked voices', value=str(len(self._tracked_voices))),
                KeyValue(key='Waiting for chatbot', value=str(self._waiting_for_chatbot)),
            ]
        )

        arr.header.stamp = self.get_clock().now().to_msg()
        arr.status = [msg]
        self._diag_pub.publish(arr)
