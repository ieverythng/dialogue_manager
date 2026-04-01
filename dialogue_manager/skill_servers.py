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

"""Skill action servers for the Dialogue Manager."""

import json
import threading
import time
from typing import Optional

from chatbot_msgs.msg import DialogueRole
from communication_skills.action import Ask, Chat, Say
from hri_actions_msgs.msg import ClosedCaption
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.publisher import Publisher

from .chatbot_client import ChatbotClient
from .dialogue import Dialogue, DialogueManager, DialogueState
from .markup.executor import ExpressionExecutor
from .tts_client import TTSClient


class SkillServers:
    """
    Manages Chat, Ask, and Say action servers.

    Handles goal acceptance, execution, and cancellation for all skills.
    """

    def __init__(
        self,
        node: Node,
        dialogue_manager: DialogueManager,
        chatbot_client: ChatbotClient,
        tts_client: TTSClient,
        expression_executor: Optional[ExpressionExecutor] = None,
        closed_captions_pub: Optional[Publisher] = None,
        callback_group: Optional[ReentrantCallbackGroup] = None
    ):
        """Initialize skill servers."""
        self._node = node
        self._dialogue_manager = dialogue_manager
        self._chatbot_client = chatbot_client
        self._tts_client = tts_client
        self._expression_executor = expression_executor
        self._closed_captions_pub = closed_captions_pub
        self._callback_group = callback_group

        self._chat_server: Optional[ActionServer] = None
        self._ask_server: Optional[ActionServer] = None
        self._say_server: Optional[ActionServer] = None
        self._is_active = False

    def set_active(self, active: bool) -> None:
        """Set whether the node is in active state."""
        self._is_active = active

    def create_servers(self) -> None:
        """Create all action servers."""
        self._chat_server = ActionServer(
            self._node,
            Chat,
            '/skill/chat',
            execute_callback=self._execute_chat,
            goal_callback=self._chat_goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self._callback_group
        )
        self._node.get_logger().info('[SKILLS] Created /skill/chat action server')

        self._ask_server = ActionServer(
            self._node,
            Ask,
            '/skill/ask',
            execute_callback=self._execute_ask,
            goal_callback=self._ask_goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self._callback_group
        )
        self._node.get_logger().info('[SKILLS] Created /skill/ask action server')

        self._say_server = ActionServer(
            self._node,
            Say,
            '/skill/say',
            execute_callback=self._execute_say,
            goal_callback=self._say_goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self._callback_group
        )
        self._node.get_logger().info('[SKILLS] Created /skill/say action server')

    def destroy(self) -> None:
        """Destroy all action servers."""
        if self._chat_server:
            self._chat_server.destroy()
            self._chat_server = None
        if self._ask_server:
            self._ask_server.destroy()
            self._ask_server = None
        if self._say_server:
            self._say_server.destroy()
            self._say_server = None

    # =========================================================================
    # Goal callbacks
    # =========================================================================

    def _chat_goal_callback(self, goal_request) -> GoalResponse:
        """Accept or reject Chat goals based on state and priority."""
        if not self._is_active:
            self._node.get_logger().warn('[CHAT] Rejected: node not active')
            return GoalResponse.REJECT

        priority = goal_request.meta.priority
        if not self._dialogue_manager.can_accept_priority(priority):
            self._node.get_logger().warn(f'[CHAT] Rejected: priority {priority} too low')
            return GoalResponse.REJECT

        self._node.get_logger().info(f'[CHAT] Goal accepted (priority={priority})')
        return GoalResponse.ACCEPT

    def _ask_goal_callback(self, goal_request) -> GoalResponse:
        """Accept or reject Ask goals based on state and priority."""
        if not self._is_active:
            self._node.get_logger().warn('[ASK] Rejected: node not active')
            return GoalResponse.REJECT

        priority = goal_request.meta.priority
        if not self._dialogue_manager.can_accept_priority(priority):
            self._node.get_logger().warn(f'[ASK] Rejected: priority {priority} too low')
            return GoalResponse.REJECT

        self._node.get_logger().info(f'[ASK] Goal accepted (priority={priority})')
        return GoalResponse.ACCEPT

    def _say_goal_callback(self, goal_request) -> GoalResponse:
        """Accept or reject Say goals based on state and priority."""
        if not self._is_active:
            self._node.get_logger().warn('[SAY] Rejected: node not active')
            return GoalResponse.REJECT

        priority = goal_request.meta.priority
        if not self._dialogue_manager.can_accept_priority(priority):
            self._node.get_logger().warn(f'[SAY] Rejected: priority {priority} too low')
            return GoalResponse.REJECT

        self._node.get_logger().info(f'[SAY] Goal accepted (priority={priority})')
        return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle) -> CancelResponse:
        """Accept cancellation requests."""
        self._node.get_logger().info('[SKILLS] Cancellation requested')
        return CancelResponse.ACCEPT

    # =========================================================================
    # Chat execution
    # =========================================================================

    async def _execute_chat(self, goal_handle) -> Chat.Result:
        """Execute Chat action."""
        request = goal_handle.request
        self._node.get_logger().info(f'[CHAT] Executing with role="{request.role.name}"')

        result = Chat.Result()

        # Check chatbot availability
        if not self._chatbot_client.is_available(timeout_sec=1.0):
            result.result.error_code = 134  # ENOTSUP
            result.result.error_msg = 'Chatbot not configured'
            self._node.get_logger().warn('[CHAT] Aborted: chatbot not available')
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
        self._node.get_logger().info(f'[CHAT] Created dialogue {dialogue.dialogue_id}')

        # Start dialogue with chatbot
        chatbot_handle = await self._chatbot_client.start_dialogue(request.role)
        if not chatbot_handle:
            result.result.error_code = 134
            result.result.error_msg = 'Chatbot rejected dialogue'
            self._dialogue_manager.remove_dialogue(dialogue.dialogue_id)
            goal_handle.abort()
            return result

        # Extract chatbot goal ID and assign to dialogue
        from uuid import UUID as PyUUID
        chatbot_goal_id = PyUUID(bytes=bytes(chatbot_handle.goal_id.uuid))
        dialogue.chatbot_goal_id = chatbot_goal_id
        self._node.get_logger().debug(
            f'[CHAT] Assigned chatbot_goal_id={chatbot_goal_id} to dialogue'
        )

        dialogue.state = DialogueState.ACTIVE
        self._node.get_logger().info(f'[CHAT] Dialogue {dialogue.dialogue_id} now ACTIVE')

        # If initiate=true, generate initial utterance
        if request.initiate:
            if request.initial_input:
                self._node.get_logger().info('[CHAT] Speaking initial input')
                self._tts_client.speak(request.initial_input, request.meta.priority)
            else:
                # Ask chatbot to generate greeting
                self._node.get_logger().info('[CHAT] Requesting chatbot greeting')
                self._chatbot_client.send_input(
                    dialogue.dialogue_id,
                    '__assistant__',
                    ''  # Empty input triggers generation
                )

        # Wait for dialogue to complete
        self._node.get_logger().debug('[CHAT] Waiting for completion or cancellation')
        while not goal_handle.is_cancel_requested:
            time.sleep(0.1)
            if dialogue.state == DialogueState.COMPLETED:
                break

        # Cancel chatbot dialogue
        chatbot_handle.cancel_goal_async()

        # Clean up
        self._dialogue_manager.remove_dialogue(dialogue.dialogue_id)

        if goal_handle.is_cancel_requested:
            self._node.get_logger().info('[CHAT] Cancelled by caller')
            goal_handle.canceled()
            result.result.error_code = 125  # ECANCELED
            return result

        self._node.get_logger().info('[CHAT] Completed successfully')
        goal_handle.succeed()
        return result

    # =========================================================================
    # Ask execution
    # =========================================================================

    async def _execute_ask(self, goal_handle) -> Ask.Result:
        """Execute Ask action (specialized Chat)."""
        request = goal_handle.request
        self._node.get_logger().info(f'[ASK] Executing: "{request.question}"')

        result = Ask.Result()

        # Build ASK_ROLE configuration
        role = DialogueRole()
        role.name = DialogueRole.ASK_ROLE
        role.configuration = json.dumps({
            'question': request.question,
            'result_schema_properties': json.loads(request.answers_schema)
            if request.answers_schema else {}
        })

        # Check chatbot availability
        if not self._chatbot_client.is_available(timeout_sec=0.1):
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
        chatbot_handle = await self._chatbot_client.start_dialogue(role)
        if not chatbot_handle:
            result.result.error_code = 134
            result.result.error_msg = 'Chatbot rejected dialogue'
            self._dialogue_manager.remove_dialogue(dialogue.dialogue_id)
            goal_handle.abort()
            return result

        # Extract chatbot goal ID and assign to dialogue
        from uuid import UUID as PyUUID
        chatbot_goal_id = PyUUID(bytes=bytes(chatbot_handle.goal_id.uuid))
        dialogue.chatbot_goal_id = chatbot_goal_id
        self._node.get_logger().debug(
            f'[ASK] Assigned chatbot_goal_id={chatbot_goal_id} to dialogue'
        )

        dialogue.state = DialogueState.ACTIVE

        # Speak the question
        self._node.get_logger().info('[ASK] Speaking question via TTS')
        self._tts_client.speak(request.question, request.meta.priority)

        # Wait for dialogue completion
        while not goal_handle.is_cancel_requested:
            time.sleep(0.1)
            if dialogue.state == DialogueState.COMPLETED:
                break

        # Get results from chatbot
        try:
            chatbot_result_future = chatbot_handle.get_result_async()
            chatbot_result = await chatbot_result_future
            result.answers = chatbot_result.result.results
            self._node.get_logger().info(f'[ASK] Got answers: {result.answers}')
        except Exception as e:
            self._node.get_logger().warn(f'[ASK] Failed to get results: {e}')

        # Clean up
        self._dialogue_manager.remove_dialogue(dialogue.dialogue_id)

        if goal_handle.is_cancel_requested:
            chatbot_handle.cancel_goal_async()
            goal_handle.canceled()
            result.result.error_code = 125
            return result

        goal_handle.succeed()
        return result

    # =========================================================================
    # Say execution
    # =========================================================================

    async def _execute_say(self, goal_handle) -> Say.Result:
        """Execute Say action."""
        request = goal_handle.request
        if len(request.input) > 50:
            log_text = f'"{request.input[:50]}..."'
        else:
            log_text = f'"{request.input}"'
        self._node.get_logger().info(f'[SAY] Executing: {log_text}')

        result = Say.Result()
        priority = request.meta.priority

        # Set expression priority
        self._dialogue_manager.set_expression_priority(priority)

        # Check TTS availability
        if not self._tts_client.is_available(timeout_sec=1.0):
            result.result.error_code = 134
            result.result.error_msg = 'TTS not available'
            self._dialogue_manager.clear_expression_priority()
            goal_handle.abort()
            return result

        # Publish closed caption with plain text (markup stripped)
        caption = ClosedCaption()
        caption.speaker_id = ClosedCaption.SPEAKER_ID_SYSTEM
        if self._expression_executor:
            caption.text = self._expression_executor.get_plain_text(
                request.input
            )
        else:
            caption.text = request.input
        self._closed_captions_pub.publish(caption)

        # Execute expression with markup processing
        cancel_event = threading.Event()

        def _check_cancel():
            """Poll for goal cancellation in a background thread."""
            while not cancel_event.is_set():
                if goal_handle.is_cancel_requested:
                    cancel_event.set()
                    return
                time.sleep(0.05)

        cancel_thread = threading.Thread(target=_check_cancel, daemon=True)
        cancel_thread.start()

        try:
            if self._expression_executor:
                success = self._expression_executor.execute_text(
                    request.input,
                    priority=priority,
                    cancel_event=cancel_event,
                )
            else:
                success = self._tts_client.speak_and_wait(
                    request.input, priority, cancel_event
                )
        finally:
            cancel_event.set()  # Stop the cancel-check thread
            cancel_thread.join(timeout=1.0)
            self._dialogue_manager.clear_expression_priority()

        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
            result.result.error_code = 125
            return result

        if not success:
            result.result.error_msg = 'Expression execution failed'
            goal_handle.abort()
            return result

        self._node.get_logger().info('[SAY] Completed successfully')
        goal_handle.succeed()
        return result
