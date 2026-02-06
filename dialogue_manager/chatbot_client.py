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

"""Chatbot client for the Dialogue Manager."""

from typing import Callable, Optional
from uuid import UUID

from chatbot_msgs.action import Dialogue as DialogueAction
from chatbot_msgs.msg import DialogueRole
from chatbot_msgs.srv import DialogueInteraction
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.publisher import Publisher
from std_msgs.msg import Bool
from unique_identifier_msgs.msg import UUID as UUIDMsg

from .dialogue import Dialogue, DialogueManager, DialogueState
from .tts_client import TTSClient


def uuid_to_msg(uuid: UUID) -> UUIDMsg:
    """Convert a Python UUID to a ROS UUID message."""
    return UUIDMsg(uuid=list(uuid.bytes))


class ChatbotClient:
    """
    Handles chatbot interactions.

    Manages dialogue sessions with the chatbot, sends user input,
    processes responses, and publishes detected intents.
    """

    def __init__(
        self,
        node: Node,
        dialogue_manager: DialogueManager,
        tts_client: TTSClient,
        intents_pub: Publisher,
        waiting_chatbot_pub: Publisher,
        callback_group: Optional[ReentrantCallbackGroup] = None
    ):
        """Initialize the chatbot client."""
        self._node = node
        self._dialogue_manager = dialogue_manager
        self._tts_client = tts_client
        self._intents_pub = intents_pub
        self._waiting_chatbot_pub = waiting_chatbot_pub
        self._callback_group = callback_group

        self._dialogue_client: Optional[ActionClient] = None
        self._interaction_client = None
        self._waiting_for_response = False
        self._default_dialogue_id: Optional[UUID] = None

    @property
    def waiting_for_response(self) -> bool:
        """Return True if waiting for a chatbot response."""
        return self._waiting_for_response

    @property
    def default_dialogue_id(self) -> Optional[UUID]:
        """Return the default dialogue ID if active."""
        return self._default_dialogue_id

    def create_clients(self, chatbot_prefix: str) -> None:
        """Create chatbot action and service clients."""
        dialogue_action = f'{chatbot_prefix}/start_dialogue'
        interaction_srv = f'{chatbot_prefix}/dialogue_interaction'

        self._dialogue_client = ActionClient(
            self._node,
            DialogueAction,
            dialogue_action,
            callback_group=self._callback_group
        )
        self._node.get_logger().info(f'[CHATBOT] Created action client: {dialogue_action}')

        self._interaction_client = self._node.create_client(
            DialogueInteraction,
            interaction_srv,
            callback_group=self._callback_group
        )
        self._node.get_logger().info(f'[CHATBOT] Created service client: {interaction_srv}')

    def destroy(self) -> None:
        """Destroy chatbot clients."""
        if self._dialogue_client:
            self._dialogue_client.destroy()
            self._dialogue_client = None
        if self._interaction_client:
            self._node.destroy_client(self._interaction_client)
            self._interaction_client = None

    def is_available(self, timeout_sec: float = 1.0) -> bool:
        """Check if chatbot is available."""
        if not self._dialogue_client:
            return False
        return self._dialogue_client.wait_for_server(timeout_sec=timeout_sec)

    def start_default_chat(self, role_name: str, role_config: str = '{}') -> None:
        """Start the default chat dialogue."""
        log_config = f'"{role_config[:50]}..."' if len(role_config) > 50 else f'"{role_config}"'
        self._node.get_logger().info(
            f'[DEFAULT CHAT] Starting with role="{role_name}", configuration={log_config}'
        )

        role = DialogueRole()
        role.name = role_name
        role.configuration = role_config if role_config else '{}'

        if self._dialogue_client and self._dialogue_client.wait_for_server(timeout_sec=1.0):
            self._node.get_logger().debug('[DEFAULT CHAT] Server available, sending goal')
            goal = DialogueAction.Goal()
            goal.role = role
            future = self._dialogue_client.send_goal_async(goal)
            future.add_done_callback(self._on_default_dialogue_started)
        else:
            self._node.get_logger().warn('[DEFAULT CHAT] Chatbot not available')

    def _on_default_dialogue_started(self, future) -> None:
        """Handle default dialogue start result."""
        goal_handle = future.result()
        if goal_handle and goal_handle.accepted:
            # Extract the chatbot's goal UUID from the goal handle
            chatbot_goal_id = UUID(bytes=bytes(goal_handle.goal_id.uuid))
            dialogue = Dialogue(
                role=DialogueRole(name='__default__'),
                priority=0,  # Default chat has lowest priority
                state=DialogueState.ACTIVE,
                chatbot_goal_id=chatbot_goal_id
            )
            self._default_dialogue_id = dialogue.dialogue_id
            self._dialogue_manager.add_dialogue(dialogue)
            self._node.get_logger().info(
                f'[DEFAULT CHAT] Started successfully, '
                f'internal_id={dialogue.dialogue_id}, chatbot_goal_id={chatbot_goal_id}'
            )
        else:
            self._node.get_logger().warn('[DEFAULT CHAT] Failed - chatbot rejected goal')

    def send_input(
        self,
        dialogue_id: UUID,
        user_id: str,
        text: str,
        response_callback: Optional[Callable] = None
    ) -> bool:
        """Send user input to chatbot."""
        if not self._interaction_client:
            self._node.get_logger().warn('[CHATBOT] No interaction client available')
            return False

        dialogue = self._dialogue_manager.get_dialogue(dialogue_id)
        if not dialogue:
            self._node.get_logger().warn(f'[CHATBOT] Dialogue {dialogue_id} not found')
            return False

        if not dialogue.chatbot_goal_id:
            self._node.get_logger().warn(
                f'[CHATBOT] Dialogue {dialogue_id} has no chatbot goal ID'
            )
            return False

        self._node.get_logger().info(
            f'[CHATBOT REQUEST] chatbot_goal_id={dialogue.chatbot_goal_id}, '
            f'user_id="{user_id}", text="{text}"'
        )

        # Set waiting state
        self._waiting_for_response = True
        self._waiting_chatbot_pub.publish(Bool(data=True))
        dialogue.state = DialogueState.WAITING_RESPONSE
        self._node.get_logger().debug('[CHATBOT] State set to WAITING_RESPONSE')

        # Build and send request - use chatbot_goal_id, not internal dialogue_id
        request = DialogueInteraction.Request()
        request.dialogue_id = uuid_to_msg(dialogue.chatbot_goal_id)
        request.user_id = user_id
        request.input = text
        request.response_expected = True

        self._node.get_logger().debug('[CHATBOT] Calling dialogue_interaction async')
        future = self._interaction_client.call_async(request)
        future.add_done_callback(
            lambda f: self._on_response(f, dialogue_id, response_callback)
        )
        return True

    def _on_response(
        self,
        future,
        dialogue_id: UUID,
        callback: Optional[Callable] = None
    ) -> None:
        """Handle chatbot response."""
        self._waiting_for_response = False
        self._waiting_chatbot_pub.publish(Bool(data=False))

        dialogue = self._dialogue_manager.get_dialogue(dialogue_id)
        if dialogue:
            dialogue.state = DialogueState.ACTIVE

        try:
            response = future.result()
        except Exception as e:
            self._node.get_logger().error(f'[CHATBOT RESPONSE] Failed: {e}')
            return

        if response.error_msg:
            self._node.get_logger().warn(
                f'[CHATBOT RESPONSE] Error: {response.error_msg}'
            )
            return

        if len(response.response) > 100:
            log_text = f'"{response.response[:100]}..."'
        else:
            log_text = f'"{response.response}"'
        self._node.get_logger().info(
            f'[CHATBOT RESPONSE] dialogue_id={dialogue_id}: {log_text}'
        )

        # Publish intents
        if response.intents:
            self._node.get_logger().info(
                f'[CHATBOT RESPONSE] {len(response.intents)} intent(s) detected'
            )
            for i, intent in enumerate(response.intents):
                if len(intent.data) > 80:
                    log_data = f'"{intent.data[:80]}..."'
                else:
                    log_data = f'"{intent.data}"'
                self._node.get_logger().info(
                    f'[INTENT {i+1}] type="{intent.intent}", data={log_data}'
                )
                self._intents_pub.publish(intent)

        # Speak response
        if response.response:
            self._node.get_logger().info('[CHATBOT RESPONSE] Speaking via TTS')
            self._tts_client.speak(response.response)
        else:
            self._node.get_logger().debug('[CHATBOT RESPONSE] No text to speak')

        # Invoke callback
        if callback:
            try:
                callback(response)
            except Exception as e:
                self._node.get_logger().error(f'[CHATBOT] Response callback failed: {e}')

    async def start_dialogue(
        self,
        role: DialogueRole,
        timeout_sec: float = 5.0
    ) -> Optional[object]:
        """Start a new dialogue session."""
        if not self._dialogue_client:
            self._node.get_logger().warn('[CHATBOT] No dialogue client available')
            return None

        if not self._dialogue_client.wait_for_server(timeout_sec=timeout_sec):
            self._node.get_logger().warn('[CHATBOT] Server not available')
            return None

        goal = DialogueAction.Goal()
        goal.role = role

        self._node.get_logger().info(f'[CHATBOT] Starting dialogue with role="{role.name}"')
        send_future = self._dialogue_client.send_goal_async(goal)
        goal_handle = await send_future

        if not goal_handle or not goal_handle.accepted:
            self._node.get_logger().warn('[CHATBOT] Dialogue goal rejected')
            return None

        self._node.get_logger().info('[CHATBOT] Dialogue started successfully')
        return goal_handle
