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

from collections.abc import Callable
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

from .dialogue import (
    Dialogue,
    DialogueManager,
    DialogueState,
    ROBOT_SPEAKER_ID,
)
from .markup.executor import ExpressionExecutor
from .say_client import SayClient


# Per chatbot_msgs/srv/DialogueInteraction.srv
SYSTEM_USER_ID = '__system__'
ASSISTANT_USER_ID = '__assistant__'


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
        say_client: SayClient,
        expression_executor: ExpressionExecutor | None = None,
        intents_pub: Publisher | None = None,
        waiting_chatbot_pub: Publisher | None = None,
        callback_group: ReentrantCallbackGroup | None = None
    ):
        """Initialize the chatbot client."""
        self._node = node
        self._dialogue_manager = dialogue_manager
        self._say_client = say_client
        self._expression_executor = expression_executor
        self._intents_pub = intents_pub
        self._waiting_chatbot_pub = waiting_chatbot_pub
        self._callback_group = callback_group

        self._dialogue_client: ActionClient | None = None
        self._interaction_client = None
        self._waiting_for_response = False

    def _now(self) -> float:
        """Return the current time in epoch seconds, from the node clock."""
        return self._node.get_clock().now().nanoseconds / 1e9

    @property
    def waiting_for_response(self) -> bool:
        """Return True if waiting for a chatbot response."""
        return self._waiting_for_response

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

    def attach_to_dialogue(
        self, dialogue: Dialogue, role: DialogueRole
    ) -> bool:
        """Asynchronously start a chatbot dialogue and attach it to `dialogue`.

        Sends a start_dialogue goal to the chatbot in the background; on
        acceptance, sets `dialogue.chatbot_goal_id` so subsequent
        SpeechHandler input is forwarded to the chatbot. Returns True if
        the goal was dispatched (server reachable), False otherwise.

        Speech that arrives *between* this call returning and the goal
        being accepted is still recorded into the dialogue history but
        not forwarded to the chatbot — a small first-utterance race that
        callers should be aware of.
        """
        if not self._dialogue_client or not self._dialogue_client.wait_for_server(
            timeout_sec=1.0
        ):
            self._node.get_logger().warn(
                '[CHATBOT] attach_to_dialogue: server not available'
            )
            return False
        goal = DialogueAction.Goal()
        goal.role = role
        future = self._dialogue_client.send_goal_async(goal)
        future.add_done_callback(
            lambda f, d=dialogue: self._on_attached_goal_response(f, d)
        )
        self._node.get_logger().info(
            f'[CHATBOT] Attaching chatbot dialogue to {dialogue.dialogue_id} '
            f'(role="{role.name}", interlocutor={dialogue.interlocutor.key})'
        )
        return True

    def _on_attached_goal_response(self, future, dialogue: Dialogue) -> None:
        """Handle the chatbot goal acceptance for an attached dialogue."""
        goal_handle = future.result()
        if not goal_handle or not goal_handle.accepted:
            self._node.get_logger().warn(
                f'[CHATBOT] Chatbot rejected dialogue '
                f'{dialogue.dialogue_id}; running passive'
            )
            return
        chatbot_goal_id = UUID(bytes=bytes(goal_handle.goal_id.uuid))
        dialogue.chatbot_goal_id = chatbot_goal_id
        self._dialogue_manager.notify_change(dialogue.dialogue_id)
        self._node.get_logger().info(
            f'[CHATBOT] Attached chatbot_goal_id={chatbot_goal_id} to '
            f'dialogue {dialogue.dialogue_id}'
        )

    def send_input(
        self,
        dialogue_id: UUID,
        user_id: str,
        text: str,
        response_callback: Callable | None = None
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

        # NOTE: user-attributable utterances are recorded by the SpeechHandler
        # before send_input is called, so the dialogue history stays coherent
        # in chatbot-less mode too. Do not re-record here.

        # Set waiting state
        self._waiting_for_response = True
        self._waiting_chatbot_pub.publish(Bool(data=True))
        dialogue.state = DialogueState.WAITING_RESPONSE
        self._dialogue_manager.notify_change(dialogue.dialogue_id)
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
        callback: Callable | None = None
    ) -> None:
        """Handle chatbot response."""
        self._waiting_for_response = False
        self._waiting_chatbot_pub.publish(Bool(data=False))

        dialogue = self._dialogue_manager.get_dialogue(dialogue_id)
        if dialogue:
            dialogue.state = DialogueState.ACTIVE
            self._dialogue_manager.notify_change(dialogue.dialogue_id)

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

        # Record the robot's utterance in the dialogue history.
        if dialogue and response.response:
            dialogue.add_utterance(
                ROBOT_SPEAKER_ID, response.response, self._now()
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

        # Speak response (with markup processing if executor available)
        if response.response:
            self._node.get_logger().info(
                '[CHATBOT RESPONSE] Speaking via Say sub-skill'
            )
            person_id = (
                dialogue.interlocutor.person_id
                if dialogue and dialogue.interlocutor else ''
            )
            group_id = (
                dialogue.interlocutor.group_id
                if dialogue and dialogue.interlocutor else ''
            )
            priority = dialogue.priority if dialogue else 128
            if self._expression_executor:
                self._expression_executor.execute_text(
                    response.response,
                    priority=priority,
                    person_id=person_id,
                    group_id=group_id,
                )
            else:
                self._say_client.speak(
                    response.response,
                    priority=priority,
                    person_id=person_id,
                    group_id=group_id,
                )
        else:
            self._node.get_logger().debug('[CHATBOT RESPONSE] No text to speak')

        # Invoke callback
        if callback:
            try:
                callback(response)
            except Exception as e:
                self._node.get_logger().error(f'[CHATBOT] Response callback failed: {e}')

    def inject_context(self, dialogue_id: UUID, context: str) -> bool:
        """
        Push a `__system__` priming message into the chatbot for this dialogue.

        Used at dialogue start to deliver the per-interlocutor conversation
        context. No response is expected and no history is recorded (system
        messages are not utterances per DIALOGUE_FLOW.md).

        See TODO.md — context delivery for the longer-term redesign.
        """
        if not context:
            return False
        if not self._interaction_client:
            return False

        dialogue = self._dialogue_manager.get_dialogue(dialogue_id)
        if not dialogue or not dialogue.chatbot_goal_id:
            return False

        request = DialogueInteraction.Request()
        request.dialogue_id = uuid_to_msg(dialogue.chatbot_goal_id)
        request.user_id = SYSTEM_USER_ID
        request.input = context
        request.response_expected = False

        self._node.get_logger().info(
            f'[CHATBOT] Injecting context for dialogue {dialogue_id} '
            f'({len(context)} chars)'
        )
        self._interaction_client.call_async(request)
        return True

    async def start_dialogue(
        self,
        role: DialogueRole,
        timeout_sec: float = 5.0
    ) -> object | None:
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
