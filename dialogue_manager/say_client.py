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

"""Say sub-skill client for the Dialogue Manager."""

from collections.abc import Callable
import threading

from communication_skills.action import Say
from hri_actions_msgs.msg import ClosedCaption
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.publisher import Publisher
from std_msgs.msg import String

from .dialogue import DialogueManager


CALLER_NAME = 'dialogue_manager'


class SayClient:
    """
    Client for the Say sub-skill action server.

    Sends utterances to an external Say action server (acting as a frontend
    to the TTS engine), publishes closed captions for robot speech, and
    forwards per-word feedback received via Feedback.data_str.
    """

    def __init__(
        self,
        node: Node,
        dialogue_manager: DialogueManager,
        closed_captions_pub: Publisher,
        robot_speech_pub: Publisher,
        callback_group: ReentrantCallbackGroup | None = None
    ):
        """Initialize the Say client."""
        self._node = node
        self._dialogue_manager = dialogue_manager
        self._closed_captions_pub = closed_captions_pub
        self._robot_speech_pub = robot_speech_pub
        self._callback_group = callback_group

        self._say_client: ActionClient | None = None
        self._on_complete_callback: Callable[[], None] | None = None

    def create_client(self, action_name: str = '/tts/say') -> None:
        """Create the Say action client."""
        self._say_client = ActionClient(
            self._node,
            Say,
            action_name,
            callback_group=self._callback_group
        )
        self._node.get_logger().info(f'[SAY] Created action client: {action_name}')

    def destroy(self) -> None:
        """Destroy the Say action client."""
        if self._say_client:
            self._say_client.destroy()
            self._say_client = None

    def is_available(self, timeout_sec: float = 1.0) -> bool:
        """Check if Say server is available."""
        if not self._say_client:
            return False
        return self._say_client.wait_for_server(timeout_sec=timeout_sec)

    def speak(
        self,
        text: str,
        priority: int = 128,
        person_id: str = '',
        group_id: str = '',
        on_complete: Callable[[], None] | None = None,
    ) -> bool:
        """Send text to the Say sub-skill (markup-stripped plain text)."""
        if not self._say_client:
            self._node.get_logger().warn('[SAY] No Say client available')
            return False

        log_text = f'"{text[:80]}..."' if len(text) > 80 else f'"{text}"'
        self._node.get_logger().info(
            f'[SAY] Sending text (priority={priority}): {log_text}'
        )

        self._dialogue_manager.set_expression_priority(priority)
        self._on_complete_callback = on_complete

        goal = Say.Goal()
        goal.meta.caller = CALLER_NAME
        goal.meta.priority = priority
        goal.person_id = person_id
        goal.group_id = group_id
        goal.input = text

        if self._say_client.wait_for_server(timeout_sec=1.0):
            self._node.get_logger().debug('[SAY] Server available, sending goal')
            send_future = self._say_client.send_goal_async(
                goal, feedback_callback=self._on_feedback
            )
            send_future.add_done_callback(self._on_goal_response)

            caption = ClosedCaption()
            caption.speaker_id = ClosedCaption.SPEAKER_ID_SYSTEM
            caption.text = text
            self._closed_captions_pub.publish(caption)
            self._node.get_logger().debug('[SAY] Published robot closed caption')
            return True
        else:
            self._node.get_logger().warn('[SAY] Server not available (timeout after 1s)')
            self._dialogue_manager.clear_expression_priority()
            return False

    def speak_and_wait(
        self,
        text: str,
        priority: int = 128,
        cancel_event: threading.Event | None = None,
        person_id: str = '',
        group_id: str = '',
    ) -> bool:
        """Send text to Say and block until complete or cancelled."""
        done_event = threading.Event()

        success = self.speak(
            text,
            priority=priority,
            person_id=person_id,
            group_id=group_id,
            on_complete=done_event.set,
        )
        if not success:
            return False

        while not done_event.is_set():
            if cancel_event and cancel_event.is_set():
                return False
            done_event.wait(timeout=0.05)

        return True

    def _on_goal_response(self, future) -> None:
        """Handle Say goal acceptance."""
        goal_handle = future.result()
        if not goal_handle or not goal_handle.accepted:
            self._node.get_logger().warn('[SAY] Goal rejected')
            self._dialogue_manager.clear_expression_priority()
            self._invoke_complete_callback()
            return

        self._node.get_logger().debug('[SAY] Goal accepted, waiting for result')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_result)

    def _on_feedback(self, feedback_msg) -> None:
        """Forward Say feedback (per-word in data_str) to robot_speech topic."""
        word = feedback_msg.feedback.feedback.data_str
        if word:
            self._robot_speech_pub.publish(String(data=word))

    def _on_result(self, future) -> None:
        """Handle Say completion."""
        self._dialogue_manager.clear_expression_priority()
        try:
            result = future.result()
            if result.result.result.error_msg:
                self._node.get_logger().warn(
                    f'[SAY] Error: {result.result.result.error_msg}'
                )
            else:
                self._node.get_logger().debug('[SAY] Completed successfully')
        except Exception as e:
            self._node.get_logger().error(f'[SAY] Failed: {e}')
        finally:
            self._invoke_complete_callback()

    def _invoke_complete_callback(self) -> None:
        """Invoke the completion callback if set."""
        if self._on_complete_callback:
            try:
                self._on_complete_callback()
            except Exception as e:
                self._node.get_logger().error(f'[SAY] Complete callback failed: {e}')
            finally:
                self._on_complete_callback = None
