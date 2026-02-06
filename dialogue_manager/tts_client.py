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

"""TTS client for the Dialogue Manager."""

from typing import Callable, Optional

from hri_actions_msgs.msg import ClosedCaption
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.publisher import Publisher
from std_msgs.msg import String
from tts_msgs.action import TTS

from .dialogue import DialogueManager


class TTSClient:
    """
    Handles text-to-speech interactions.

    Manages the TTS action client, publishes closed captions for robot speech,
    and forwards word-by-word feedback.
    """

    def __init__(
        self,
        node: Node,
        dialogue_manager: DialogueManager,
        closed_captions_pub: Publisher,
        robot_speech_pub: Publisher,
        callback_group: Optional[ReentrantCallbackGroup] = None
    ):
        """Initialize the TTS client."""
        self._node = node
        self._dialogue_manager = dialogue_manager
        self._closed_captions_pub = closed_captions_pub
        self._robot_speech_pub = robot_speech_pub
        self._callback_group = callback_group

        self._tts_client: Optional[ActionClient] = None
        self._on_complete_callback: Optional[Callable[[], None]] = None

    def create_client(self, action_name: str = 'tts_engine/tts') -> None:
        """Create the TTS action client."""
        self._tts_client = ActionClient(
            self._node,
            TTS,
            action_name,
            callback_group=self._callback_group
        )
        self._node.get_logger().info(f'[TTS] Created action client: {action_name}')

    def destroy(self) -> None:
        """Destroy the TTS action client."""
        if self._tts_client:
            self._tts_client.destroy()
            self._tts_client = None

    def is_available(self, timeout_sec: float = 1.0) -> bool:
        """Check if TTS server is available."""
        if not self._tts_client:
            return False
        return self._tts_client.wait_for_server(timeout_sec=timeout_sec)

    def speak(
        self,
        text: str,
        priority: int = 128,
        on_complete: Optional[Callable[[], None]] = None
    ) -> bool:
        """Send text to TTS engine."""
        if not self._tts_client:
            self._node.get_logger().warn('[TTS] No TTS client available')
            return False

        log_text = f'"{text[:80]}..."' if len(text) > 80 else f'"{text}"'
        self._node.get_logger().info(
            f'[TTS] Speaking text (priority={priority}): {log_text}'
        )

        self._dialogue_manager.set_expression_priority(priority)
        self._on_complete_callback = on_complete

        goal = TTS.Goal()
        goal.input = text

        if self._tts_client.wait_for_server(timeout_sec=1.0):
            self._node.get_logger().debug('[TTS] Server available, sending goal')
            send_future = self._tts_client.send_goal_async(
                goal, feedback_callback=self._on_feedback
            )
            send_future.add_done_callback(self._on_goal_response)

            # Publish closed caption
            caption = ClosedCaption()
            caption.speaker_id = ClosedCaption.SPEAKER_ID_SYSTEM
            caption.text = text
            self._closed_captions_pub.publish(caption)
            self._node.get_logger().debug('[TTS] Published robot closed caption')
            return True
        else:
            self._node.get_logger().warn('[TTS] Server not available (timeout after 1s)')
            self._dialogue_manager.clear_expression_priority()
            return False

    def _on_goal_response(self, future) -> None:
        """Handle TTS goal acceptance."""
        goal_handle = future.result()
        if not goal_handle or not goal_handle.accepted:
            self._node.get_logger().warn('[TTS] Goal rejected')
            self._dialogue_manager.clear_expression_priority()
            self._invoke_complete_callback()
            return

        self._node.get_logger().debug('[TTS] Goal accepted, waiting for result')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_result)

    def _on_feedback(self, feedback_msg) -> None:
        """Forward TTS feedback to robot_speech topic."""
        word = feedback_msg.feedback.word
        if word:
            self._robot_speech_pub.publish(String(data=word))

    def _on_result(self, future) -> None:
        """Handle TTS completion."""
        self._dialogue_manager.clear_expression_priority()
        try:
            result = future.result()
            if result.result.error_msg:
                self._node.get_logger().warn(f'[TTS] Error: {result.result.error_msg}')
            else:
                self._node.get_logger().debug('[TTS] Completed successfully')
        except Exception as e:
            self._node.get_logger().error(f'[TTS] Failed: {e}')
        finally:
            self._invoke_complete_callback()

    def _invoke_complete_callback(self) -> None:
        """Invoke the completion callback if set."""
        if self._on_complete_callback:
            try:
                self._on_complete_callback()
            except Exception as e:
                self._node.get_logger().error(f'[TTS] Complete callback failed: {e}')
            finally:
                self._on_complete_callback = None
