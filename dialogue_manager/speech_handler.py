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

"""Speech input handler for the Dialogue Manager."""

import json
from typing import Dict, Optional, Set

from hri_actions_msgs.msg import ClosedCaption, Intent
from hri_msgs.msg import IdsList, LiveSpeech
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription

from .chatbot_client import ChatbotClient
from .dialogue import DialogueManager


class SpeechHandler:
    """
    Handles speech input from tracked voices.

    Manages dynamic subscriptions to voice speech topics,
    routes speech to the appropriate dialogue or publishes as raw intent.
    """

    def __init__(
        self,
        node: Node,
        dialogue_manager: DialogueManager,
        chatbot_client: ChatbotClient,
        closed_captions_pub: Publisher,
        intents_pub: Publisher
    ):
        """Initialize the speech handler."""
        self._node = node
        self._dialogue_manager = dialogue_manager
        self._chatbot_client = chatbot_client
        self._closed_captions_pub = closed_captions_pub
        self._intents_pub = intents_pub

        self._tracked_voices: Set[str] = set()
        self._voice_subscriptions: Dict[str, Subscription] = {}
        self._voices_sub: Optional[Subscription] = None
        self._chatbot_enabled = True

    def set_chatbot_enabled(self, enabled: bool) -> None:
        """Set whether chatbot is enabled for speech routing."""
        self._chatbot_enabled = enabled

    def subscribe_to_voices(self) -> None:
        """Subscribe to the tracked voices topic."""
        self._voices_sub = self._node.create_subscription(
            IdsList,
            '/humans/voices/tracked',
            self._on_voices_tracked,
            10
        )
        self._node.get_logger().info('[SPEECH] Subscribed to /humans/voices/tracked')

    def unsubscribe_all(self) -> None:
        """Unsubscribe from all voice topics."""
        # Unsubscribe from individual voices
        for voice_id, sub in self._voice_subscriptions.items():
            self._node.destroy_subscription(sub)
            self._node.get_logger().debug(f'[SPEECH] Unsubscribed from voice {voice_id}')
        self._voice_subscriptions.clear()
        self._tracked_voices.clear()

        # Unsubscribe from tracked voices
        if self._voices_sub:
            self._node.destroy_subscription(self._voices_sub)
            self._voices_sub = None

    def _on_voices_tracked(self, msg: IdsList) -> None:
        """Handle updates to tracked voices list."""
        current_voices = set(msg.ids)

        # Subscribe to new voices
        for voice_id in current_voices - self._tracked_voices:
            topic = f'/humans/voices/{voice_id}/speech'
            self._node.get_logger().info(f'[SPEECH] Subscribing to {topic}')
            sub = self._node.create_subscription(
                LiveSpeech,
                topic,
                lambda m, vid=voice_id: self._on_speech(vid, m),
                10
            )
            self._voice_subscriptions[voice_id] = sub

        # Unsubscribe from removed voices
        for voice_id in self._tracked_voices - current_voices:
            if voice_id in self._voice_subscriptions:
                self._node.destroy_subscription(self._voice_subscriptions.pop(voice_id))
                self._node.get_logger().info(f'[SPEECH] Unsubscribed from voice {voice_id}')

        self._tracked_voices = current_voices

    def _on_speech(self, voice_id: str, msg: LiveSpeech) -> None:
        """Handle incoming speech from a user."""
        if not msg.final:
            return  # Only process final speech

        self._node.get_logger().info(
            f'[SPEECH INPUT] voice_id="{voice_id}": "{msg.final}" '
            f'(locale={msg.locale}, confidence={msg.confidence:.2f})'
        )

        if self._chatbot_client.waiting_for_response:
            self._node.get_logger().warn(
                f'[SPEECH INPUT] Ignoring while waiting for chatbot: "{msg.final}"'
            )
            return

        # Publish closed caption for user speech
        self._publish_user_caption(voice_id, msg.final, msg.locale)

        if not self._chatbot_enabled:
            self._node.get_logger().info(
                '[SPEECH INPUT] Chatbot disabled, forwarding as RAW_USER_INPUT'
            )
            self._publish_raw_intent(msg.final, voice_id, msg.locale)
            return

        # Find dialogue for this voice/person
        # TODO: Map voice_id to person_id via ROS4HRI
        dialogue = self._dialogue_manager.get_dialogue_for_person(voice_id)

        if dialogue:
            self._node.get_logger().info(
                f'[SPEECH INPUT] Found dialogue {dialogue.dialogue_id}, forwarding to chatbot'
            )
            self._chatbot_client.send_input(dialogue.dialogue_id, voice_id, msg.final)
        elif self._chatbot_client.default_dialogue_id:
            self._node.get_logger().info(
                f'[SPEECH INPUT] Using default dialogue {self._chatbot_client.default_dialogue_id}'
            )
            self._chatbot_client.send_input(
                self._chatbot_client.default_dialogue_id, voice_id, msg.final
            )
        else:
            self._node.get_logger().info(
                '[SPEECH INPUT] No active dialogue, publishing as RAW_USER_INPUT'
            )
            self._publish_raw_intent(msg.final, voice_id, msg.locale)

    def _publish_user_caption(self, speaker_id: str, text: str, locale: str) -> None:
        """Publish closed caption for user speech."""
        caption = ClosedCaption()
        caption.speaker_id = speaker_id
        caption.text = text
        caption.locale = locale if locale else ''
        self._closed_captions_pub.publish(caption)
        self._node.get_logger().debug('[CLOSED CAPTION] Published user speech')

    def _publish_raw_intent(self, text: str, source: str, locale: str = '') -> None:
        """Publish a RAW_USER_INPUT intent."""
        self._node.get_logger().info(
            f'[INTENT] RAW_USER_INPUT: text="{text}", source="{source}"'
        )
        intent = Intent()
        intent.intent = Intent.RAW_USER_INPUT
        intent.data = json.dumps({'input': text, 'locale': locale})
        intent.source = source if source else Intent.UNKNOWN_AGENT
        intent.modality = Intent.MODALITY_SPEECH
        intent.priority = 128
        intent.confidence = 1.0
        self._intents_pub.publish(intent)
