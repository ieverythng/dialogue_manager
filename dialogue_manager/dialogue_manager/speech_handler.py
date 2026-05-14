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

from chatbot_msgs.msg import DialogueRole
from hri_actions_msgs.msg import ClosedCaption, Intent
from hri_msgs.msg import IdsList, LiveSpeech
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.subscription import Subscription

from .chatbot_client import ChatbotClient
from .conversations_history import ConversationsHistoryStore
from .dialogue import Dialogue, DialogueManager, DialogueState, Interlocutor
from .fanout import related_interlocutors
from .group_handler import GroupHandler


# Lowest-priority bucket for auto-spawned default-chat dialogues.
DEFAULT_CHAT_PRIORITY = 0


class SpeechHandler:
    """
    Handle speech input from tracked voices.

    Manages dynamic subscriptions to voice speech topics, records user
    utterances against per-person and per-group dialogues, and either
    forwards them to a chatbot (when one is attached) or publishes them
    as RAW_USER_INPUT intents.

    When the `default_chat` mode is enabled, the first utterance from a
    new speaker auto-spawns a per-person `Dialogue`; if the speaker is
    in any known ROS4HRI group, the group's dialogue and a per-person
    dialogue for every co-member are also spawned, and the utterance is
    recorded into all of them.
    """

    def __init__(
        self,
        node: Node,
        dialogue_manager: DialogueManager,
        chatbot_client: ChatbotClient | None,
        conversations_store: ConversationsHistoryStore,
        closed_captions_pub: Publisher,
        intents_pub: Publisher,
        group_handler: GroupHandler | None = None,
    ):
        """Initialize the speech handler."""
        self._node = node
        self._dialogue_manager = dialogue_manager
        self._chatbot_client = chatbot_client
        self._conversations_store = conversations_store
        self._closed_captions_pub = closed_captions_pub
        self._intents_pub = intents_pub
        self._group_handler = group_handler

        self._tracked_voices: set[str] = set()
        self._voice_subscriptions: dict[str, Subscription] = {}
        self._voices_sub: Subscription | None = None
        self._chatbot_enabled = True

        self._default_chat_role: str = ''
        self._default_chat_role_config: str = '{}'

    def _now(self) -> float:
        """Return the current time in epoch seconds, from the node clock."""
        return self._node.get_clock().now().nanoseconds / 1e9

    def set_chatbot_enabled(self, enabled: bool) -> None:
        """Set whether chatbot is enabled for speech routing."""
        self._chatbot_enabled = enabled

    def set_default_chat(self, role_name: str, role_config: str = '{}') -> None:
        """
        Enable per-person default-chat auto-spawning with the given role.

        Pass an empty `role_name` to disable.
        """
        self._default_chat_role = role_name or ''
        self._default_chat_role_config = role_config or '{}'

    def subscribe_to_voices(self) -> None:
        """Subscribe to the tracked voices topic (latched / TRANSIENT_LOCAL)."""
        qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._voices_sub = self._node.create_subscription(
            IdsList,
            '/humans/voices/tracked',
            self._on_voices_tracked,
            qos,
        )
        self._node.get_logger().info('[SPEECH] Subscribed to /humans/voices/tracked')

    def unsubscribe_all(self) -> None:
        """Unsubscribe from all voice topics."""
        for voice_id, sub in self._voice_subscriptions.items():
            self._node.destroy_subscription(sub)
            self._node.get_logger().debug(f'[SPEECH] Unsubscribed from voice {voice_id}')
        self._voice_subscriptions.clear()
        self._tracked_voices.clear()

        if self._voices_sub:
            self._node.destroy_subscription(self._voices_sub)
            self._voices_sub = None

    def _on_voices_tracked(self, msg: IdsList) -> None:
        """Handle updates to tracked voices list."""
        current_voices = set(msg.ids)

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

        for voice_id in self._tracked_voices - current_voices:
            if voice_id in self._voice_subscriptions:
                self._node.destroy_subscription(self._voice_subscriptions.pop(voice_id))
                self._node.get_logger().info(f'[SPEECH] Unsubscribed from voice {voice_id}')

        self._tracked_voices = current_voices

    def is_voice_tracked(self, voice_id: str) -> bool:
        """
        Return True if `voice_id` is currently in /humans/voices/tracked.

        This is the canonical "is this person present" query. The
        SkillServers consult it via the `presence_query` callable to
        skip dialogues for people who have left during broadcast and
        addressed Say fan-out.
        """
        return voice_id in self._tracked_voices

    def _on_speech(self, voice_id: str, msg: LiveSpeech) -> None:
        """Handle incoming speech from a user."""
        if not msg.final:
            return  # Only process final speech

        self._node.get_logger().info(
            f'[SPEECH INPUT] voice_id="{voice_id}": "{msg.final}" '
            f'(locale={msg.locale}, confidence={msg.confidence:.2f})'
        )

        if (self._chatbot_client is not None
                and self._chatbot_client.waiting_for_response):
            self._node.get_logger().warn(
                f'[SPEECH INPUT] Ignoring while waiting for chatbot: "{msg.final}"'
            )
            return

        self._publish_user_caption(voice_id, msg.final, msg.locale)

        # Collect every dialogue that should receive this utterance:
        # speaker's own + each group the speaker belongs to + each
        # co-member's per-person dialogue. Missing dialogues are
        # auto-spawned when default-chat is enabled.
        # TODO: Map voice_id to person_id via ROS4HRI
        recipients = self._recipient_dialogues_for_speaker(voice_id)

        for dialogue in recipients:
            dialogue.add_utterance(voice_id, msg.final, self._now())

        # Trigger a chatbot turn for the speaker's dialogue (the first
        # recipient when one exists). The chatbot is stateless wrt.
        # history; `interact()` snapshots `dialogue.history` and ships
        # the full conversation in a single service call. No attach
        # handshake — `prepare_dialogue` (when reachable) was already
        # called at spawn time, fire-and-forget.
        speaker_dialogue = recipients[0] if recipients else None
        chatbot_reachable = (
            self._chatbot_enabled
            and self._chatbot_client is not None
            and self._chatbot_client.is_available(timeout_sec=0.0)
        )
        if chatbot_reachable and speaker_dialogue is not None:
            self._node.get_logger().info(
                f'[SPEECH INPUT] Triggering chatbot turn for dialogue '
                f'{speaker_dialogue.dialogue_id}'
            )
            self._chatbot_client.interact(speaker_dialogue)
            return

        self._node.get_logger().info(
            '[SPEECH INPUT] Publishing as RAW_USER_INPUT'
        )
        self._publish_raw_intent(msg.final, voice_id, msg.locale)

    def _recipient_dialogues_for_speaker(self, voice_id: str) -> list[Dialogue]:
        """
        Return every dialogue that should receive `voice_id`'s utterance.

        Walks the interlocutor fan-out graph from the speaker. Missing
        person/group dialogues are auto-spawned when default-chat is
        enabled. Co-members whose voice is no longer tracked are
        skipped *before* spawn so we don't materialise dialogues for
        people who have already left.
        """
        result: list[Dialogue] = []
        speaker_il = Interlocutor(person_id=voice_id)
        for it in related_interlocutors(speaker_il, self._group_handler):
            # Filter absent co-members before spawn. Don't filter the
            # speaker — they just spoke, presence is implicit. Group
            # dialogues are never gated on voice tracking.
            if (it.person_id and it.person_id != voice_id
                    and not self.is_voice_tracked(it.person_id)):
                continue
            d = self._get_or_spawn_dialogue_for(it)
            if d is not None and d not in result:
                result.append(d)
        return result

    def _get_or_spawn_dialogue_for(
        self, interlocutor: Interlocutor
    ) -> Dialogue | None:
        """Route to the right spawn helper based on interlocutor kind."""
        if interlocutor.is_group:
            return self._get_or_spawn_group_dialogue(interlocutor.group_id)
        if interlocutor.person_id:
            return self._get_or_spawn_person_dialogue(interlocutor.person_id)
        return None

    # ------------------------------------------------------------ spawning

    def _get_or_spawn_person_dialogue(self, person_id: str) -> Dialogue | None:
        """
        Return the active person dialogue for `person_id`, spawning if needed.

        Returns None if default-chat is disabled and no dialogue exists.
        Spawned person dialogues are warmed up against the chatbot
        (best-effort fire-and-forget) when one is configured.
        """
        existing = self._dialogue_manager.get_dialogue_for_interlocutor(
            Interlocutor(person_id=person_id)
        )
        if existing is not None:
            return existing
        if not self._default_chat_role:
            return None
        return self._spawn_dialogue(
            Interlocutor(person_id=person_id),
            warm_up_chatbot=True,
        )

    def _get_or_spawn_group_dialogue(self, group_id: str) -> Dialogue | None:
        """
        Return the active group dialogue, spawning if needed.

        Group dialogues are observational containers — the chatbot is
        currently a per-person conversation partner and is not warmed
        up for the group.
        """
        existing = self._dialogue_manager.get_dialogue_for_interlocutor(
            Interlocutor(group_id=group_id)
        )
        if existing is not None:
            return existing
        if not self._default_chat_role:
            return None
        return self._spawn_dialogue(
            Interlocutor(group_id=group_id),
            warm_up_chatbot=False,
        )

    def _spawn_dialogue(
        self, interlocutor: Interlocutor, warm_up_chatbot: bool
    ) -> Dialogue:
        """Create + register a fresh default-chat Dialogue for `interlocutor`."""
        role = DialogueRole()
        role.name = self._default_chat_role
        role.configuration = self._default_chat_role_config
        dialogue = Dialogue(
            role=role,
            interlocutor=interlocutor,
            priority=DEFAULT_CHAT_PRIORITY,
            state=DialogueState.ACTIVE,
        )
        if self._conversations_store.preload_into(dialogue, self._now()):
            self._node.get_logger().info(
                f'[DEFAULT CHAT] Pre-filled prior summary for {interlocutor.key}'
            )
        self._dialogue_manager.add_dialogue(dialogue)
        self._node.get_logger().info(
            f'[DEFAULT CHAT] Spawned dialogue {dialogue.dialogue_id} for '
            f'{interlocutor.key} (role="{role.name}")'
        )
        # Fire-and-forget warm-up. The chatbot is stateless: if the
        # warm-up service call fails or arrives late, the first
        # `interact()` will still work — the backend has no per-dialogue
        # state to be missing.
        if (warm_up_chatbot
                and self._chatbot_client is not None
                and self._chatbot_enabled):
            self._chatbot_client.prepare(dialogue)
        return dialogue

    # ------------------------------------------------------------ helpers

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
