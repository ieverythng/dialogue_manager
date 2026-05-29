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

import time

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from hri_actions_msgs.msg import ClosedCaption, Intent
from planner_common import parse_json_object
from planner_common import PlannerDialogueAct
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
        self._planner_dialogue_act_sub = None
        self._last_planner_dialogue_signature = ''
        self._last_planner_dialogue_ts = 0.0

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
        self.declare_parameter(
            'planner_dialogue_act_topic', '/planner/dialogue_act',
            ParameterDescriptor(
                description='Planner-owned dialogue-act topic for asynchronous execution feedback.'
            )
        )
        self.declare_parameter(
            'planner_dialogue_wording_mode', 'chatbot',
            ParameterDescriptor(
                description='How planner dialogue-act wording is produced: direct or chatbot.'
            )
        )
        self.declare_parameter(
            'planner_completion_wording_mode', 'chatbot',
            ParameterDescriptor(
                description='How notify_completion wording is produced: direct or chatbot.'
            )
        )
        self.declare_parameter(
            'planner_dialogue_dedupe_window_sec', 1.5,
            ParameterDescriptor(
                description='TTL window used to suppress duplicate planner dialogue acts.'
            )
        )
        self.declare_parameter(
            'use_llm_completion_wording', False,
            ParameterDescriptor(
                description='Legacy override: when true, route notify_completion wording through chatbot.'
            )
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

        planner_dialogue_act_topic = self.get_parameter(
            'planner_dialogue_act_topic'
        ).get_parameter_value().string_value
        if planner_dialogue_act_topic:
            self._planner_dialogue_act_sub = self.create_subscription(
                String,
                planner_dialogue_act_topic,
                self._on_planner_dialogue_act,
                10,
            )
            self.get_logger().info(
                '[ACTIVATE] Planner dialogue acts subscribed on "%s"'
                % planner_dialogue_act_topic
            )

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
        if self._planner_dialogue_act_sub is not None:
            self.destroy_subscription(self._planner_dialogue_act_sub)
            self._planner_dialogue_act_sub = None

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
        if self._planner_dialogue_act_sub is not None:
            self.destroy_subscription(self._planner_dialogue_act_sub)
            self._planner_dialogue_act_sub = None

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

    def _on_planner_dialogue_act(self, msg: String) -> None:
        """Speak planner-owned asynchronous dialogue acts without duplicating chatbot acks."""
        if self._tts_client is None:
            self.get_logger().warn('[PLANNER ACT] Ignored because TTS client is unavailable')
            return

        payload = parse_json_object(msg.data)
        if not payload:
            self.get_logger().warn('[PLANNER ACT] Ignored malformed or empty payload')
            return

        dialogue_act = PlannerDialogueAct.from_payload(payload)
        if self._is_duplicate_planner_dialogue_act(dialogue_act):
            self.get_logger().debug(
                '[PLANNER ACT] Ignoring duplicate act=%s goal_id=%s within dedupe window'
                % (dialogue_act.act, dialogue_act.goal_id)
            )
            return
        if dialogue_act.act == 'acknowledge':
            self.get_logger().debug(
                '[PLANNER ACT] Ignoring acknowledge act for goal_id=%s to avoid duplicate speech'
                % dialogue_act.goal_id
            )
            return

        if self._use_chatbot_planner_wording(dialogue_act) and self._ask_chatbot_for_planner_reply(
            dialogue_act
        ):
            return

        speech_text = _planner_dialogue_text(dialogue_act)
        if not speech_text:
            self.get_logger().debug(
                '[PLANNER ACT] No speech text resolved for act=%s goal_id=%s'
                % (dialogue_act.act, dialogue_act.goal_id)
            )
            return

        self.get_logger().info(
            '[PLANNER ACT] Speaking act=%s goal_id=%s await_user_response=%s'
            % (
                dialogue_act.act,
                dialogue_act.goal_id,
                dialogue_act.await_user_response,
            )
        )
        self._tts_client.speak(
            speech_text,
            priority=_planner_tts_priority(dialogue_act.priority),
        )

    def _use_chatbot_planner_wording(self, dialogue_act: PlannerDialogueAct) -> bool:
        """Return whether one planner dialogue act should be rendered by chatbot_llm."""
        mode = str(
            self.get_parameter('planner_dialogue_wording_mode').get_parameter_value().string_value
        ).strip().lower()
        if mode not in ('direct', 'chatbot'):
            self.get_logger().warn(
                '[PLANNER ACT] Invalid planner_dialogue_wording_mode=%s; falling back to direct'
                % mode
            )
            mode = 'direct'
        if mode == 'chatbot':
            return True
        if dialogue_act.act == 'notify_completion':
            return self._use_chatbot_completion_wording()
        return False

    def _use_chatbot_completion_wording(self) -> bool:
        """Return whether completion wording should be delegated to chatbot_llm."""
        use_llm_override = bool(
            self.get_parameter('use_llm_completion_wording').get_parameter_value().bool_value
        )
        if use_llm_override:
            return True

        mode = str(
            self.get_parameter('planner_completion_wording_mode').get_parameter_value().string_value
        ).strip().lower()
        if mode not in ('direct', 'chatbot'):
            self.get_logger().warn(
                '[PLANNER ACT] Invalid planner_completion_wording_mode=%s; falling back to direct'
                % mode
            )
            return False
        return mode == 'chatbot'

    def _ask_chatbot_for_planner_reply(self, dialogue_act: PlannerDialogueAct) -> bool:
        """Route planner dialogue wording through chatbot_llm using structured context."""
        if self._chatbot_client is None:
            return False
        dialogue_context = _planner_dialogue_context(dialogue_act)
        if not dialogue_context:
            return False
        sent = self._chatbot_client.send_planner_dialogue_context(dialogue_context)
        if sent:
            self.get_logger().info(
                '[PLANNER ACT] Requested chatbot wording for act=%s goal_id=%s'
                % (dialogue_act.act, dialogue_act.goal_id)
            )
        return sent

    def _is_duplicate_planner_dialogue_act(self, dialogue_act: PlannerDialogueAct) -> bool:
        window_sec = float(
            self.get_parameter('planner_dialogue_dedupe_window_sec').get_parameter_value().double_value
        )
        if window_sec <= 0.0:
            return False
        signature = _planner_dialogue_signature(dialogue_act)
        now = time.monotonic()
        if (
            signature
            and signature == self._last_planner_dialogue_signature
            and (now - self._last_planner_dialogue_ts) <= window_sec
        ):
            return True
        self._last_planner_dialogue_signature = signature
        self._last_planner_dialogue_ts = now
        return False


def _planner_dialogue_text(dialogue_act: PlannerDialogueAct) -> str:
    """Resolve planner dialogue text, preferring planner-provided wording."""
    act = str(dialogue_act.act or '').strip()
    text_hint = str(dialogue_act.text_hint or '').strip()
    if act in {'ask_clarification', 'ask_for_help'}:
        question_text = _planner_question_text(dialogue_act, text_hint)
        if question_text:
            return question_text
    if text_hint:
        return text_hint
    context = dict(dialogue_act.context or {})
    if act == 'notify_completion':
        result_payload = context.get('result_payload', {})
        if isinstance(result_payload, dict):
            summary_text = str(result_payload.get('summary_text', '')).strip()
            if summary_text:
                return summary_text
        result_summary = str(context.get('result_summary', '')).strip()
        if result_summary:
            return result_summary

    fallback_by_act = {
        'progress_update': 'I am working on it now.',
        'ask_clarification': 'I need a bit more detail before I continue.',
        'ask_for_help': 'I need help to continue this task.',
        'explain_failure': 'I could not complete that task.',
        'notify_completion': 'I finished that task.',
        'notify_cancellation': 'Okay, I will stop working on that.',
    }
    if act in {'ask_clarification', 'ask_for_help'}:
        question_text = _planner_question_text(
            dialogue_act,
            str(dialogue_act.reason or '').strip(),
        )
        if question_text:
            return question_text
    if act in {'explain_failure', 'ask_for_help'}:
        return fallback_by_act.get(act, '').strip()
    if dialogue_act.reason:
        return str(dialogue_act.reason).strip()
    return fallback_by_act.get(act, '').strip()


def _planner_question_text(dialogue_act: PlannerDialogueAct, base_text: str) -> str:
    """Render ask_* planner dialogue acts as explicit user-facing questions."""
    act = str(dialogue_act.act or '').strip()
    clean_base = str(base_text or '').strip()
    if clean_base.endswith('?'):
        return clean_base

    slots = [str(item).strip() for item in list(dialogue_act.slots_needed or []) if str(item).strip()]
    slot_phrase = _planner_slot_phrase(slots)

    if act == 'ask_clarification':
        question = (
            f'Could you clarify {slot_phrase}?'
            if slot_phrase
            else 'Could you clarify what you want me to do next?'
        )
    elif act == 'ask_for_help':
        question = (
            f'How should I proceed with {slot_phrase}?'
            if slot_phrase
            else 'What should I try next?'
        )
    else:
        return clean_base

    if clean_base:
        trimmed = clean_base.rstrip()
        if trimmed and trimmed[-1] not in '.!?':
            trimmed = f'{trimmed}.'
        return f'{trimmed} {question}'.strip()
    return question


def _planner_slot_phrase(slots: list[str]) -> str:
    """Turn planner slots into a short spoken phrase."""
    if not slots:
        return ''
    if len(slots) == 1:
        return slots[0]
    if len(slots) == 2:
        return f'{slots[0]} and {slots[1]}'
    return f"{', '.join(slots[:-1])}, and {slots[-1]}"


def _planner_completion_context(dialogue_act: PlannerDialogueAct) -> dict:
    """Extract structured completion context to be rendered by chatbot_llm."""
    context = dict(dialogue_act.context or {})
    goal_text = str(context.get('goal_text', '')).strip()
    result_summary = str(context.get('result_summary', '')).strip()
    result_payload = context.get('result_payload', {})
    if not result_summary and isinstance(result_payload, dict):
        result_summary = str(result_payload.get('summary_text', '')).strip()
    text_hint = str(dialogue_act.text_hint or '').strip()
    requested_intents = context.get('requested_intents', [])
    if not isinstance(requested_intents, list):
        requested_intents = []
    clean_intents = [
        str(item).strip()
        for item in requested_intents
        if str(item).strip()
    ]
    normalized_payload = result_payload if isinstance(result_payload, dict) else {}
    if not (goal_text or result_summary or text_hint or clean_intents or normalized_payload):
        return {}
    return {
        'goal_id': str(dialogue_act.goal_id or '').strip(),
        'goal_token': str(getattr(dialogue_act, 'goal_token', '') or dialogue_act.goal_id or '').strip(),
        'goal_text': goal_text,
        'result_summary': result_summary,
        'result_payload': normalized_payload,
        'text_hint': text_hint,
        'requested_intents': clean_intents,
    }


def _planner_dialogue_context(dialogue_act: PlannerDialogueAct) -> dict:
    """Extract one planner dialogue act context for chatbot_llm wording."""
    act = str(dialogue_act.act or '').strip().lower()
    if not act:
        return {}
    context = dict(dialogue_act.context or {})
    payload = {
        'act': act,
        'goal_id': str(dialogue_act.goal_id or '').strip(),
        'goal_token': str(getattr(dialogue_act, 'goal_token', '') or dialogue_act.goal_id or '').strip(),
        'plan_id': str(dialogue_act.plan_id or '').strip(),
        'plan_version': int(dialogue_act.plan_version or 0),
        'reason': str(dialogue_act.reason or '').strip(),
        'text_hint': str(dialogue_act.text_hint or '').strip(),
        'await_user_response': bool(dialogue_act.await_user_response),
        'slots_needed': [
            str(item).strip()
            for item in list(dialogue_act.slots_needed or [])
            if str(item).strip()
        ],
        'context': context,
    }
    if act == 'notify_completion':
        payload['completion_context'] = _planner_completion_context(dialogue_act)
    if not any(
        payload.get(key)
        for key in ('reason', 'text_hint', 'slots_needed', 'context', 'completion_context')
    ):
        return {}
    return payload


def _planner_tts_priority(priority_name: str) -> int:
    """Map planner dialogue priority labels onto the local TTS priority scale."""
    return {
        'low': 96,
        'normal': 128,
        'high': 192,
        'critical': 255,
    }.get(str(priority_name or '').strip().lower(), 128)


def _planner_dialogue_signature(dialogue_act: PlannerDialogueAct) -> str:
    """Build one stable signature used for duplicate planner act suppression."""
    parts = (
        str(dialogue_act.goal_id or '').strip(),
        str(getattr(dialogue_act, 'goal_token', '') or '').strip(),
        str(dialogue_act.plan_id or '').strip(),
        str(int(dialogue_act.plan_version or 0)),
        str(dialogue_act.act or '').strip().lower(),
        str(dialogue_act.text_hint or '').strip(),
        str(dialogue_act.reason or '').strip(),
        str(bool(dialogue_act.await_user_response)),
    )
    return '|'.join(parts).strip('|')
