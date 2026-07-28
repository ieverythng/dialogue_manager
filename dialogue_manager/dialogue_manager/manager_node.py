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

import json
import os
from pathlib import Path

try:  # pragma: no cover - runtime dependency
    from ament_index_python.packages import get_package_share_directory
except ImportError:  # pragma: no cover - unit-test fallback

    def get_package_share_directory(_package_name: str) -> str:
        raise RuntimeError('ament_index_python is unavailable in this environment')
from chatbot_msgs.msg import DialogueRole
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from hri_actions_msgs.msg import ClosedCaption, Intent
from planner_common import parse_json_object, PlannerDialogueAct
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.lifecycle import LifecycleNode, State, TransitionCallbackReturn
from std_msgs.msg import Bool, String

from .chatbot_client import ChatbotClient
from .conversations_history import ConversationsHistoryStore
from .debug_publisher import DebugStatePublisher
from .dialogue import (
    Dialogue,
    DialogueManager,
    DialogueState,
    Interlocutor,
    SYSTEM_SPEAKER_ID,
)
from .dialogue_lifecycle import DialogueLifecycle
from .group_handler import GroupHandler
from .markup import ActionLibrary, ExpressionExecutor
from .say_client import SayClient
from .skill_servers import SkillServers
from .speech_handler import SpeechHandler


# How often the in-memory conversation store is flushed to disk while the
# node is in the active state. Kept as a module constant (not a ROS param) to
# minimize configuration surface.
CONVERSATIONS_PERSIST_INTERVAL_SEC = 30.0
MAX_PLANNER_ACT_EMISSION_LEDGER = 256


class DialogueManagerNode(LifecycleNode):
    """
    Dialogue Manager ROS2 Lifecycle Node.

    Implements the chat, ask, and say skills for human-robot dialogue.
    Orchestrates chatbot, Say sub-skill, and speech input handling through
    composition of specialized handler classes.
    """

    def __init__(self) -> None:
        """Construct the node."""
        # enable_logger_service=True exposes ~/get_logger_levels and
        # ~/set_logger_levels services so log levels (including the
        # DEBUG dump in chatbot_client.interact()) can be flipped at
        # runtime without restarting the node.
        super().__init__('dialogue_manager', enable_logger_service=True)

        # Callback group for concurrent action handling
        self._callback_group = ReentrantCallbackGroup()

        # Core dialogue tracking
        self._dialogue_manager = DialogueManager()
        self._conversations_store: ConversationsHistoryStore | None = None

        # Handlers (created in on_configure)
        self._say_client: SayClient | None = None
        self._chatbot_client: ChatbotClient | None = None
        self._speech_handler: SpeechHandler | None = None
        self._skill_servers: SkillServers | None = None
        self._action_library: ActionLibrary | None = None
        self._expression_executor: ExpressionExecutor | None = None
        self._debug_publisher: DebugStatePublisher | None = None
        self._group_handler: GroupHandler | None = None
        self._lifecycle: DialogueLifecycle | None = None

        # Publishers (created in on_configure)
        self._closed_captions_pub = None
        self._robot_speech_pub = None
        self._waiting_chatbot_pub = None
        self._intents_pub = None
        self._diag_pub = None

        # Timers
        self._diag_timer = None
        self._persist_timer = None
        self._planner_dialogue_act_sub = None
        self._planner_act_emission_order: list[str] = []
        self._planner_act_emission_ledger: set[str] = set()

        self._declare_parameters()
        self.get_logger().info('Dialogue Manager node created, awaiting configuration.')

    def _declare_parameters(self) -> None:
        """Declare all ROS parameters."""
        self.declare_parameter(
            'chatbot', 'chatbot',
            ParameterDescriptor(description='Chatbot node FQN for action/service prefix')
        )
        self.declare_parameter(
            'say_action', '/tts/say',
            ParameterDescriptor(
                description='Action name of the Say sub-skill server (TTS frontend)'
            )
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
            'markup_libraries', ['config/00-default_actions.yaml'],
            ParameterDescriptor(description='Paths to markup action definitions')
        )
        self.declare_parameter(
            'disabled_markup_actions', ['motion'],
            ParameterDescriptor(description='Markup actions to skip')
        )
        self.declare_parameter(
            'conversations_storage_dir',
            '~/.ros/dialogue_manager/conversations',
            ParameterDescriptor(
                description='Directory where per-person/group conversation '
                'histories are persisted. Empty = in-memory only.'
            )
        )
        self.declare_parameter(
            'planner_dialogue_act_topic', '/planner/dialogue_act',
            ParameterDescriptor(
                description='Planner-owned dialogue-act topic for asynchronous execution feedback.'
            )
        )

    # =========================================================================
    # Lifecycle callbacks
    # =========================================================================

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        """Configure the node: create publishers and handlers."""
        self.get_logger().info('Configuring Dialogue Manager...')
        self._planner_act_emission_order.clear()
        self._planner_act_emission_ledger.clear()

        # Conversations history (load from disk if configured)
        storage_dir_param = (
            self.get_parameter('conversations_storage_dir')
            .get_parameter_value().string_value
        )
        storage_dir = Path(storage_dir_param).expanduser() if storage_dir_param else None
        self._conversations_store = ConversationsHistoryStore(
            storage_dir=storage_dir,
            logger=self.get_logger(),
        )
        try:
            self._conversations_store.load()
            self.get_logger().info(
                f'[CONFIGURE] Conversations store ready (dir={storage_dir})'
            )
        except Exception as exc:
            self.get_logger().warn(
                f'[CONFIGURE] Failed to load conversations history: {exc}'
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
        self._intents_pub = self.create_publisher(Intent, '/intents', 10)
        self._diag_pub = self.create_publisher(DiagnosticArray, '/diagnostics', 10)

        # Create Say sub-skill client
        self._say_client = SayClient(
            node=self,
            dialogue_manager=self._dialogue_manager,
            closed_captions_pub=self._closed_captions_pub,
            robot_speech_pub=self._robot_speech_pub,
            callback_group=self._callback_group
        )
        say_action = (
            self.get_parameter('say_action').get_parameter_value().string_value
        )
        self._say_client.create_client(say_action)
        self.get_logger().debug(
            f'[CONFIGURE] Say sub-skill client created (action="{say_action}")'
        )

        # Create action library and expression executor
        pkg_share = get_package_share_directory('dialogue_manager')
        markup_libs = [
            os.path.join(pkg_share, p)
            for p in self.get_parameter('markup_libraries')
            .get_parameter_value().string_array_value
        ]
        disabled_actions = (
            self.get_parameter('disabled_markup_actions')
            .get_parameter_value().string_array_value
        )
        action_timeout = (
            self.get_parameter('markup_action_timeout')
            .get_parameter_value().double_value
        )
        expr_timeout = (
            self.get_parameter('multi_modal_expression_timeout')
            .get_parameter_value().double_value
        )

        self._action_library = ActionLibrary(
            _node=self,
            _disabled_actions=list(disabled_actions),
            _default_timeout=action_timeout,
            _callback_group=self._callback_group,
        )
        self._action_library.load(list(markup_libs))
        self._action_library.create_clients()
        self.get_logger().debug('[CONFIGURE] Action library created')

        self._expression_executor = ExpressionExecutor(
            node=self,
            say_client=self._say_client,
            action_library=self._action_library,
            expression_timeout=expr_timeout,
        )
        self.get_logger().debug('[CONFIGURE] Expression executor created')

        # Create chatbot client (if configured)
        chatbot = self.get_parameter('chatbot').get_parameter_value().string_value
        if chatbot:
            self._chatbot_client = ChatbotClient(
                node=self,
                dialogue_manager=self._dialogue_manager,
                say_client=self._say_client,
                expression_executor=self._expression_executor,
                intents_pub=self._intents_pub,
                waiting_chatbot_pub=self._waiting_chatbot_pub,
                callback_group=self._callback_group
            )
            self._chatbot_client.create_clients(chatbot)
            self.get_logger().debug(f'[CONFIGURE] Chatbot client created for "{chatbot}"')
        else:
            self.get_logger().info('[CONFIGURE] No chatbot configured')

        # Create group handler (tracks ROS4HRI groups, drives group-aware
        # dialogue recording).
        self._group_handler = GroupHandler(
            node=self,
            on_group_dispersed=self._on_group_dispersed,
        )
        self.get_logger().debug('[CONFIGURE] Group handler created')

        # Create speech handler
        self._speech_handler = SpeechHandler(
            node=self,
            dialogue_manager=self._dialogue_manager,
            chatbot_client=self._chatbot_client,
            conversations_store=self._conversations_store,
            closed_captions_pub=self._closed_captions_pub,
            intents_pub=self._intents_pub,
            group_handler=self._group_handler,
        )
        self._speech_handler.set_chatbot_enabled(chatbot != '')
        self.get_logger().debug('[CONFIGURE] Speech handler created')

        # Dialogue lifecycle (finalize + archive + summarize). Shared by
        # SkillServers (Chat/Ask completion) and direct callers below
        # (group dispersal, deactivation).
        self._lifecycle = DialogueLifecycle(
            node=self,
            dialogue_manager=self._dialogue_manager,
            conversations_store=self._conversations_store,
            group_handler=self._group_handler,
        )

        # Create skill servers
        self._skill_servers = SkillServers(
            node=self,
            dialogue_manager=self._dialogue_manager,
            chatbot_client=self._chatbot_client,
            say_client=self._say_client,
            conversations_store=self._conversations_store,
            expression_executor=self._expression_executor,
            closed_captions_pub=self._closed_captions_pub,
            callback_group=self._callback_group,
            group_handler=self._group_handler,
            presence_query=self._speech_handler.is_voice_tracked,
            lifecycle=self._lifecycle,
        )
        self._skill_servers.create_servers()
        self.get_logger().debug('[CONFIGURE] Skill servers created')

        # Debug-state publisher (on-change snapshot for rqt_dialogues etc.).
        self._debug_publisher = DebugStatePublisher(
            node=self,
            dialogue_manager=self._dialogue_manager,
            conversations_store=self._conversations_store,
            chatbot_status=self._chatbot_status_snapshot,
            active_status=lambda: (
                self._skill_servers is not None
                and self._skill_servers._is_active
            ),
        )
        self._debug_publisher.create_publisher()
        self._dialogue_manager.set_change_callback(
            lambda did=None: self._debug_publisher.notify(did)
        )
        self._debug_publisher.notify()  # initial snapshot

        # Diagnostics timer
        self._diag_timer = self.create_timer(1.0, self._publish_diagnostics)

        self.get_logger().info('Dialogue Manager configured.')
        return TransitionCallbackReturn.SUCCESS

    def _chatbot_status_snapshot(self) -> dict:
        """Return the current chatbot status for debug snapshots."""
        if self._chatbot_client is None:
            return {
                'configured': False,
                'available': False,
                'waiting_for_response': False,
            }
        return {
            'configured': True,
            'available': self._chatbot_client.is_available(timeout_sec=0.0),
            'waiting_for_response': self._chatbot_client.waiting_for_response,
        }

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        """Activate the node: subscribe to speech, optionally start default chat."""
        self.get_logger().info('Activating Dialogue Manager...')

        # Enable skill servers
        self._skill_servers.set_active(True)

        # Subscribe to voices and groups
        self._speech_handler.subscribe_to_voices()
        if self._group_handler is not None:
            self._group_handler.subscribe()

        # Start periodic conversation-history persistence.
        if self._conversations_store is not None and self._persist_timer is None:
            self._persist_timer = self.create_timer(
                CONVERSATIONS_PERSIST_INTERVAL_SEC,
                self._persist_conversations,
            )
            self.get_logger().debug(
                f'[ACTIVATE] Persistence timer started '
                f'(interval={CONVERSATIONS_PERSIST_INTERVAL_SEC}s)'
            )

        # Enable per-person default-chat auto-spawning in the SpeechHandler.
        # The first utterance from a new speaker spawns a fresh Dialogue;
        # if a chatbot is configured, a chatbot dialogue is attached
        # asynchronously.
        if self.get_parameter('enable_default_chat').get_parameter_value().bool_value:
            role_name = (
                self.get_parameter('default_chat_role')
                .get_parameter_value().string_value
            )
            role_config = (
                self.get_parameter('default_chat_configuration')
                .get_parameter_value().string_value
            )
            self._speech_handler.set_default_chat(role_name, role_config)
            self.get_logger().info(
                f'[DEFAULT CHAT] Per-person auto-spawning enabled '
                f'(role="{role_name}")'
            )

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
                f'[ACTIVATE] Planner dialogue acts subscribed on "{planner_dialogue_act_topic}"'
            )

        if self._debug_publisher is not None:
            self._debug_publisher.notify()  # active state changed

        self.get_logger().info('Dialogue Manager activated.')
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        """Deactivate the node: unsubscribe, disable servers."""
        self.get_logger().info('Deactivating Dialogue Manager...')

        # Disable skill servers
        self._skill_servers.set_active(False)

        # Unsubscribe from voices and groups
        self._speech_handler.unsubscribe_all()
        if self._group_handler is not None:
            self._group_handler.unsubscribe()

        if self._planner_dialogue_act_sub is not None:
            self.destroy_subscription(self._planner_dialogue_act_sub)
            self._planner_dialogue_act_sub = None

        if self._debug_publisher is not None:
            self._debug_publisher.notify()  # active state changed

        # Stop the persistence timer.
        if self._persist_timer is not None:
            self.destroy_timer(self._persist_timer)
            self._persist_timer = None

        # Finalize (and trigger summarization + archival of) any active
        # dialogues before clearing them, so their history isn't lost.
        for dialogue in list(self._dialogue_manager.active_dialogues.values()):
            if dialogue.state == DialogueState.COMPLETED:
                continue
            try:
                self._lifecycle.finalize_and_archive(dialogue)
            except Exception as exc:
                self.get_logger().warn(
                    f'[DEACTIVATE] Failed to finalize dialogue '
                    f'{dialogue.dialogue_id}: {exc}'
                )
        self._dialogue_manager.clear_all()

        # One immediate flush to capture utterances that arrived since the
        # last periodic tick. (Summaries that are still running may land
        # after this and will be picked up on the next activation's first
        # tick, or by the on_shutdown bulk save.)
        self._persist_conversations()

        self.get_logger().info('Dialogue Manager deactivated.')
        return super().on_deactivate(state)

    def _persist_conversations(self) -> None:
        """
        Snapshot active bound dialogues into the store, then save to disk.

        Without the snapshot step, long-lived dialogues never reach
        disk between activate and deactivate (the store only contains
        explicitly-finalised dialogues). Archive is idempotent — the
        snapshot just registers active dialogues as in-progress entries
        under their interlocutor's history; subsequent saves pick up
        new utterances directly because bucket entries are references
        to the live Dialogue objects.
        """
        if self._conversations_store is None:
            return
        for dialogue in self._dialogue_manager.active_dialogues.values():
            if not dialogue.interlocutor.is_bound:
                continue
            if not dialogue.session_utterances:
                continue
            members = (
                self._resolve_group_members(dialogue.interlocutor.group_id)
                if dialogue.interlocutor.is_group else None
            )
            try:
                self._conversations_store.archive(
                    dialogue, group_members=members
                )
            except Exception as exc:
                self.get_logger().warn(
                    f'[PERSIST] Snapshot of dialogue '
                    f'{dialogue.dialogue_id} failed: {exc}'
                )
        try:
            self._conversations_store.save()
        except Exception as exc:
            self.get_logger().warn(
                f'[PERSIST] Failed to persist conversations: {exc}'
            )

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        """Shutdown the node: destroy all handlers."""
        self.get_logger().info('Shutting down Dialogue Manager...')

        if self._diag_timer:
            self.destroy_timer(self._diag_timer)
        if self._persist_timer is not None:
            self.destroy_timer(self._persist_timer)
            self._persist_timer = None

        if self._conversations_store is not None:
            try:
                self._conversations_store.save()
                self.get_logger().info(
                    '[SHUTDOWN] Conversations history persisted'
                )
            except Exception as exc:
                self.get_logger().warn(
                    f'[SHUTDOWN] Failed to persist conversations history: {exc}'
                )

        if self._skill_servers:
            self._skill_servers.destroy()
        if self._action_library:
            self._action_library.destroy()
        if self._say_client:
            self._say_client.destroy()
        if self._chatbot_client:
            self._chatbot_client.destroy()
        if self._planner_dialogue_act_sub is not None:
            self.destroy_subscription(self._planner_dialogue_act_sub)
            self._planner_dialogue_act_sub = None
        if self._debug_publisher is not None:
            self._dialogue_manager.set_change_callback(None)
            self._debug_publisher.destroy()
            self._debug_publisher = None

        self.get_logger().info('Dialogue Manager shutdown complete.')
        return TransitionCallbackReturn.SUCCESS

    # =========================================================================
    # Helpers
    # =========================================================================

    def _resolve_group_members(self, group_id: str) -> list[str]:
        """Resolve a ROS4HRI group ID to its current member person IDs."""
        if self._group_handler is None:
            return []
        return self._group_handler.members_of(group_id)

    def _on_group_dispersed(self, group_id: str) -> None:
        """Finalize and archive a dispersed group's dialogue, if any."""
        if self._lifecycle is None:
            return
        dialogue = self._dialogue_manager.get_dialogue_for_interlocutor(
            Interlocutor(group_id=group_id)
        )
        if dialogue is None:
            return
        try:
            self._lifecycle.finalize_and_archive(dialogue)
        except Exception as exc:
            self.get_logger().warn(
                f'[GROUPS] Failed to finalize group {group_id} dialogue: {exc}'
            )

    def _on_planner_dialogue_act(self, msg: String) -> None:
        """Route planner-owned asynchronous dialogue acts through chatbot wording."""
        if self._say_client is None:
            self.get_logger().warn('[PLANNER ACT] Ignored because Say client is unavailable')
            return

        payload = parse_json_object(msg.data)
        if not payload:
            self.get_logger().warn('[PLANNER ACT] Ignored malformed or empty payload')
            return

        dialogue_act = PlannerDialogueAct.from_payload(payload)
        if dialogue_act.act == 'acknowledge':
            self.get_logger().debug(
                '[PLANNER ACT] Ignoring acknowledge act for goal_id=%s'
                % dialogue_act.goal_id
            )
            return

        signature = _planner_dialogue_act_signature(dialogue_act)
        if signature in self._planner_act_emission_ledger:
            self.get_logger().warn(
                '[PLANNER ACT] Ignored duplicate semantic event act=%s goal_id=%s'
                % (dialogue_act.act, dialogue_act.goal_id)
            )
            return
        self._remember_planner_act_emission(signature)

        if self._ask_chatbot_for_planner_reply(dialogue_act):
            return

        fallback_text = _planner_safe_fallback_text(dialogue_act)
        if fallback_text:
            self.get_logger().warn(
                '[PLANNER ACT] Chatbot relay unavailable; speaking safe fallback '
                'act=%s goal_id=%s'
                % (dialogue_act.act, dialogue_act.goal_id)
            )
            self._say_client.speak(
                fallback_text,
                priority=_planner_say_priority(dialogue_act.priority),
            )
            return

        self.get_logger().warn(
            '[PLANNER ACT] act=%s goal_id=%s ignored because chatbot relay was unavailable'
            % (dialogue_act.act, dialogue_act.goal_id)
        )

    def _remember_planner_act_emission(self, signature: str) -> None:
        """Remember one semantic planner act in a bounded replay ledger."""
        if not signature:
            return
        self._planner_act_emission_order.append(signature)
        self._planner_act_emission_ledger.add(signature)
        if len(self._planner_act_emission_order) <= MAX_PLANNER_ACT_EMISSION_LEDGER:
            return
        expired = self._planner_act_emission_order.pop(0)
        self._planner_act_emission_ledger.discard(expired)

    def _ask_chatbot_for_planner_reply(self, dialogue_act: PlannerDialogueAct) -> bool:
        """Route planner facts through chatbot_llm for user-facing wording."""
        if self._chatbot_client is None:
            return False
        planner_context = _planner_chatbot_context(dialogue_act)
        if not planner_context:
            return False

        dialogue = self._planner_target_dialogue()
        if dialogue is None:
            return False

        payload = planner_context
        dialogue.add_utterance(
            SYSTEM_SPEAKER_ID,
            json.dumps(payload, sort_keys=True, separators=(',', ':')),
            self.get_clock().now().nanoseconds / 1e9,
        )

        sent = self._chatbot_client.interact(dialogue)
        if sent:
            self.get_logger().info(
                '[PLANNER ACT] Requested chatbot wording for act=%s goal_id=%s'
                % (dialogue_act.act, dialogue_act.goal_id)
            )
        return sent

    def _planner_target_dialogue(self) -> Dialogue | None:
        """Pick an active dialogue for planner system turns, or bootstrap one."""
        candidates = [
            d for d in self._dialogue_manager.active_dialogues.values()
            if d.state in (DialogueState.ACTIVE, DialogueState.WAITING_RESPONSE)
        ]
        if candidates:
            # Prefer the most recently active dialogue.
            return max(
                candidates,
                key=lambda d: d.history[-1].timestamp if d.history else (d.started_at or 0.0),
            )

        role = DialogueRole(name='__default__')
        dialogue = Dialogue(
            role=role,
            interlocutor=Interlocutor(person_id='__system__'),
            priority=128,
            state=DialogueState.ACTIVE,
        )
        self._dialogue_manager.add_dialogue(dialogue)
        self._chatbot_client.prepare(dialogue)
        return dialogue

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
    plan_outcome_summary = context.get('plan_outcome_summary', {})
    normalized_plan_outcome = (
        plan_outcome_summary if isinstance(plan_outcome_summary, dict) else {}
    )
    if not (
        goal_text
        or result_summary
        or text_hint
        or clean_intents
        or normalized_payload
        or normalized_plan_outcome
    ):
        return {}
    return {
        'goal_id': str(dialogue_act.goal_id or '').strip(),
        'goal_text': goal_text,
        'result_summary': result_summary,
        'result_payload': normalized_payload,
        'text_hint': text_hint,
        'requested_intents': clean_intents,
        'plan_outcome_summary': normalized_plan_outcome,
    }


def _planner_chatbot_context(dialogue_act: PlannerDialogueAct) -> dict:
    """Build the chatbot system-turn payload for a planner dialogue act."""
    if dialogue_act.act == 'notify_completion':
        completion_context = _planner_completion_context(dialogue_act)
        return {'planner_completion': completion_context} if completion_context else {}

    context = {
        'act': str(dialogue_act.act or '').strip(),
        'goal_id': str(dialogue_act.goal_id or '').strip(),
        'plan_id': str(dialogue_act.plan_id or '').strip(),
        'plan_version': int(dialogue_act.plan_version or 0),
        'reason': str(dialogue_act.reason or '').strip(),
        'text_hint': str(dialogue_act.text_hint or '').strip(),
        'await_user_response': bool(dialogue_act.await_user_response),
        'slots_needed': [
            str(item).strip()
            for item in dialogue_act.slots_needed
            if str(item).strip()
        ],
        'context': dict(dialogue_act.context or {}),
    }
    return {'planner_dialogue': context} if context['act'] else {}


def _planner_dialogue_act_signature(dialogue_act: PlannerDialogueAct) -> str:
    """Build a semantic planner-act key while allowing distinct progress messages."""
    payload = {
        'goal_id': str(dialogue_act.goal_id or '').strip(),
        'plan_id': str(dialogue_act.plan_id or '').strip(),
        'plan_version': int(dialogue_act.plan_version or 0),
        'act': str(dialogue_act.act or '').strip(),
    }
    if payload['act'] == 'progress_update':
        payload.update(
            {
                'reason': str(dialogue_act.reason or '').strip(),
                'text_hint': str(dialogue_act.text_hint or '').strip(),
                'context': dict(dialogue_act.context or {}),
            }
        )
    elif payload['act'] == 'ask_clarification':
        payload['slots_needed'] = sorted(
            str(slot).strip()
            for slot in dialogue_act.slots_needed
            if str(slot).strip()
        )
    return json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def _planner_safe_fallback_text(dialogue_act: PlannerDialogueAct) -> str:
    """Return bounded user-facing wording without exposing planner/model prose."""
    return {
        'progress_update': 'I am working on it now.',
        'ask_clarification': 'I need a bit more detail before I continue.',
        'ask_for_help': 'I need help to continue this task.',
        'explain_failure': 'I could not complete that task.',
        'notify_completion': 'I finished that task.',
        'notify_cancellation': 'Okay, I will stop working on that.',
    }.get(str(dialogue_act.act or '').strip().lower(), '')


def _planner_say_priority(priority_name: str) -> int:
    """Map planner dialogue priority labels onto the local Say priority scale."""
    return {
        'low': 96,
        'normal': 128,
        'high': 192,
        'critical': 255,
    }.get(str(priority_name or '').strip().lower(), 128)
