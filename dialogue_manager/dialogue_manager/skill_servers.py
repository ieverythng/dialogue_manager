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

import asyncio
from collections.abc import Awaitable, Callable
import json
import threading
import time

from chatbot_msgs.msg import DialogueRole
from communication_skills.action import Ask, Chat, Say
from hri_actions_msgs.msg import ClosedCaption
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.publisher import Publisher

from .chatbot_client import ChatbotClient
from .conversations_history import ConversationsHistoryStore
from .dialogue import (
    Dialogue,
    DialogueManager,
    DialogueState,
    Interlocutor,
    ROBOT_SPEAKER_ID,
    SAY_ROLE_NAME,
)
from .markup.executor import ExpressionExecutor
from .say_client import SayClient


# When Ask is used without a chatbot, this is how long we wait for the next
# user utterance from the addressed interlocutor before returning empty.
ASK_TIMEOUT_SEC = 5.0


GroupResolver = Callable[[str], list[str]]
"""Callable mapping a group ID to its member person IDs (may return [])."""

Summarizer = Callable[[Dialogue, list[Dialogue]], Awaitable[str]]
"""Async callable that produces a summary text for a finished dialogue.

Called at session end with the current dialogue and the list of prior
archived dialogues for the same interlocutor (chronological, oldest first).
Implementations are free to consult the prior dialogues to produce a
cumulative summary, or to ignore them.
"""


async def default_summarizer(
    dialogue: Dialogue, prior_dialogues: list[Dialogue]
) -> str:
    """Trivial summarizer: render the session's utterances as plain text."""
    lines = []
    for utt in dialogue.session_utterances:
        lines.append(f'{utt.speaker_id}: {utt.text}')
    return '\n'.join(lines)


class SkillServers:
    """
    Manages Chat, Ask, and Say action servers.

    Handles goal acceptance, execution, and cancellation for all skills.
    """

    def __init__(
        self,
        node: Node,
        dialogue_manager: DialogueManager,
        chatbot_client: ChatbotClient | None,
        say_client: SayClient,
        conversations_store: ConversationsHistoryStore,
        group_resolver: GroupResolver,
        expression_executor: ExpressionExecutor | None = None,
        closed_captions_pub: Publisher | None = None,
        callback_group: ReentrantCallbackGroup | None = None,
        summarizer: Summarizer | None = None,
    ):
        """Initialize skill servers."""
        self._node = node
        self._dialogue_manager = dialogue_manager
        self._chatbot_client = chatbot_client
        self._say_client = say_client
        self._conversations_store = conversations_store
        self._group_resolver = group_resolver
        self._expression_executor = expression_executor
        self._closed_captions_pub = closed_captions_pub
        self._callback_group = callback_group
        self._summarizer: Summarizer = summarizer or default_summarizer

        self._chat_server: ActionServer | None = None
        self._ask_server: ActionServer | None = None
        self._say_server: ActionServer | None = None
        self._is_active = False

    def _now(self) -> float:
        """Return the current time in epoch seconds, from the node clock."""
        return self._node.get_clock().now().nanoseconds / 1e9

    def _resolve_group_members(self, group_id: str) -> list[str]:
        try:
            return list(self._group_resolver(group_id) or [])
        except Exception as exc:
            self._node.get_logger().warn(
                f'[SKILLS] group resolver failed for "{group_id}": {exc}'
            )
            return []

    def _finalize_and_archive(self, dialogue: Dialogue) -> None:
        """Mark a dialogue completed, archive it, and remove it from tracking.

        Summarization runs asynchronously in a daemon thread so the calling
        coroutine isn't blocked by a slow (e.g. LLM-backed) summarizer. The
        summary lands on `dialogue.summary` when ready and is picked up by the
        next periodic persistence tick.
        """
        dialogue.state = DialogueState.COMPLETED
        dialogue.ended_at = self._now()
        members = (
            self._resolve_group_members(dialogue.interlocutor.group_id)
            if dialogue.interlocutor.is_group
            else None
        )
        # Snapshot prior dialogues BEFORE archive(), so the summarizer sees
        # only past sessions (not the one being summarized).
        prior_dialogues: list[Dialogue] = []
        if dialogue.interlocutor.is_bound and dialogue.session_utterances:
            prior_dialogues = self._conversations_store.history_for(
                dialogue.interlocutor
            )
        self._conversations_store.archive(dialogue, group_members=members)
        self._dialogue_manager.remove_dialogue(dialogue.dialogue_id)

        # Kick off async summarization (no-op if nothing to summarize).
        if dialogue.interlocutor.is_bound and dialogue.session_utterances:
            self._kick_off_summarization(dialogue, prior_dialogues)

    def _kick_off_summarization(
        self, dialogue: Dialogue, prior_dialogues: list[Dialogue]
    ) -> None:
        """Run the summarizer for `dialogue` in a daemon thread."""
        def runner():
            try:
                summary = asyncio.run(
                    self._summarizer(dialogue, prior_dialogues)
                )
            except Exception as exc:
                self._node.get_logger().warn(
                    f'[SKILLS] Summarizer failed for dialogue '
                    f'{dialogue.dialogue_id}: {exc}'
                )
                return
            dialogue.summary = summary
            dialogue.summary_generated_at = self._now()
            # The archived dialogue lives in the conversations store; notify
            # observers so the snapshot reflects the freshly-landed summary.
            self._dialogue_manager.notify_change(dialogue.dialogue_id)
            self._node.get_logger().debug(
                f'[SKILLS] Summary stored for dialogue {dialogue.dialogue_id}'
            )

        threading.Thread(target=runner, daemon=True).start()

    def _preload_summary(self, dialogue: Dialogue) -> None:
        """Pre-fill `dialogue.history` with prior summary, if any."""
        if self._conversations_store.preload_into(dialogue, self._now()):
            self._node.get_logger().info(
                f'[SKILLS] Pre-filled dialogue {dialogue.dialogue_id} with '
                f'prior summary for {dialogue.interlocutor.key}'
            )

    def _inject_initial_context(self, dialogue: Dialogue) -> None:
        """Build and push the conversation context to the chatbot at start."""
        if not dialogue.interlocutor.is_bound:
            return
        if self._chatbot_client is None:
            return
        context = self._conversations_store.build_context(
            dialogue.interlocutor,
            now=self._now(),
        )
        if context:
            self._chatbot_client.inject_context(dialogue.dialogue_id, context)

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
        """Execute Chat action.

        Works with or without a chatbot backend. Without a chatbot the
        Dialogue is a passive container: user utterances are recorded by the
        SpeechHandler, the controlling script handles intent generation, and
        Say goals can be addressed to the interlocutor to speak back.
        """
        request = goal_handle.request
        self._node.get_logger().info(f'[CHAT] Executing with role="{request.role.name}"')

        result = Chat.Result()
        chatbot_available = (
            self._chatbot_client is not None
            and self._chatbot_client.is_available(timeout_sec=1.0)
        )

        # Create dialogue (regardless of chatbot availability).
        dialogue = Dialogue(
            role=request.role,
            interlocutor=Interlocutor(
                person_id=request.person_id,
                group_id=request.group_id,
            ),
            priority=request.meta.priority,
            state=DialogueState.PENDING,
            goal_handle=goal_handle
        )
        self._preload_summary(dialogue)
        self._dialogue_manager.add_dialogue(dialogue)
        self._node.get_logger().info(f'[CHAT] Created dialogue {dialogue.dialogue_id}')

        chatbot_handle = None
        if chatbot_available:
            chatbot_handle = await self._chatbot_client.start_dialogue(request.role)
            if not chatbot_handle:
                result.result.error_code = 134
                result.result.error_msg = 'Chatbot rejected dialogue'
                self._dialogue_manager.remove_dialogue(dialogue.dialogue_id)
                goal_handle.abort()
                return result
            from uuid import UUID as PyUUID
            chatbot_goal_id = PyUUID(bytes=bytes(chatbot_handle.goal_id.uuid))
            dialogue.chatbot_goal_id = chatbot_goal_id
            self._node.get_logger().debug(
                f'[CHAT] Assigned chatbot_goal_id={chatbot_goal_id} to dialogue'
            )
        else:
            self._node.get_logger().info(
                '[CHAT] No chatbot available; running as passive dialogue '
                '(user utterances will be recorded, controlling script drives '
                'responses via RAW_USER_INPUT intent + Say goals)'
            )

        dialogue.state = DialogueState.ACTIVE
        self._dialogue_manager.notify_change(dialogue.dialogue_id)
        self._node.get_logger().info(f'[CHAT] Dialogue {dialogue.dialogue_id} now ACTIVE')

        # Prime the chatbot (no-op if chatbot is absent).
        self._inject_initial_context(dialogue)

        # If initiate=true, generate initial utterance.
        if request.initiate:
            if request.initial_input:
                self._node.get_logger().info('[CHAT] Speaking initial input')
                self._say_client.speak(
                    request.initial_input,
                    priority=request.meta.priority,
                    person_id=request.person_id,
                    group_id=request.group_id,
                )
            elif chatbot_available:
                self._node.get_logger().info('[CHAT] Requesting chatbot greeting')
                self._chatbot_client.send_input(
                    dialogue.dialogue_id,
                    '__assistant__',
                    ''  # Empty input triggers generation
                )
            else:
                self._node.get_logger().info(
                    '[CHAT] initiate=true but no chatbot and no initial_input; '
                    'nothing to speak'
                )

        # Wait for dialogue to complete or be cancelled.
        self._node.get_logger().debug('[CHAT] Waiting for completion or cancellation')
        while not goal_handle.is_cancel_requested:
            time.sleep(0.1)
            if dialogue.state == DialogueState.COMPLETED:
                break

        if chatbot_handle is not None:
            chatbot_handle.cancel_goal_async()

        # Mark completed, archive, summarize, and clean up tracking.
        self._finalize_and_archive(dialogue)

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
        """Execute Ask action.

        With a chatbot: speaks the question, hands schema validation off to
        the chatbot, returns schema-shaped answers.

        Without a chatbot: speaks the question and waits up to
        ASK_TIMEOUT_SEC for the next user utterance from the addressed
        interlocutor. The raw text is returned as a single free-text answer;
        `answers_schema` is ignored in this mode.
        """
        request = goal_handle.request
        self._node.get_logger().info(f'[ASK] Executing: "{request.question}"')

        result = Ask.Result()

        role = DialogueRole()
        role.name = DialogueRole.ASK_ROLE
        role.configuration = json.dumps({
            'question': request.question,
            'result_schema_properties': json.loads(request.answers_schema)
            if request.answers_schema else {}
        })

        chatbot_available = (
            self._chatbot_client is not None
            and self._chatbot_client.is_available(timeout_sec=0.1)
        )

        dialogue = Dialogue(
            role=role,
            interlocutor=Interlocutor(
                person_id=request.person_id,
                group_id=request.group_id,
            ),
            priority=request.meta.priority,
            state=DialogueState.PENDING,
            goal_handle=goal_handle
        )
        # ASK dialogues are excluded from context build by default so we do
        # not pre-fill with prior summaries here.
        self._dialogue_manager.add_dialogue(dialogue)

        chatbot_handle = None
        if chatbot_available:
            chatbot_handle = await self._chatbot_client.start_dialogue(role)
            if not chatbot_handle:
                result.result.error_code = 134
                result.result.error_msg = 'Chatbot rejected dialogue'
                self._dialogue_manager.remove_dialogue(dialogue.dialogue_id)
                goal_handle.abort()
                return result
            from uuid import UUID as PyUUID
            chatbot_goal_id = PyUUID(bytes=bytes(chatbot_handle.goal_id.uuid))
            dialogue.chatbot_goal_id = chatbot_goal_id
            self._node.get_logger().debug(
                f'[ASK] Assigned chatbot_goal_id={chatbot_goal_id} to dialogue'
            )

        dialogue.state = DialogueState.ACTIVE
        self._dialogue_manager.notify_change(dialogue.dialogue_id)

        self._node.get_logger().info('[ASK] Speaking question via Say sub-skill')
        self._say_client.speak(
            request.question,
            priority=request.meta.priority,
            person_id=request.person_id,
            group_id=request.group_id,
        )
        dialogue.add_utterance(
            ROBOT_SPEAKER_ID, request.question, self._now()
        )

        if chatbot_available:
            await self._await_chatbot_ask_answers(
                goal_handle, dialogue, chatbot_handle, result
            )
        else:
            self._await_raw_ask_answer(goal_handle, dialogue, result)

        self._finalize_and_archive(dialogue)

        if goal_handle.is_cancel_requested:
            if chatbot_handle is not None:
                chatbot_handle.cancel_goal_async()
            goal_handle.canceled()
            result.result.error_code = 125
            return result

        goal_handle.succeed()
        return result

    async def _await_chatbot_ask_answers(
        self,
        goal_handle,
        dialogue: Dialogue,
        chatbot_handle,
        result: Ask.Result,
    ) -> None:
        """Wait for the chatbot dialogue to complete and capture answers."""
        while not goal_handle.is_cancel_requested:
            time.sleep(0.1)
            if dialogue.state == DialogueState.COMPLETED:
                break
        try:
            chatbot_result_future = chatbot_handle.get_result_async()
            chatbot_result = await chatbot_result_future
            result.answers = chatbot_result.result.results
            self._node.get_logger().info(f'[ASK] Got answers: {result.answers}')
        except Exception as e:
            self._node.get_logger().warn(f'[ASK] Failed to get results: {e}')

    def _await_raw_ask_answer(
        self,
        goal_handle,
        dialogue: Dialogue,
        result: Ask.Result,
    ) -> None:
        """Wait up to ASK_TIMEOUT_SEC for the next user utterance.

        Treats the first non-robot utterance recorded against this dialogue
        as the answer. Times out silently with empty `answers` if no
        utterance arrives in time.
        """
        baseline = len(dialogue.session_utterances)
        deadline = time.monotonic() + ASK_TIMEOUT_SEC
        self._node.get_logger().info(
            f'[ASK] Waiting up to {ASK_TIMEOUT_SEC:.1f}s for user response'
        )
        while not goal_handle.is_cancel_requested:
            if time.monotonic() > deadline:
                self._node.get_logger().info(
                    '[ASK] Timeout waiting for user response'
                )
                result.result.error_msg = 'timeout waiting for response'
                return
            new = dialogue.session_utterances[baseline:]
            for utt in new:
                if utt.speaker_id != ROBOT_SPEAKER_ID:
                    result.answers = json.dumps({'input': utt.text})
                    self._node.get_logger().info(
                        f'[ASK] Captured raw answer: "{utt.text}"'
                    )
                    return
            time.sleep(0.05)

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

        # Check Say sub-skill availability
        if not self._say_client.is_available(timeout_sec=1.0):
            result.result.error_code = 134
            result.result.error_msg = 'Say sub-skill not available'
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
                    person_id=request.person_id,
                    group_id=request.group_id,
                )
            else:
                success = self._say_client.speak_and_wait(
                    request.input,
                    priority=priority,
                    cancel_event=cancel_event,
                    person_id=request.person_id,
                    group_id=request.group_id,
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

        # If addressed at a specific person/group, record the spoken text in
        # the corresponding dialogue history (or as a one-shot synthetic
        # __say__ dialogue when no chat is active for that interlocutor).
        if request.person_id or request.group_id:
            self._record_say_utterance(
                Interlocutor(
                    person_id=request.person_id,
                    group_id=request.group_id,
                ),
                request.input,
            )

        self._node.get_logger().info('[SAY] Completed successfully')
        goal_handle.succeed()
        return result

    def _record_say_utterance(
        self, interlocutor: Interlocutor, raw_input: str
    ) -> None:
        """Record a Say utterance against an interlocutor's history."""
        plain_text = (
            self._expression_executor.get_plain_text(raw_input)
            if self._expression_executor else raw_input
        )
        if not plain_text:
            return

        timestamp = self._now()
        active = self._dialogue_manager.get_dialogue_for_interlocutor(interlocutor)
        if active is not None:
            active.add_utterance(ROBOT_SPEAKER_ID, plain_text, timestamp)
            return

        synthetic = Dialogue(
            role=DialogueRole(name=SAY_ROLE_NAME),
            interlocutor=interlocutor,
            state=DialogueState.COMPLETED,
        )
        synthetic.add_utterance(ROBOT_SPEAKER_ID, plain_text, timestamp)
        synthetic.ended_at = timestamp
        members = (
            self._resolve_group_members(interlocutor.group_id)
            if interlocutor.is_group else None
        )
        self._conversations_store.archive(synthetic, group_members=members)
