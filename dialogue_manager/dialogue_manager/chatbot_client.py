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

"""
Chatbot client for the Dialogue Manager.

Speaks the stateless `chatbot_msgs` contract (v4): every chatbot turn
is a single `DialogueInteraction` service call carrying the full
dialogue history plus an optional prior-session summary. The chatbot
backend retains no per-dialogue state between calls. An optional
`PrepareDialogue` service is invoked fire-and-forget when a dialogue
is created, giving backends a chance to warm up role-specific
resources.
"""

from uuid import UUID

from chatbot_msgs.msg import Utterance
from chatbot_msgs.srv import DialogueInteraction, PrepareDialogue
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.logging import LoggingSeverity
from rclpy.node import Node
from rclpy.publisher import Publisher
from std_msgs.msg import Bool
from unique_identifier_msgs.msg import UUID as UUIDMsg

from .conversations_history import ConversationsHistoryStore
from .dialogue import (
    Dialogue,
    DialogueManager,
    DialogueState,
    ROBOT_SPEAKER_ID,
    SESSION_BREAK_SPEAKER_ID,
    SUMMARY_SPEAKER_ID,
    SYSTEM_SPEAKER_ID,
)
from .markup.executor import ExpressionExecutor
from .say_client import SayClient


def uuid_to_msg(uuid: UUID) -> UUIDMsg:
    """Convert a Python UUID to a ROS UUID message."""
    return UUIDMsg(uuid=list(uuid.bytes))


def _debug_enabled(logger) -> bool:
    """
    Return True iff `logger`'s effective level is DEBUG or lower.

    Wraps the comparison so unit tests using a MagicMock logger
    don't trip a TypeError when MagicMock vs. LoggingSeverity is
    compared.
    """
    try:
        return logger.get_effective_level() <= LoggingSeverity.DEBUG
    except TypeError:
        return False


def _to_utterance_msg(speaker_id: str, text: str, timestamp: float) -> Utterance:
    """Map a `dialogue_manager` Utterance into a `chatbot_msgs/Utterance`."""
    if speaker_id == ROBOT_SPEAKER_ID:
        speaker = Utterance.ASSISTANT
    elif speaker_id == SYSTEM_SPEAKER_ID:
        speaker = Utterance.SYSTEM
    else:
        speaker = speaker_id
    return Utterance(speaker=speaker, text=text, timestamp=timestamp)


class ChatbotClient:
    """
    Handles chatbot interactions over the stateless v4 contract.

    Per dialogue:
      - `prepare(dialogue)` is fire-and-forget (called when the
        dialogue is created); the chatbot may use it to warm up
        role-specific resources but is not required to.
      - `interact(dialogue)` is called once per user-driven turn (or
        any time we want the chatbot to produce a turn). It snapshots
        the dialogue's current session history, ships it as a single
        `DialogueInteraction` request, and on response: records the
        robot's utterance in the dialogue history, publishes any
        detected intents, speaks the response, and propagates a
        chatbot-signalled terminal flag back to the dialogue.

    The client is otherwise stateless: no goal handles, no per-dialogue
    bookkeeping. The chatbot backend is also stateless: there is no
    "attach" / "detach" lifecycle to coordinate.
    """

    def __init__(
        self,
        node: Node,
        dialogue_manager: DialogueManager,
        say_client: SayClient,
        conversations_store: ConversationsHistoryStore | None = None,
        expression_executor: ExpressionExecutor | None = None,
        intents_pub: Publisher | None = None,
        waiting_chatbot_pub: Publisher | None = None,
        callback_group: ReentrantCallbackGroup | None = None,
    ):
        """Initialize the chatbot client."""
        self._node = node
        self._dialogue_manager = dialogue_manager
        self._say_client = say_client
        self._conversations_store = conversations_store
        self._expression_executor = expression_executor
        self._intents_pub = intents_pub
        self._waiting_chatbot_pub = waiting_chatbot_pub
        self._callback_group = callback_group

        self._prepare_client = None
        self._interaction_client = None
        self._waiting_for_response = False

    def _now(self) -> float:
        """Return the current time in epoch seconds, from the node clock."""
        return self._node.get_clock().now().nanoseconds / 1e9

    @property
    def waiting_for_response(self) -> bool:
        """Return True if a dialogue_interaction call is in flight."""
        return self._waiting_for_response

    def create_clients(self, chatbot_prefix: str) -> None:
        """Create the prepare_dialogue + dialogue_interaction service clients."""
        prepare_srv = f'{chatbot_prefix}/prepare_dialogue'
        interaction_srv = f'{chatbot_prefix}/dialogue_interaction'

        self._prepare_client = self._node.create_client(
            PrepareDialogue, prepare_srv, callback_group=self._callback_group,
        )
        self._node.get_logger().info(
            f'[CHATBOT] Created service client: {prepare_srv}'
        )

        self._interaction_client = self._node.create_client(
            DialogueInteraction,
            interaction_srv,
            callback_group=self._callback_group,
        )
        self._node.get_logger().info(
            f'[CHATBOT] Created service client: {interaction_srv}'
        )

    def destroy(self) -> None:
        """Destroy the chatbot service clients."""
        if self._prepare_client:
            self._node.destroy_client(self._prepare_client)
            self._prepare_client = None
        if self._interaction_client:
            self._node.destroy_client(self._interaction_client)
            self._interaction_client = None

    def is_available(self, timeout_sec: float = 1.0) -> bool:
        """Check if the chatbot backend is reachable."""
        if not self._interaction_client:
            return False
        return self._interaction_client.wait_for_service(timeout_sec=timeout_sec)

    # ------------------------------------------------------------------
    # PrepareDialogue (optional warm-up).
    # ------------------------------------------------------------------

    def prepare(self, dialogue: Dialogue) -> bool:
        """
        Fire-and-forget warm-up for `dialogue`.

        Calls the backend's optional `prepare_dialogue` service so it
        can initialise role-specific resources before the first turn.
        We do not wait for the response — backends that have nothing
        to warm up may treat the service as a no-op, and a backend
        outage here must not prevent us from later attempting an
        interaction (the interaction itself is the load-bearing call).
        """
        if not self._prepare_client:
            return False
        # service_is_ready avoids a blocking wait in case the backend
        # isn't yet up; an unavailable prepare is non-fatal.
        if not self._prepare_client.service_is_ready():
            self._node.get_logger().debug(
                f'[CHATBOT] prepare_dialogue not ready for dialogue '
                f'{dialogue.dialogue_id}; skipping warm-up'
            )
            return False

        request = PrepareDialogue.Request()
        request.dialogue_id = uuid_to_msg(dialogue.dialogue_id)
        request.role = dialogue.role
        self._node.get_logger().info(
            f'[CHATBOT] prepare_dialogue: id={dialogue.dialogue_id} '
            f'role={dialogue.role.name!r}'
        )
        self._prepare_client.call_async(request)
        return True

    # ------------------------------------------------------------------
    # DialogueInteraction (main turn driver).
    # ------------------------------------------------------------------

    def interact(self, dialogue: Dialogue) -> bool:
        """
        Request a chatbot turn for `dialogue` based on its current history.

        Snapshots `dialogue.session_utterances` into a
        `chatbot_msgs/Utterance[]`, builds the prior-session summary
        (from the conversations store, when configured), packs it into
        a `DialogueInteraction` request, and calls the service async.
        The response is handled on the executor thread: see
        `_on_response` for the post-call pipeline (record robot speech,
        publish intents, speak, propagate terminal flag).

        Callers are expected to have already appended the triggering
        turn (e.g. a user utterance, or a `__system__` directive) to
        `dialogue.history` before invoking this method; the chatbot
        must see what it's being asked to react to.
        """
        if not self._interaction_client:
            self._node.get_logger().warn(
                '[CHATBOT] No interaction client available'
            )
            return False

        if not dialogue.session_utterances:
            self._node.get_logger().warn(
                f'[CHATBOT] interact() called with empty session history '
                f'for dialogue {dialogue.dialogue_id}; skipping'
            )
            return False

        history = []
        for utt in dialogue.session_utterances:
            # The pre-fill sentinels (SUMMARY / SESSION_BREAK) live
            # before `session_start_index`, so they should never appear
            # here — guard anyway in case the index gets out of sync.
            if utt.speaker_id in (SUMMARY_SPEAKER_ID, SESSION_BREAK_SPEAKER_ID):
                continue
            history.append(_to_utterance_msg(
                utt.speaker_id, utt.text, utt.timestamp,
            ))

        summary = ''
        if (self._conversations_store is not None
                and dialogue.interlocutor.is_bound):
            summary = self._conversations_store.build_context(
                dialogue.interlocutor,
                now=self._now(),
            )

        request = DialogueInteraction.Request()
        request.dialogue_id = uuid_to_msg(dialogue.dialogue_id)
        request.role = dialogue.role
        request.summary = summary
        request.history = history

        # Mark waiting state for diagnostics + the speech handler's
        # "drop while busy" guard.
        self._waiting_for_response = True
        if self._waiting_chatbot_pub:
            self._waiting_chatbot_pub.publish(Bool(data=True))
        dialogue.state = DialogueState.WAITING_RESPONSE
        self._dialogue_manager.notify_change(dialogue.dialogue_id)

        if len(history) > 0:
            last = history[-1]
            self._node.get_logger().info(
                f'[CHATBOT REQUEST] dialogue_id={dialogue.dialogue_id} '
                f'role={dialogue.role.name!r} history_len={len(history)} '
                f'last={last.speaker!r}:"{last.text[:80]}"'
            )

        # Verbose dump of the exact history about to ship, gated at DEBUG
        # so it doesn't bloat normal runs. The `if` short-circuits the
        # f-string formatting when DEBUG is off. Enable with e.g.
        # `--ros-args --log-level dialogue_manager:=debug`.
        logger = self._node.get_logger()
        if _debug_enabled(logger):
            history_dump = '\n'.join(
                f'  [{u.speaker}] {u.text!r}' for u in history
            )
            logger.debug(
                f'[CHATBOT REQUEST] outbound history '
                f'({len(history)} entries):\n{history_dump}'
            )
            if summary:
                logger.debug(
                    f'[CHATBOT REQUEST] outbound summary '
                    f'({len(summary)} chars):\n{summary}'
                )

        future = self._interaction_client.call_async(request)
        future.add_done_callback(
            lambda f, did=dialogue.dialogue_id: self._on_response(f, did)
        )
        return True

    def _on_response(self, future, dialogue_id: UUID) -> None:
        """Handle the `DialogueInteraction` service response."""
        self._waiting_for_response = False
        if self._waiting_chatbot_pub:
            self._waiting_chatbot_pub.publish(Bool(data=False))

        dialogue = self._dialogue_manager.get_dialogue(dialogue_id)
        if dialogue and dialogue.state == DialogueState.WAITING_RESPONSE:
            dialogue.state = DialogueState.ACTIVE
            self._dialogue_manager.notify_change(dialogue.dialogue_id)

        try:
            response = future.result()
        except Exception as exc:
            self._node.get_logger().error(
                f'[CHATBOT RESPONSE] failed for {dialogue_id}: {exc}'
            )
            return

        if response.error_msg:
            self._node.get_logger().warn(
                f'[CHATBOT RESPONSE] error for {dialogue_id}: '
                f'{response.error_msg}'
            )
            return

        if len(response.response) > 100:
            log_text = f'"{response.response[:100]}..."'
        else:
            log_text = f'"{response.response}"'
        self._node.get_logger().info(
            f'[CHATBOT RESPONSE] dialogue_id={dialogue_id}: {log_text} '
            f'terminal={response.dialogue_terminal}'
        )

        # Record the robot's utterance in the dialogue history. The
        # backend is stateless wrt. history; the next turn will carry
        # this back to it via `interact()`.
        if dialogue and response.response:
            dialogue.add_utterance(
                ROBOT_SPEAKER_ID, response.response, self._now(),
            )

        # Publish intents.
        if response.intents and self._intents_pub:
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

        # Speak the response (with markup processing if available).
        if response.response and dialogue:
            person_id = (
                dialogue.interlocutor.person_id
                if dialogue.interlocutor else ''
            )
            group_id = (
                dialogue.interlocutor.group_id
                if dialogue.interlocutor else ''
            )
            priority = dialogue.priority
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

        # Role-driven termination signal: stash the structured results
        # and flip the dialogue to COMPLETED so the skill server's
        # execute coroutine can pick them up.
        if response.dialogue_terminal and dialogue:
            self._node.get_logger().info(
                f'[CHATBOT RESPONSE] dialogue_terminal=True for {dialogue_id}; '
                f'results="{response.results[:120]}"'
            )
            dialogue.results = response.results
            dialogue.state = DialogueState.COMPLETED
            self._dialogue_manager.notify_change(dialogue.dialogue_id)
