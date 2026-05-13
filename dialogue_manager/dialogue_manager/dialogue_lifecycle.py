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
Dialogue session endgame: finalize, archive, summarize.

These operations live outside `SkillServers` so the same code path is
used regardless of who decided the dialogue should end — a Chat/Ask
goal completing, a ROS4HRI group dispersing, or the node being
deactivated. Previously `manager_node` reached into
`SkillServers._finalize_and_archive` to drive this; that leaky private
is now a public method on `DialogueLifecycle`.
"""

import asyncio
from collections.abc import Awaitable, Callable
import threading

from rclpy.node import Node

from .conversations_history import ConversationsHistoryStore
from .dialogue import Dialogue, DialogueManager, DialogueState


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
    """Render the session's utterances as plain text. Trivial fallback."""
    lines = []
    for utt in dialogue.session_utterances:
        lines.append(f'{utt.speaker_id}: {utt.text}')
    return '\n'.join(lines)


class DialogueLifecycle:
    """
    Owns the "end of session" logic: finalize, archive, summarize.

    Decoupled from `SkillServers` so any caller (skill execution,
    group dispersal, lifecycle deactivation) can drive it through a
    public API instead of a private method.
    """

    def __init__(
        self,
        node: Node,
        dialogue_manager: DialogueManager,
        conversations_store: ConversationsHistoryStore,
        group_handler=None,  # GroupHandler | None — avoid circular import
        summarizer: Summarizer | None = None,
    ):
        """
        Initialize the lifecycle helper.

        Parameters
        ----------
        node
            Lifecycle node used for clock and logging.
        dialogue_manager
            Source of truth for active dialogues.
        conversations_store
            Where finished dialogues are archived.
        group_handler
            Used to resolve group members at finalize time so the
            archive can be fanned out into each member's history.
        summarizer
            Async callable producing the session summary. Defaults to
            `default_summarizer`, which renders raw utterances as text.

        """
        self._node = node
        self._dialogue_manager = dialogue_manager
        self._conversations_store = conversations_store
        self._group_handler = group_handler
        self._summarizer: Summarizer = summarizer or default_summarizer

    def _now(self) -> float:
        return self._node.get_clock().now().nanoseconds / 1e9

    def _resolve_group_members(self, group_id: str) -> list[str]:
        if self._group_handler is None:
            return []
        try:
            return list(self._group_handler.members_of(group_id) or [])
        except Exception as exc:
            self._node.get_logger().warn(
                f'[LIFECYCLE] group handler failed for "{group_id}": {exc}'
            )
            return []

    def finalize_and_archive(self, dialogue: Dialogue) -> None:
        """
        Mark `dialogue` completed, archive it, and remove from tracking.

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
                    f'[LIFECYCLE] Summarizer failed for dialogue '
                    f'{dialogue.dialogue_id}: {exc}'
                )
                return
            dialogue.summary = summary
            dialogue.summary_generated_at = self._now()
            # The archived dialogue lives in the conversations store; notify
            # observers so the snapshot reflects the freshly-landed summary.
            self._dialogue_manager.notify_change(dialogue.dialogue_id)
            self._node.get_logger().debug(
                f'[LIFECYCLE] Summary stored for dialogue {dialogue.dialogue_id}'
            )

        threading.Thread(target=runner, daemon=True).start()
