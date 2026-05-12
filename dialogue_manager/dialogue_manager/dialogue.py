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

"""Data structures for tracking active dialogues."""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from uuid import UUID, uuid4

from chatbot_msgs.msg import DialogueRole


# Speaker ID used in `Utterance.speaker_id` to denote the robot itself.
ROBOT_SPEAKER_ID = '__myself__'

# Speaker ID used to mark a pre-filled session summary (one entry carrying the
# summary text of a prior dialogue). Excluded from `session_utterances`.
SUMMARY_SPEAKER_ID = '__summary__'

# Speaker ID used as a session-break marker between pre-filled prior content and
# the current session's utterances. Excluded from `session_utterances`.
SESSION_BREAK_SPEAKER_ID = '__session_break__'

# Special role name used to wrap one-shot Say.action utterances recorded against
# a person/group when no chat dialogue is active for that interlocutor.
SAY_ROLE_NAME = '__say__'


class DialogueState(Enum):
    """State of a dialogue session."""

    PENDING = 'pending'  # Dialogue created but not yet started with chatbot
    ACTIVE = 'active'  # Dialogue is active and processing
    WAITING_RESPONSE = 'waiting_response'  # Waiting for chatbot response
    COMPLETED = 'completed'  # Dialogue has finished


@dataclass(frozen=True)
class Interlocutor:
    """
    A ROS4HRI person ID or group ID, mutually exclusive.

    An empty Interlocutor (both fields blank) denotes "not yet bound" and is
    used for the default chat that waits for the first speaker.
    """

    person_id: str = ''
    group_id: str = ''

    def __post_init__(self):
        """Validate mutual exclusivity."""
        if self.person_id and self.group_id:
            raise ValueError(
                'Interlocutor cannot have both person_id and group_id set'
            )

    @property
    def is_group(self) -> bool:
        """Return True if this interlocutor refers to a group of persons."""
        return bool(self.group_id)

    @property
    def is_bound(self) -> bool:
        """Return True if this interlocutor refers to a specific person or group."""
        return bool(self.person_id) or bool(self.group_id)

    @property
    def key(self) -> str:
        """Stable index key — 'person:<id>', 'group:<id>' or 'anonymous'."""
        if self.person_id:
            return f'person:{self.person_id}'
        if self.group_id:
            return f'group:{self.group_id}'
        return 'anonymous'


@dataclass(frozen=True)
class Utterance:
    """One entry in a dialogue history: who said what, when."""

    timestamp: float  # Epoch seconds
    speaker_id: str  # ROBOT_SPEAKER_ID or a person_id
    text: str


@dataclass
class Dialogue:
    """
    A conversation between the robot and a person/group.

    The (dialogue_id, role, interlocutor) triple identifies the dialogue and
    is conceptually fixed. Priority, state, timestamps and history change
    over the dialogue's lifetime.

    `started_at` is set lazily on the first utterance; an empty dialogue
    therefore has `started_at is None` and is not considered to have started.
    """

    role: DialogueRole
    interlocutor: Interlocutor = field(default_factory=Interlocutor)
    priority: int = 128
    state: DialogueState = DialogueState.PENDING
    dialogue_id: UUID = field(default_factory=uuid4)
    started_at: float | None = None  # Set on first utterance
    ended_at: float | None = None    # Set when state -> COMPLETED
    history: list[Utterance] = field(default_factory=list)
    # Index into `history` at which the *current* session's utterances start.
    # Anything before this index is pre-filled context (summaries + session-
    # break markers) and is excluded from `session_utterances`, archival, and
    # persistence so it cannot bleed into the next session.
    session_start_index: int = 0
    summary: str | None = None  # Cached LLM summary; None ⇒ not yet generated
    summary_generated_at: float | None = None
    chatbot_goal_id: UUID | None = None  # The chatbot action goal UUID
    goal_handle: object | None = None  # The skill action goal handle
    # Hook fired after any state-changing mutation on this dialogue
    # (add_utterance for now). Set by DialogueManager.add_dialogue so the
    # debug-state publisher can react to changes without invasive plumbing.
    _change_callback: Callable[[], None] | None = field(
        default=None, repr=False, compare=False,
    )

    def __post_init__(self):
        """Validate priority range."""
        if not 0 <= self.priority <= 255:
            raise ValueError(f'Priority must be 0-255, got {self.priority}')

    @property
    def session_utterances(self) -> list[Utterance]:
        """Return only this session's utterances (post-preload, real content)."""
        return self.history[self.session_start_index:]

    def add_utterance(
        self, speaker_id: str, text: str, timestamp: float
    ) -> Utterance:
        """
        Append an utterance, lazily setting `started_at` on the first one.

        Any new utterance invalidates a cached summary (the dialogue's content
        has changed). The `_change_callback` (if set) is fired so downstream
        consumers (e.g. the debug-state publisher) can react.
        """
        utt = Utterance(timestamp=timestamp, speaker_id=speaker_id, text=text)
        self.history.append(utt)
        if self.started_at is None:
            self.started_at = timestamp
        self.summary = None
        self.summary_generated_at = None
        if self._change_callback is not None:
            try:
                self._change_callback()
            except Exception:
                pass  # never let a hook break utterance recording
        return utt


class DialogueManager:
    """
    Manages multiple active dialogues.

    Handles priority-based dialogue selection and tracking. Persistence and
    long-term per-person/group history live in `ConversationsHistoryStore`.
    """

    def __init__(self):
        """Initialize the dialogue manager."""
        self._dialogues: dict[UUID, Dialogue] = {}
        self._current_expression_priority: int = -1
        # The catch-all 'default' dialogue: anything spoken to the robot when
        # no other dialogue is active for that interlocutor falls into here.
        # May be backed by the chatbot or run chatbot-less (in which case the
        # dialogue is purely a history container driven by the SpeechHandler).
        self._default_dialogue_id: UUID | None = None
        # Optional observer hook invoked whenever the manager's state changes.
        # `dialogue_id` is the UUID string of the affected dialogue, or None
        # for changes that don't pertain to one specific dialogue
        # (e.g. expression priority, clear_all).
        self._on_change: Callable[[str | None], None] | None = None

    @property
    def active_dialogues(self) -> dict[UUID, Dialogue]:
        """Return all tracked dialogues."""
        return self._dialogues

    @property
    def current_max_priority(self) -> int:
        """Return the maximum priority of all active dialogues and expressions."""
        dialogue_priorities = [
            d.priority for d in self._dialogues.values()
            if d.state in (DialogueState.ACTIVE, DialogueState.WAITING_RESPONSE)
        ]
        if dialogue_priorities:
            return max(max(dialogue_priorities), self._current_expression_priority)
        return self._current_expression_priority

    def can_accept_priority(self, priority: int) -> bool:
        """Check if a new goal with given priority can be accepted."""
        return priority > self.current_max_priority

    @property
    def default_dialogue_id(self) -> UUID | None:
        """Return the current default-dialogue UUID, if any."""
        return self._default_dialogue_id

    def set_default_dialogue_id(self, dialogue_id: UUID | None) -> None:
        """Mark a dialogue as the default (catch-all) for unbound speech."""
        self._default_dialogue_id = dialogue_id
        self.notify_change(dialogue_id)

    def set_change_callback(
        self, callback: Callable[[str | None], None] | None
    ) -> None:
        """Register an observer fired when manager state changes.

        Existing tracked dialogues have their `_change_callback` wired in
        immediately so utterance additions also propagate.
        """
        self._on_change = callback
        for d in self._dialogues.values():
            d._change_callback = self._make_dialogue_hook(d.dialogue_id)

    def notify_change(self, dialogue_id: UUID | None = None) -> None:
        """Manually notify observers of a state change (state, summary, ...)."""
        if self._on_change is None:
            return
        try:
            self._on_change(str(dialogue_id) if dialogue_id else None)
        except Exception:
            pass  # never let a debug-only hook break the main flow

    def _make_dialogue_hook(self, dialogue_id: UUID) -> Callable[[], None]:
        did = str(dialogue_id)

        def _hook() -> None:
            if self._on_change is not None:
                try:
                    self._on_change(did)
                except Exception:
                    pass

        return _hook

    def add_dialogue(self, dialogue: Dialogue) -> None:
        """Add a new dialogue to track."""
        self._dialogues[dialogue.dialogue_id] = dialogue
        dialogue._change_callback = self._make_dialogue_hook(dialogue.dialogue_id)
        self.notify_change(dialogue.dialogue_id)

    def remove_dialogue(self, dialogue_id: UUID) -> Dialogue | None:
        """Remove and return a dialogue by ID."""
        removed = self._dialogues.pop(dialogue_id, None)
        if removed is not None:
            removed._change_callback = None
            if self._default_dialogue_id == dialogue_id:
                self._default_dialogue_id = None
            self.notify_change(dialogue_id)
        return removed

    def get_dialogue(self, dialogue_id: UUID) -> Dialogue | None:
        """Get a dialogue by ID."""
        return self._dialogues.get(dialogue_id)

    def get_dialogue_for_interlocutor(
        self, interlocutor: Interlocutor
    ) -> Dialogue | None:
        """Find an active dialogue for a person or group interlocutor."""
        for dialogue in self._dialogues.values():
            if (
                dialogue.interlocutor == interlocutor
                and dialogue.state == DialogueState.ACTIVE
            ):
                return dialogue
        return None

    def set_expression_priority(self, priority: int) -> None:
        """Set the priority of the currently executing expression."""
        self._current_expression_priority = priority
        self.notify_change()

    def clear_expression_priority(self) -> None:
        """Clear the expression priority (no expression running)."""
        self._current_expression_priority = -1
        self.notify_change()

    def clear_all(self) -> None:
        """Clear all active dialogues."""
        for d in self._dialogues.values():
            d._change_callback = None
        self._dialogues.clear()
        self._default_dialogue_id = None
        self.notify_change()
