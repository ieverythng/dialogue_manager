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

from dataclasses import dataclass, field
from enum import Enum
from uuid import UUID, uuid4

from chatbot_msgs.msg import DialogueRole


# Speaker ID used in `Utterance.speaker_id` to denote the robot itself.
ROBOT_SPEAKER_ID = '__myself__'

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
        """True if this interlocutor refers to a group of persons."""
        return bool(self.group_id)

    @property
    def is_bound(self) -> bool:
        """True if this interlocutor refers to a specific person or group."""
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
    summary: str | None = None  # Cached LLM summary; None ⇒ stale or not generated
    summary_generated_at: float | None = None
    chatbot_goal_id: UUID | None = None  # The chatbot action goal UUID
    goal_handle: object | None = None  # The skill action goal handle

    def __post_init__(self):
        """Validate priority range."""
        if not 0 <= self.priority <= 255:
            raise ValueError(f'Priority must be 0-255, got {self.priority}')

    def add_utterance(
        self, speaker_id: str, text: str, timestamp: float
    ) -> Utterance:
        """
        Append an utterance, lazily setting `started_at` on the first one.

        Any new utterance invalidates a cached summary (the dialogue's content
        has changed).
        """
        utt = Utterance(timestamp=timestamp, speaker_id=speaker_id, text=text)
        self.history.append(utt)
        if self.started_at is None:
            self.started_at = timestamp
        self.summary = None
        self.summary_generated_at = None
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

    def add_dialogue(self, dialogue: Dialogue) -> None:
        """Add a new dialogue to track."""
        self._dialogues[dialogue.dialogue_id] = dialogue

    def remove_dialogue(self, dialogue_id: UUID) -> Dialogue | None:
        """Remove and return a dialogue by ID."""
        return self._dialogues.pop(dialogue_id, None)

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

    def clear_expression_priority(self) -> None:
        """Clear the expression priority (no expression running)."""
        self._current_expression_priority = -1

    def clear_all(self) -> None:
        """Clear all active dialogues."""
        self._dialogues.clear()
