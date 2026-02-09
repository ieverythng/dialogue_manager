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
from typing import Optional
from uuid import UUID, uuid4

from chatbot_msgs.msg import DialogueRole


class DialogueState(Enum):
    """State of a dialogue session."""

    PENDING = 'pending'  # Dialogue created but not yet started with chatbot
    ACTIVE = 'active'  # Dialogue is active and processing
    WAITING_RESPONSE = 'waiting_response'  # Waiting for chatbot response
    COMPLETED = 'completed'  # Dialogue has finished


@dataclass
class Dialogue:
    """
    Represents an active dialogue session.

    Tracks the state and metadata of a dialogue between the robot and user(s).
    """

    role: DialogueRole
    person_id: str = ''
    group_id: str = ''
    priority: int = 128
    state: DialogueState = DialogueState.PENDING
    dialogue_id: UUID = field(default_factory=uuid4)  # Internal tracking ID
    chatbot_goal_id: Optional[UUID] = None  # The chatbot action goal UUID
    goal_handle: Optional[object] = None  # The skill action goal handle

    def __post_init__(self):
        """Validate priority range."""
        if not 0 <= self.priority <= 255:
            raise ValueError(f'Priority must be 0-255, got {self.priority}')


class DialogueManager:
    """
    Manages multiple active dialogues.

    Handles priority-based dialogue selection and tracking.
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

    def remove_dialogue(self, dialogue_id: UUID) -> Optional[Dialogue]:
        """Remove and return a dialogue by ID."""
        return self._dialogues.pop(dialogue_id, None)

    def get_dialogue(self, dialogue_id: UUID) -> Optional[Dialogue]:
        """Get a dialogue by ID."""
        return self._dialogues.get(dialogue_id)

    def get_dialogue_for_person(self, person_id: str) -> Optional[Dialogue]:
        """Find an active dialogue for a specific person."""
        for dialogue in self._dialogues.values():
            if dialogue.person_id == person_id and dialogue.state == DialogueState.ACTIVE:
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
