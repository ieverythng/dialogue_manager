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

"""Unit tests for dialogue.py data structures."""

from uuid import UUID, uuid4

from chatbot_msgs.msg import DialogueRole
from dialogue_manager.dialogue import (
    Dialogue,
    DialogueManager,
    DialogueState,
)
import pytest


class TestDialogueState:
    """Tests for DialogueState enum."""

    def test_state_values(self):
        """Verify all state values are correct."""
        assert DialogueState.PENDING.value == 'pending'
        assert DialogueState.ACTIVE.value == 'active'
        assert DialogueState.WAITING_RESPONSE.value == 'waiting_response'
        assert DialogueState.COMPLETED.value == 'completed'

    def test_state_count(self):
        """Verify we have exactly 4 states."""
        assert len(DialogueState) == 4


class TestDialogue:
    """Tests for the Dialogue dataclass."""

    def test_create_minimal_dialogue(self):
        """Create dialogue with only required field."""
        role = DialogueRole(name='test_role')
        dialogue = Dialogue(role=role)

        assert dialogue.role == role
        assert dialogue.person_id == ''
        assert dialogue.group_id == ''
        assert dialogue.priority == 128
        assert dialogue.state == DialogueState.PENDING
        assert isinstance(dialogue.dialogue_id, UUID)
        assert dialogue.chatbot_goal_id is None
        assert dialogue.goal_handle is None

    def test_create_full_dialogue(self):
        """Create dialogue with all fields specified."""
        role = DialogueRole(name='full_role')
        dialogue_id = uuid4()
        chatbot_id = uuid4()

        dialogue = Dialogue(
            role=role,
            person_id='person_123',
            group_id='group_456',
            priority=200,
            state=DialogueState.ACTIVE,
            dialogue_id=dialogue_id,
            chatbot_goal_id=chatbot_id,
            goal_handle='mock_handle'
        )

        assert dialogue.role == role
        assert dialogue.person_id == 'person_123'
        assert dialogue.group_id == 'group_456'
        assert dialogue.priority == 200
        assert dialogue.state == DialogueState.ACTIVE
        assert dialogue.dialogue_id == dialogue_id
        assert dialogue.chatbot_goal_id == chatbot_id
        assert dialogue.goal_handle == 'mock_handle'

    def test_priority_valid_min(self):
        """Priority at minimum valid value (0) is accepted."""
        role = DialogueRole(name='test')
        dialogue = Dialogue(role=role, priority=0)
        assert dialogue.priority == 0

    def test_priority_valid_max(self):
        """Priority at maximum valid value (255) is accepted."""
        role = DialogueRole(name='test')
        dialogue = Dialogue(role=role, priority=255)
        assert dialogue.priority == 255

    def test_priority_invalid_negative(self):
        """Priority below 0 raises ValueError."""
        role = DialogueRole(name='test')
        with pytest.raises(ValueError, match='Priority must be 0-255'):
            Dialogue(role=role, priority=-1)

    def test_priority_invalid_too_high(self):
        """Priority above 255 raises ValueError."""
        role = DialogueRole(name='test')
        with pytest.raises(ValueError, match='Priority must be 0-255'):
            Dialogue(role=role, priority=256)

    def test_unique_dialogue_ids(self):
        """Each dialogue gets a unique ID by default."""
        role = DialogueRole(name='test')
        d1 = Dialogue(role=role)
        d2 = Dialogue(role=role)
        assert d1.dialogue_id != d2.dialogue_id


class TestDialogueManager:
    """Tests for the DialogueManager class."""

    def test_init_empty(self):
        """Manager starts with no dialogues."""
        manager = DialogueManager()
        assert len(manager.active_dialogues) == 0
        assert manager.current_max_priority == -1

    def test_add_dialogue(self):
        """Add a dialogue to the manager."""
        manager = DialogueManager()
        role = DialogueRole(name='test')
        dialogue = Dialogue(role=role)

        manager.add_dialogue(dialogue)

        assert len(manager.active_dialogues) == 1
        assert dialogue.dialogue_id in manager.active_dialogues

    def test_get_dialogue_existing(self):
        """Get an existing dialogue by ID."""
        manager = DialogueManager()
        role = DialogueRole(name='test')
        dialogue = Dialogue(role=role)
        manager.add_dialogue(dialogue)

        result = manager.get_dialogue(dialogue.dialogue_id)

        assert result is dialogue

    def test_get_dialogue_nonexistent(self):
        """Get a nonexistent dialogue returns None."""
        manager = DialogueManager()
        result = manager.get_dialogue(uuid4())
        assert result is None

    def test_remove_dialogue(self):
        """Remove a dialogue from the manager."""
        manager = DialogueManager()
        role = DialogueRole(name='test')
        dialogue = Dialogue(role=role)
        manager.add_dialogue(dialogue)

        removed = manager.remove_dialogue(dialogue.dialogue_id)

        assert removed is dialogue
        assert len(manager.active_dialogues) == 0

    def test_remove_dialogue_nonexistent(self):
        """Remove nonexistent dialogue returns None."""
        manager = DialogueManager()
        result = manager.remove_dialogue(uuid4())
        assert result is None

    def test_current_max_priority_with_active_dialogue(self):
        """Max priority considers ACTIVE dialogues."""
        manager = DialogueManager()
        role = DialogueRole(name='test')
        dialogue = Dialogue(role=role, priority=150, state=DialogueState.ACTIVE)
        manager.add_dialogue(dialogue)

        assert manager.current_max_priority == 150

    def test_current_max_priority_with_waiting_dialogue(self):
        """Max priority considers WAITING_RESPONSE dialogues."""
        manager = DialogueManager()
        role = DialogueRole(name='test')
        dialogue = Dialogue(
            role=role, priority=175, state=DialogueState.WAITING_RESPONSE
        )
        manager.add_dialogue(dialogue)

        assert manager.current_max_priority == 175

    def test_current_max_priority_ignores_pending(self):
        """Max priority ignores PENDING dialogues."""
        manager = DialogueManager()
        role = DialogueRole(name='test')
        dialogue = Dialogue(role=role, priority=200, state=DialogueState.PENDING)
        manager.add_dialogue(dialogue)

        assert manager.current_max_priority == -1

    def test_current_max_priority_ignores_completed(self):
        """Max priority ignores COMPLETED dialogues."""
        manager = DialogueManager()
        role = DialogueRole(name='test')
        dialogue = Dialogue(role=role, priority=200, state=DialogueState.COMPLETED)
        manager.add_dialogue(dialogue)

        assert manager.current_max_priority == -1

    def test_current_max_priority_multiple_dialogues(self):
        """Max priority returns highest among multiple active dialogues."""
        manager = DialogueManager()
        role = DialogueRole(name='test')

        d1 = Dialogue(role=role, priority=100, state=DialogueState.ACTIVE)
        d2 = Dialogue(role=role, priority=200, state=DialogueState.ACTIVE)
        d3 = Dialogue(role=role, priority=150, state=DialogueState.WAITING_RESPONSE)

        manager.add_dialogue(d1)
        manager.add_dialogue(d2)
        manager.add_dialogue(d3)

        assert manager.current_max_priority == 200

    def test_expression_priority_affects_max(self):
        """Expression priority is considered in max priority."""
        manager = DialogueManager()
        manager.set_expression_priority(175)
        assert manager.current_max_priority == 175

    def test_expression_priority_combined_with_dialogue(self):
        """Expression and dialogue priorities are combined."""
        manager = DialogueManager()
        role = DialogueRole(name='test')
        dialogue = Dialogue(role=role, priority=100, state=DialogueState.ACTIVE)
        manager.add_dialogue(dialogue)
        manager.set_expression_priority(150)

        assert manager.current_max_priority == 150

    def test_clear_expression_priority(self):
        """Clearing expression priority resets to -1."""
        manager = DialogueManager()
        manager.set_expression_priority(100)
        manager.clear_expression_priority()
        assert manager.current_max_priority == -1

    def test_can_accept_priority_higher(self):
        """Higher priority than current max is accepted."""
        manager = DialogueManager()
        role = DialogueRole(name='test')
        dialogue = Dialogue(role=role, priority=100, state=DialogueState.ACTIVE)
        manager.add_dialogue(dialogue)

        assert manager.can_accept_priority(150) is True

    def test_can_accept_priority_equal(self):
        """Equal priority to current max is rejected."""
        manager = DialogueManager()
        role = DialogueRole(name='test')
        dialogue = Dialogue(role=role, priority=100, state=DialogueState.ACTIVE)
        manager.add_dialogue(dialogue)

        assert manager.can_accept_priority(100) is False

    def test_can_accept_priority_lower(self):
        """Lower priority than current max is rejected."""
        manager = DialogueManager()
        role = DialogueRole(name='test')
        dialogue = Dialogue(role=role, priority=100, state=DialogueState.ACTIVE)
        manager.add_dialogue(dialogue)

        assert manager.can_accept_priority(50) is False

    def test_get_dialogue_for_person_found(self):
        """Find active dialogue for a person."""
        manager = DialogueManager()
        role = DialogueRole(name='test')
        dialogue = Dialogue(
            role=role, person_id='person_123', state=DialogueState.ACTIVE
        )
        manager.add_dialogue(dialogue)

        result = manager.get_dialogue_for_person('person_123')

        assert result is dialogue

    def test_get_dialogue_for_person_not_found(self):
        """Return None when no dialogue exists for person."""
        manager = DialogueManager()
        result = manager.get_dialogue_for_person('unknown_person')
        assert result is None

    def test_get_dialogue_for_person_inactive(self):
        """Return None when dialogue exists but is not ACTIVE."""
        manager = DialogueManager()
        role = DialogueRole(name='test')
        dialogue = Dialogue(
            role=role, person_id='person_123', state=DialogueState.PENDING
        )
        manager.add_dialogue(dialogue)

        result = manager.get_dialogue_for_person('person_123')

        assert result is None

    def test_get_dialogue_for_person_multiple(self):
        """Return the active dialogue when multiple exist for person."""
        manager = DialogueManager()
        role = DialogueRole(name='test')

        d1 = Dialogue(
            role=role, person_id='person_123', state=DialogueState.COMPLETED
        )
        d2 = Dialogue(
            role=role, person_id='person_123', state=DialogueState.ACTIVE
        )

        manager.add_dialogue(d1)
        manager.add_dialogue(d2)

        result = manager.get_dialogue_for_person('person_123')

        assert result is d2
