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
    Interlocutor,
    ROBOT_SPEAKER_ID,
    SESSION_BREAK_SPEAKER_ID,
    SUMMARY_SPEAKER_ID,
    Utterance,
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


class TestInterlocutor:
    """Tests for the Interlocutor union type."""

    def test_empty_interlocutor_unbound(self):
        """An empty Interlocutor is anonymous and unbound."""
        i = Interlocutor()
        assert not i.is_bound
        assert not i.is_group
        assert i.key == 'anonymous'

    def test_person_interlocutor(self):
        """A person interlocutor is bound and not a group."""
        i = Interlocutor(person_id='alice')
        assert i.is_bound
        assert not i.is_group
        assert i.key == 'person:alice'

    def test_group_interlocutor(self):
        """A group interlocutor is bound and reports as group."""
        i = Interlocutor(group_id='visitors')
        assert i.is_bound
        assert i.is_group
        assert i.key == 'group:visitors'

    def test_both_set_raises(self):
        """Setting both person_id and group_id is rejected."""
        with pytest.raises(ValueError, match='cannot have both'):
            Interlocutor(person_id='alice', group_id='visitors')

    def test_equality(self):
        """Interlocutors with the same fields compare equal and hash equal."""
        a = Interlocutor(person_id='alice')
        b = Interlocutor(person_id='alice')
        assert a == b
        assert hash(a) == hash(b)


class TestUtterance:
    """Tests for the Utterance dataclass."""

    def test_create_utterance(self):
        """Utterances carry timestamp, speaker, text."""
        u = Utterance(timestamp=1.5, speaker_id='alice', text='hi')
        assert u.timestamp == 1.5
        assert u.speaker_id == 'alice'
        assert u.text == 'hi'

    def test_utterance_frozen(self):
        """Utterances are immutable."""
        u = Utterance(timestamp=1.0, speaker_id='alice', text='hi')
        with pytest.raises(Exception):
            u.text = 'bye'  # type: ignore[misc]


class TestDialogue:
    """Tests for the Dialogue dataclass."""

    def test_create_minimal_dialogue(self):
        """Create dialogue with only required field."""
        role = DialogueRole(name='test_role')
        dialogue = Dialogue(role=role)

        assert dialogue.role == role
        assert dialogue.interlocutor == Interlocutor()
        assert dialogue.priority == 128
        assert dialogue.state == DialogueState.PENDING
        assert isinstance(dialogue.dialogue_id, UUID)
        assert dialogue.started_at is None
        assert dialogue.ended_at is None
        assert dialogue.history == []
        assert dialogue.summary is None
        assert dialogue.chatbot_goal_id is None
        assert dialogue.goal_handle is None

    def test_create_full_dialogue(self):
        """Create dialogue with all fields specified."""
        role = DialogueRole(name='full_role')
        dialogue_id = uuid4()
        chatbot_id = uuid4()
        interlocutor = Interlocutor(person_id='person_123')

        dialogue = Dialogue(
            role=role,
            interlocutor=interlocutor,
            priority=200,
            state=DialogueState.ACTIVE,
            dialogue_id=dialogue_id,
            chatbot_goal_id=chatbot_id,
            goal_handle='mock_handle'
        )

        assert dialogue.role == role
        assert dialogue.interlocutor == interlocutor
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

    def test_add_utterance_appends(self):
        """Utterances accumulate in history order."""
        d = Dialogue(role=DialogueRole(name='test'))
        d.add_utterance('alice', 'hello', timestamp=1.0)
        d.add_utterance(ROBOT_SPEAKER_ID, 'hi alice', timestamp=2.0)
        assert len(d.history) == 2
        assert d.history[0].speaker_id == 'alice'
        assert d.history[1].speaker_id == ROBOT_SPEAKER_ID

    def test_add_utterance_lazy_started_at(self):
        """`started_at` is set on the first utterance only."""
        d = Dialogue(role=DialogueRole(name='test'))
        assert d.started_at is None
        d.add_utterance('alice', 'one', timestamp=10.0)
        assert d.started_at == 10.0
        d.add_utterance('alice', 'two', timestamp=20.0)
        assert d.started_at == 10.0  # unchanged

    def test_add_utterance_invalidates_summary(self):
        """A new utterance discards the cached summary."""
        d = Dialogue(role=DialogueRole(name='test'))
        d.summary = 'previously summarized'
        d.summary_generated_at = 1.0
        d.add_utterance('alice', 'new turn', timestamp=42.0)
        assert d.summary is None
        assert d.summary_generated_at is None


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

    def test_can_start_dialogue_higher(self):
        """A new dialogue at strictly higher priority is accepted."""
        manager = DialogueManager()
        role = DialogueRole(name='test')
        dialogue = Dialogue(role=role, priority=100, state=DialogueState.ACTIVE)
        manager.add_dialogue(dialogue)

        assert manager.can_start_dialogue(150) is True

    def test_can_start_dialogue_equal_is_rejected(self):
        """Equal-priority new dialogue is rejected (no preemption)."""
        manager = DialogueManager()
        role = DialogueRole(name='test')
        dialogue = Dialogue(role=role, priority=100, state=DialogueState.ACTIVE)
        manager.add_dialogue(dialogue)

        assert manager.can_start_dialogue(100) is False

    def test_can_start_dialogue_lower_is_rejected(self):
        """Lower-priority new dialogue is rejected."""
        manager = DialogueManager()
        role = DialogueRole(name='test')
        dialogue = Dialogue(role=role, priority=100, state=DialogueState.ACTIVE)
        manager.add_dialogue(dialogue)

        assert manager.can_start_dialogue(50) is False

    def test_can_start_dialogue_ignores_expression_priority(self):
        """Active expression does NOT block a new dialogue."""
        manager = DialogueManager()
        manager.set_expression_priority(200)

        assert manager.can_start_dialogue(50) is True

    def test_can_speak_equal_is_accepted(self):
        """Equal-priority Say is accepted (cooperative semantics)."""
        manager = DialogueManager()
        manager.set_expression_priority(0)

        assert manager.can_speak(0) is True

    def test_can_speak_higher_is_accepted(self):
        """Higher-priority Say is accepted."""
        manager = DialogueManager()
        manager.set_expression_priority(50)

        assert manager.can_speak(100) is True

    def test_can_speak_lower_is_rejected(self):
        """Lower-priority Say is rejected."""
        manager = DialogueManager()
        manager.set_expression_priority(100)

        assert manager.can_speak(50) is False

    def test_can_speak_ignores_dialogue_priority(self):
        """Active dialogues at any priority do NOT block Says."""
        manager = DialogueManager()
        role = DialogueRole(name='test')
        dialogue = Dialogue(role=role, priority=200, state=DialogueState.ACTIVE)
        manager.add_dialogue(dialogue)

        # No expression currently speaking → any Say can speak.
        assert manager.can_speak(0) is True
        assert manager.can_speak(255) is True

    def test_get_dialogue_for_interlocutor_found(self):
        """Find active dialogue for an interlocutor."""
        manager = DialogueManager()
        role = DialogueRole(name='test')
        interlocutor = Interlocutor(person_id='person_123')
        dialogue = Dialogue(
            role=role, interlocutor=interlocutor, state=DialogueState.ACTIVE
        )
        manager.add_dialogue(dialogue)

        result = manager.get_dialogue_for_interlocutor(interlocutor)

        assert result is dialogue

    def test_get_dialogue_for_interlocutor_not_found(self):
        """Return None when no dialogue exists for interlocutor."""
        manager = DialogueManager()
        result = manager.get_dialogue_for_interlocutor(
            Interlocutor(person_id='unknown')
        )
        assert result is None

    def test_get_dialogue_for_interlocutor_inactive(self):
        """Return None when dialogue exists but is not ACTIVE."""
        manager = DialogueManager()
        role = DialogueRole(name='test')
        interlocutor = Interlocutor(person_id='person_123')
        dialogue = Dialogue(
            role=role, interlocutor=interlocutor, state=DialogueState.PENDING
        )
        manager.add_dialogue(dialogue)

        result = manager.get_dialogue_for_interlocutor(interlocutor)

        assert result is None

    def test_get_dialogue_for_interlocutor_multiple(self):
        """Return the active dialogue when multiple exist for interlocutor."""
        manager = DialogueManager()
        role = DialogueRole(name='test')
        interlocutor = Interlocutor(person_id='person_123')

        d1 = Dialogue(
            role=role, interlocutor=interlocutor, state=DialogueState.COMPLETED
        )
        d2 = Dialogue(
            role=role, interlocutor=interlocutor, state=DialogueState.ACTIVE
        )

        manager.add_dialogue(d1)
        manager.add_dialogue(d2)

        result = manager.get_dialogue_for_interlocutor(interlocutor)

        assert result is d2

    def test_get_dialogue_for_group_interlocutor(self):
        """Group interlocutors are looked up the same way."""
        manager = DialogueManager()
        role = DialogueRole(name='test')
        interlocutor = Interlocutor(group_id='group_42')
        dialogue = Dialogue(
            role=role, interlocutor=interlocutor, state=DialogueState.ACTIVE
        )
        manager.add_dialogue(dialogue)

        assert manager.get_dialogue_for_interlocutor(interlocutor) is dialogue


class TestDialogueSessionUtterances:
    """Tests for session_start_index / session_utterances pre-fill semantics."""

    def test_session_utterances_default_full(self):
        """With session_start_index=0, session_utterances == history."""
        d = Dialogue(role=DialogueRole(name='test'))
        d.add_utterance('alice', 'hello', 1.0)
        d.add_utterance(ROBOT_SPEAKER_ID, 'hi', 2.0)
        assert d.session_utterances == d.history

    def test_session_utterances_skips_prefill(self):
        """session_utterances excludes pre-filled summary + break entries."""
        d = Dialogue(role=DialogueRole(name='test'))
        d.history.append(Utterance(
            timestamp=0.5, speaker_id=SUMMARY_SPEAKER_ID, text='earlier summary',
        ))
        d.history.append(Utterance(
            timestamp=0.9, speaker_id=SESSION_BREAK_SPEAKER_ID, text='',
        ))
        d.session_start_index = len(d.history)

        d.add_utterance('alice', 'hello', 1.0)
        d.add_utterance(ROBOT_SPEAKER_ID, 'hi', 2.0)

        assert len(d.history) == 4
        assert len(d.session_utterances) == 2
        assert d.session_utterances[0].text == 'hello'
        assert d.session_utterances[1].text == 'hi'

    def test_session_utterances_empty_after_prefill_only(self):
        """A pre-filled dialogue with no real session utterances is empty."""
        d = Dialogue(role=DialogueRole(name='test'))
        d.history.append(Utterance(
            timestamp=0.5, speaker_id=SUMMARY_SPEAKER_ID, text='earlier',
        ))
        d.session_start_index = len(d.history)
        assert d.session_utterances == []
