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

"""Unit tests for the interlocutor fan-out helper."""

from unittest.mock import MagicMock

from chatbot_msgs.msg import DialogueRole
from dialogue_manager.dialogue import Dialogue, Interlocutor
from dialogue_manager.fanout import (
    is_dialogue_present,
    related_interlocutors,
)


class TestRelatedInterlocutors:
    """Graph walk: primary → groups/members → co-members."""

    def test_no_group_handler_yields_only_primary(self):
        it = Interlocutor(person_id='alice')
        result = list(related_interlocutors(it, None))
        assert result == [it]

    def test_unbound_interlocutor_with_no_handler_yields_nothing(self):
        result = list(related_interlocutors(Interlocutor(), None))
        assert result == []

    def test_person_yields_primary_then_groups_then_co_members(self):
        gh = MagicMock()
        gh.groups_containing.return_value = ['group_a']
        gh.co_members_of.return_value = ['bob', 'charlie']
        gh.members_of.return_value = []

        result = list(related_interlocutors(
            Interlocutor(person_id='alice'), gh
        ))

        assert result == [
            Interlocutor(person_id='alice'),
            Interlocutor(group_id='group_a'),
            Interlocutor(person_id='bob'),
            Interlocutor(person_id='charlie'),
        ]

    def test_group_yields_primary_then_members(self):
        gh = MagicMock()
        gh.members_of.return_value = ['alice', 'bob']

        result = list(related_interlocutors(
            Interlocutor(group_id='group_a'), gh
        ))

        assert result == [
            Interlocutor(group_id='group_a'),
            Interlocutor(person_id='alice'),
            Interlocutor(person_id='bob'),
        ]
        # group_handler.groups_containing should NOT be called for a group
        # interlocutor — there's no symmetric "groups containing a group".
        gh.groups_containing.assert_not_called()

    def test_no_duplicates(self):
        """Same interlocutor yielded by multiple edges appears once."""
        gh = MagicMock()
        # Pathological: alice is co-member of herself (shouldn't happen but be safe)
        gh.groups_containing.return_value = ['group_a']
        gh.co_members_of.return_value = ['alice', 'bob']
        gh.members_of.return_value = []

        result = list(related_interlocutors(
            Interlocutor(person_id='alice'), gh
        ))

        person_ids = [it.person_id for it in result if it.person_id]
        assert person_ids.count('alice') == 1

    def test_empty_member_ids_skipped(self):
        gh = MagicMock()
        gh.members_of.return_value = ['', 'alice', '']

        result = list(related_interlocutors(
            Interlocutor(group_id='group_a'), gh
        ))

        assert result == [
            Interlocutor(group_id='group_a'),
            Interlocutor(person_id='alice'),
        ]


class TestIsDialoguePresent:
    """Group dialogues always present; persons defer to the query."""

    def test_group_dialogue_always_present(self):
        d = Dialogue(
            role=DialogueRole(name='__default__'),
            interlocutor=Interlocutor(group_id='group_a'),
        )
        assert is_dialogue_present(d, lambda _pid: False) is True

    def test_person_dialogue_consults_query(self):
        d = Dialogue(
            role=DialogueRole(name='__default__'),
            interlocutor=Interlocutor(person_id='alice'),
        )
        assert is_dialogue_present(d, lambda pid: pid == 'alice') is True
        assert is_dialogue_present(d, lambda pid: pid != 'alice') is False

    def test_unbound_dialogue_treated_as_present(self):
        d = Dialogue(role=DialogueRole(name='__default__'))
        assert is_dialogue_present(d, lambda _pid: False) is True
