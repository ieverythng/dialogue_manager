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
Interlocutor fan-out graph traversal helpers.

Both speech-time recording (`SpeechHandler`) and Say-time recording
(`SkillServers`) need to walk the same graph from a given Interlocutor:

- The interlocutor themselves.
- If they're a person: every group they're in, plus every co-member.
- If they're a group: every member.

This module captures *only* that graph traversal. Each caller still
decides what to do with each interlocutor (spawn vs fetch a dialogue,
filter by presence, etc.) because those rules are inherently context-
specific.
"""

from collections.abc import Iterator

from .dialogue import Dialogue, Interlocutor


def related_interlocutors(
    interlocutor: Interlocutor, group_handler
) -> Iterator[Interlocutor]:
    """
    Yield interlocutors related to `interlocutor`, primary first.

    Order:
      1. The interlocutor themselves (if bound).
      2. For a person: every group they're in, then every co-member.
         For a group: every member.

    The same interlocutor is never yielded twice. When `group_handler`
    is ``None``, only the primary is yielded.
    """
    seen: set[tuple[str, str]] = set()

    def _yield(it: Interlocutor) -> Iterator[Interlocutor]:
        key = (it.person_id, it.group_id)
        if key in seen:
            return
        seen.add(key)
        yield it

    if interlocutor.is_bound:
        yield from _yield(interlocutor)

    if group_handler is None:
        return

    if interlocutor.is_group:
        for member_id in group_handler.members_of(interlocutor.group_id):
            if member_id:
                yield from _yield(Interlocutor(person_id=member_id))
        return

    if interlocutor.person_id:
        for group_id in group_handler.groups_containing(
            interlocutor.person_id
        ):
            yield from _yield(Interlocutor(group_id=group_id))
        for co_member_id in group_handler.co_members_of(
            interlocutor.person_id
        ):
            if co_member_id:
                yield from _yield(Interlocutor(person_id=co_member_id))


def is_dialogue_present(
    dialogue: Dialogue, presence_query
) -> bool:
    """
    Return True if `dialogue` should accept fan-out under the given query.

    Group dialogues are always considered present — group "absence" is
    signalled by dispersal (the dialogue is finalised). Person
    dialogues defer to `presence_query(person_id)`.
    """
    if dialogue.interlocutor.is_group:
        return True
    if not dialogue.interlocutor.person_id:
        return True
    return presence_query(dialogue.interlocutor.person_id)
