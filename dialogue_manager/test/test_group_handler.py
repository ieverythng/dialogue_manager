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

"""Unit tests for the GroupHandler."""

from unittest.mock import MagicMock

from dialogue_manager.group_handler import GroupHandler
from hri_msgs.msg import Group


def _make_msg(group_id: str, members: list[str]) -> Group:
    msg = Group()
    msg.group_id = group_id
    msg.members = list(members)
    return msg


def _build() -> tuple[GroupHandler, list, list]:
    """Return a handler plus the lists each callback writes to."""
    changes: list[tuple[str, frozenset[str]]] = []
    dispersals: list[str] = []
    handler = GroupHandler(
        node=MagicMock(),
        on_group_changed=lambda gid, members: changes.append((gid, members)),
        on_group_dispersed=lambda gid: dispersals.append(gid),
    )
    return handler, changes, dispersals


class TestGroupTracking:
    """Receiving Group messages updates membership and fires callbacks."""

    def test_first_message_registers_group(self):
        handler, changes, _ = _build()
        handler._on_group(_make_msg('g1', ['alice', 'bob']))

        assert handler.members_of('g1') == ['alice', 'bob']
        assert changes == [('g1', frozenset({'alice', 'bob'}))]

    def test_same_membership_does_not_refire_change(self):
        handler, changes, _ = _build()
        handler._on_group(_make_msg('g1', ['alice', 'bob']))
        handler._on_group(_make_msg('g1', ['alice', 'bob']))
        assert len(changes) == 1

    def test_membership_change_fires_change(self):
        handler, changes, _ = _build()
        handler._on_group(_make_msg('g1', ['alice']))
        handler._on_group(_make_msg('g1', ['alice', 'bob']))
        assert len(changes) == 2
        assert changes[-1] == ('g1', frozenset({'alice', 'bob'}))

    def test_empty_members_disperses(self):
        handler, _, dispersals = _build()
        handler._on_group(_make_msg('g1', ['alice', 'bob']))
        handler._on_group(_make_msg('g1', []))
        assert handler.members_of('g1') == []
        assert dispersals == ['g1']

    def test_dispersal_of_unknown_group_is_noop(self):
        handler, _, dispersals = _build()
        handler._on_group(_make_msg('unknown', []))
        assert dispersals == []


class TestGroupQueries:
    """The helper queries return expected sets/lists."""

    def test_groups_containing(self):
        handler, _, _ = _build()
        handler._on_group(_make_msg('g1', ['alice', 'bob']))
        handler._on_group(_make_msg('g2', ['alice', 'charlie']))

        assert handler.groups_containing('alice') == ['g1', 'g2']
        assert handler.groups_containing('bob') == ['g1']
        assert handler.groups_containing('dave') == []

    def test_co_members_of(self):
        handler, _, _ = _build()
        handler._on_group(_make_msg('g1', ['alice', 'bob']))
        handler._on_group(_make_msg('g2', ['alice', 'charlie']))

        co = handler.co_members_of('alice')
        assert co == {'bob', 'charlie'}
        # Alice herself isn't in her own co-member set.
        assert 'alice' not in co

    def test_known_groups(self):
        handler, _, _ = _build()
        handler._on_group(_make_msg('g1', ['alice']))
        handler._on_group(_make_msg('g2', ['bob']))
        handler._on_group(_make_msg('g1', []))
        assert handler.known_groups == ['g2']
