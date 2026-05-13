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

"""Unit tests for skill_servers.py with mocked ROS2 dependencies."""

import time
from unittest.mock import MagicMock, patch

from chatbot_msgs.msg import DialogueRole
from dialogue_manager.conversations_history import ConversationsHistoryStore
from dialogue_manager.dialogue import (
    Dialogue,
    DialogueManager,
    DialogueState,
    Interlocutor,
    SESSION_BREAK_SPEAKER_ID,
    SUMMARY_SPEAKER_ID,
)
from dialogue_manager.skill_servers import default_summarizer, SkillServers


def _empty_group_resolver(_group_id: str) -> list[str]:
    return []


class TestSkillServersInit:
    """Tests for SkillServers initialization."""

    def test_init_sets_properties(self):
        """Servers store all provided dependencies."""
        mock_node = MagicMock()
        mock_dialogue_manager = MagicMock()
        mock_chatbot_client = MagicMock()
        mock_say_client = MagicMock()
        mock_captions_pub = MagicMock()

        servers = SkillServers(
            node=mock_node,
            dialogue_manager=mock_dialogue_manager,
            chatbot_client=mock_chatbot_client,
            say_client=mock_say_client,
            conversations_store=ConversationsHistoryStore(),
            group_resolver=_empty_group_resolver,
            closed_captions_pub=mock_captions_pub
        )

        assert servers._node is mock_node
        assert servers._dialogue_manager is mock_dialogue_manager
        assert servers._chatbot_client is mock_chatbot_client
        assert servers._say_client is mock_say_client
        assert servers._closed_captions_pub is mock_captions_pub

    def test_init_starts_inactive(self):
        """Servers start in inactive state."""
        servers = SkillServers(
            node=MagicMock(),
            dialogue_manager=MagicMock(),
            chatbot_client=MagicMock(),
            say_client=MagicMock(),
            conversations_store=ConversationsHistoryStore(),
            group_resolver=_empty_group_resolver,
            closed_captions_pub=MagicMock()
        )

        assert servers._is_active is False


class TestSkillServersSetActive:
    """Tests for set_active method."""

    def test_set_active_true(self):
        """set_active sets to True."""
        servers = SkillServers(
            node=MagicMock(),
            dialogue_manager=MagicMock(),
            chatbot_client=MagicMock(),
            say_client=MagicMock(),
            conversations_store=ConversationsHistoryStore(),
            group_resolver=_empty_group_resolver,
            closed_captions_pub=MagicMock()
        )

        servers.set_active(True)

        assert servers._is_active is True

    def test_set_active_false(self):
        """set_active sets to False."""
        servers = SkillServers(
            node=MagicMock(),
            dialogue_manager=MagicMock(),
            chatbot_client=MagicMock(),
            say_client=MagicMock(),
            conversations_store=ConversationsHistoryStore(),
            group_resolver=_empty_group_resolver,
            closed_captions_pub=MagicMock()
        )
        servers._is_active = True

        servers.set_active(False)

        assert servers._is_active is False


class TestSkillServersCreateServers:
    """Tests for create_servers method."""

    def test_create_servers_creates_all(self):
        """create_servers creates Chat, Ask, and Say action servers."""
        mock_node = MagicMock()
        servers = SkillServers(
            node=mock_node,
            dialogue_manager=MagicMock(),
            chatbot_client=MagicMock(),
            say_client=MagicMock(),
            conversations_store=ConversationsHistoryStore(),
            group_resolver=_empty_group_resolver,
            closed_captions_pub=MagicMock()
        )

        with patch('dialogue_manager.skill_servers.ActionServer') as mock_as:
            servers.create_servers()

        # Should create 3 action servers
        assert mock_as.call_count == 3


class TestSkillServersDestroy:
    """Tests for destroy method."""

    def test_destroy_cleans_up_servers(self):
        """Destroy cleans up all action servers."""
        servers = SkillServers(
            node=MagicMock(),
            dialogue_manager=MagicMock(),
            chatbot_client=MagicMock(),
            say_client=MagicMock(),
            conversations_store=ConversationsHistoryStore(),
            group_resolver=_empty_group_resolver,
            closed_captions_pub=MagicMock()
        )

        mock_chat_server = MagicMock()
        mock_ask_server = MagicMock()
        mock_say_server = MagicMock()

        servers._chat_server = mock_chat_server
        servers._ask_server = mock_ask_server
        servers._say_server = mock_say_server

        servers.destroy()

        mock_chat_server.destroy.assert_called_once()
        mock_ask_server.destroy.assert_called_once()
        mock_say_server.destroy.assert_called_once()


class TestSkillServersGoalCallbacks:
    """Tests for goal acceptance callbacks."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_node = MagicMock()
        self.mock_dialogue_manager = DialogueManager()
        self.mock_chatbot_client = MagicMock()
        self.mock_say_client = MagicMock()

        self.servers = SkillServers(
            node=self.mock_node,
            dialogue_manager=self.mock_dialogue_manager,
            chatbot_client=self.mock_chatbot_client,
            say_client=self.mock_say_client,
            conversations_store=ConversationsHistoryStore(),
            group_resolver=_empty_group_resolver,
            closed_captions_pub=MagicMock()
        )

    def test_chat_goal_rejected_when_inactive(self):
        """Chat goal rejected when server is inactive."""
        mock_request = MagicMock()
        mock_request.meta.priority = 128

        from rclpy.action import GoalResponse
        result = self.servers._chat_goal_callback(mock_request)

        assert result == GoalResponse.REJECT

    def test_chat_goal_accepted_when_active_with_higher_priority(self):
        """Chat goal accepted when active with higher priority."""
        self.servers._is_active = True

        mock_request = MagicMock()
        mock_request.meta.priority = 128

        from rclpy.action import GoalResponse
        result = self.servers._chat_goal_callback(mock_request)

        assert result == GoalResponse.ACCEPT

    def test_ask_goal_rejected_when_inactive(self):
        """Ask goal rejected when server is inactive."""
        mock_request = MagicMock()
        mock_request.meta.priority = 128

        from rclpy.action import GoalResponse
        result = self.servers._ask_goal_callback(mock_request)

        assert result == GoalResponse.REJECT

    def test_ask_goal_accepted_when_active_with_higher_priority(self):
        """Ask goal accepted when active with higher priority."""
        self.servers._is_active = True

        mock_request = MagicMock()
        mock_request.meta.priority = 128

        from rclpy.action import GoalResponse
        result = self.servers._ask_goal_callback(mock_request)

        assert result == GoalResponse.ACCEPT

    def test_say_goal_rejected_when_inactive(self):
        """Say goal rejected when server is inactive."""
        mock_request = MagicMock()
        mock_request.meta.priority = 128

        from rclpy.action import GoalResponse
        result = self.servers._say_goal_callback(mock_request)

        assert result == GoalResponse.REJECT

    def test_say_goal_accepted_when_active_with_higher_priority(self):
        """Say goal accepted when active with higher priority."""
        self.servers._is_active = True

        mock_request = MagicMock()
        mock_request.meta.priority = 128

        from rclpy.action import GoalResponse
        result = self.servers._say_goal_callback(mock_request)

        assert result == GoalResponse.ACCEPT


class TestSkillServersPriorityRejection:
    """Tests for priority-based goal rejection."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_dialogue_manager = DialogueManager()

        self.servers = SkillServers(
            node=MagicMock(),
            dialogue_manager=self.mock_dialogue_manager,
            chatbot_client=MagicMock(),
            say_client=MagicMock(),
            conversations_store=ConversationsHistoryStore(),
            group_resolver=_empty_group_resolver,
            closed_captions_pub=MagicMock()
        )
        self.servers._is_active = True

    def test_chat_goal_rejected_when_lower_than_active_dialogue(self):
        """Chat goal rejected when lower than an active dialogue's priority."""
        active = Dialogue(
            role=DialogueRole(name='test'),
            interlocutor=Interlocutor(person_id='alice'),
            priority=200,
            state=DialogueState.ACTIVE,
        )
        self.mock_dialogue_manager.add_dialogue(active)

        mock_request = MagicMock()
        mock_request.meta.priority = 100

        from rclpy.action import GoalResponse
        result = self.servers._chat_goal_callback(mock_request)

        assert result == GoalResponse.REJECT

    def test_chat_goal_ignores_expression_priority(self):
        """Chat acceptance is decoupled from current expression priority."""
        # An expression speaking at high priority must NOT block a new Chat.
        self.mock_dialogue_manager.set_expression_priority(200)

        mock_request = MagicMock()
        mock_request.meta.priority = 100

        from rclpy.action import GoalResponse
        result = self.servers._chat_goal_callback(mock_request)

        assert result == GoalResponse.ACCEPT

    def test_ask_goal_rejected_when_lower_than_active_dialogue(self):
        """Ask goal rejected when lower than an active dialogue's priority."""
        active = Dialogue(
            role=DialogueRole(name='test'),
            interlocutor=Interlocutor(person_id='alice'),
            priority=200,
            state=DialogueState.ACTIVE,
        )
        self.mock_dialogue_manager.add_dialogue(active)

        mock_request = MagicMock()
        mock_request.meta.priority = 100

        from rclpy.action import GoalResponse
        result = self.servers._ask_goal_callback(mock_request)

        assert result == GoalResponse.REJECT

    def test_say_goal_rejected_when_lower_than_expression(self):
        """Say goal rejected when below current expression priority."""
        self.mock_dialogue_manager.set_expression_priority(200)

        mock_request = MagicMock()
        mock_request.meta.priority = 100

        from rclpy.action import GoalResponse
        result = self.servers._say_goal_callback(mock_request)

        assert result == GoalResponse.REJECT

    def test_say_goal_ignores_dialogue_priority(self):
        """Say goal can speak alongside any active dialogue."""
        active = Dialogue(
            role=DialogueRole(name='test'),
            interlocutor=Interlocutor(person_id='alice'),
            priority=200,
            state=DialogueState.ACTIVE,
        )
        self.mock_dialogue_manager.add_dialogue(active)

        mock_request = MagicMock()
        mock_request.meta.priority = 0

        from rclpy.action import GoalResponse
        result = self.servers._say_goal_callback(mock_request)

        assert result == GoalResponse.ACCEPT

    def test_say_goal_accepted_at_equal_expression_priority(self):
        """Equal-priority Say is accepted (cooperative)."""
        self.mock_dialogue_manager.set_expression_priority(100)

        mock_request = MagicMock()
        mock_request.meta.priority = 100

        from rclpy.action import GoalResponse
        result = self.servers._say_goal_callback(mock_request)

        assert result == GoalResponse.ACCEPT


class TestBroadcastSayUtterance:
    """Say without person_id/group_id should land in every active dialogue."""

    def _build(self):
        dm = DialogueManager()
        servers = SkillServers(
            node=MagicMock(),
            dialogue_manager=dm,
            chatbot_client=MagicMock(),
            say_client=MagicMock(),
            conversations_store=ConversationsHistoryStore(),
            group_resolver=_empty_group_resolver,
            closed_captions_pub=MagicMock(),
        )
        # _strip_markup falls back to raw text when no executor is configured.
        return dm, servers

    def test_broadcast_records_into_all_active_bound_dialogues(self):
        from dialogue_manager.dialogue import ROBOT_SPEAKER_ID
        dm, servers = self._build()
        alice = Dialogue(
            role=DialogueRole(name='__default__'),
            interlocutor=Interlocutor(person_id='alice'),
            state=DialogueState.ACTIVE,
        )
        bob = Dialogue(
            role=DialogueRole(name='__default__'),
            interlocutor=Interlocutor(person_id='bob'),
            state=DialogueState.ACTIVE,
        )
        dm.add_dialogue(alice)
        dm.add_dialogue(bob)

        servers._broadcast_say_utterance('Hello guys')

        assert len(alice.history) == 1
        assert len(bob.history) == 1
        for d in (alice, bob):
            assert d.history[0].speaker_id == ROBOT_SPEAKER_ID
            assert d.history[0].text == 'Hello guys'

    def test_broadcast_skips_unbound_dialogues(self):
        dm, servers = self._build()
        unbound = Dialogue(
            role=DialogueRole(name='__default__'),
            interlocutor=Interlocutor(),
            state=DialogueState.ACTIVE,
        )
        dm.add_dialogue(unbound)

        servers._broadcast_say_utterance('Hello')

        assert unbound.history == []

    def test_broadcast_no_active_dialogues_is_noop(self):
        _, servers = self._build()
        # Must not raise.
        servers._broadcast_say_utterance('Hello')

    def test_broadcast_includes_active_group_dialogues(self):
        from dialogue_manager.dialogue import ROBOT_SPEAKER_ID
        dm, servers = self._build()
        alice = Dialogue(
            role=DialogueRole(name='__default__'),
            interlocutor=Interlocutor(person_id='alice'),
            state=DialogueState.ACTIVE,
        )
        group_a = Dialogue(
            role=DialogueRole(name='__default__'),
            interlocutor=Interlocutor(group_id='group_a'),
            state=DialogueState.ACTIVE,
        )
        dm.add_dialogue(alice)
        dm.add_dialogue(group_a)

        servers._broadcast_say_utterance('Hello group')

        for d in (alice, group_a):
            assert len(d.history) == 1
            assert d.history[0].text == 'Hello group'
            assert d.history[0].speaker_id == ROBOT_SPEAKER_ID

    def test_broadcast_skips_absent_person_dialogues(self):
        """A person dialogue whose interlocutor is absent is skipped.

        Group dialogues remain — group "absence" is handled by dispersal,
        not by the presence flag.
        """
        dm, servers = self._build()
        alice = Dialogue(
            role=DialogueRole(name='__default__'),
            interlocutor=Interlocutor(person_id='alice'),
            state=DialogueState.ACTIVE,
        )
        bob = Dialogue(
            role=DialogueRole(name='__default__'),
            interlocutor=Interlocutor(person_id='bob'),
            state=DialogueState.ACTIVE,
        )
        group_a = Dialogue(
            role=DialogueRole(name='__default__'),
            interlocutor=Interlocutor(group_id='group_a'),
            state=DialogueState.ACTIVE,
        )
        alice.interlocutor_present = False
        for d in (alice, bob, group_a):
            dm.add_dialogue(d)

        servers._broadcast_say_utterance('Hello room')

        assert alice.history == []
        assert len(bob.history) == 1
        assert len(group_a.history) == 1


class TestAddressedSayFanOut:
    """A Say addressed to a person or group fans out to related dialogues."""

    def _build(self, group_handler=None, group_resolver=None):
        dm = DialogueManager()
        servers = SkillServers(
            node=MagicMock(),
            dialogue_manager=dm,
            chatbot_client=MagicMock(),
            say_client=MagicMock(),
            conversations_store=ConversationsHistoryStore(),
            group_resolver=group_resolver or _empty_group_resolver,
            closed_captions_pub=MagicMock(),
            group_handler=group_handler,
        )
        return dm, servers

    def test_group_say_fans_out_to_members(self):
        from dialogue_manager.dialogue import ROBOT_SPEAKER_ID
        dm, servers = self._build(
            group_resolver=lambda gid: ['alice', 'bob'] if gid == 'group_a' else []
        )
        alice = Dialogue(
            role=DialogueRole(name='__default__'),
            interlocutor=Interlocutor(person_id='alice'),
            state=DialogueState.ACTIVE,
        )
        bob = Dialogue(
            role=DialogueRole(name='__default__'),
            interlocutor=Interlocutor(person_id='bob'),
            state=DialogueState.ACTIVE,
        )
        group_a = Dialogue(
            role=DialogueRole(name='__default__'),
            interlocutor=Interlocutor(group_id='group_a'),
            state=DialogueState.ACTIVE,
        )
        for d in (alice, bob, group_a):
            dm.add_dialogue(d)

        servers._record_say_utterance(
            Interlocutor(group_id='group_a'), "Let's dance"
        )

        for d in (alice, bob, group_a):
            assert len(d.history) == 1
            assert d.history[0].text == "Let's dance"
            assert d.history[0].speaker_id == ROBOT_SPEAKER_ID

    def test_person_say_fans_out_to_groups_and_co_members(self):
        from dialogue_manager.dialogue import ROBOT_SPEAKER_ID
        gh = MagicMock()
        gh.groups_containing.side_effect = (
            lambda pid: ['group_a'] if pid == 'alice' else []
        )
        gh.co_members_of.side_effect = (
            lambda pid: {'bob'} if pid == 'alice' else set()
        )
        dm, servers = self._build(group_handler=gh)
        alice = Dialogue(
            role=DialogueRole(name='__default__'),
            interlocutor=Interlocutor(person_id='alice'),
            state=DialogueState.ACTIVE,
        )
        bob = Dialogue(
            role=DialogueRole(name='__default__'),
            interlocutor=Interlocutor(person_id='bob'),
            state=DialogueState.ACTIVE,
        )
        group_a = Dialogue(
            role=DialogueRole(name='__default__'),
            interlocutor=Interlocutor(group_id='group_a'),
            state=DialogueState.ACTIVE,
        )
        for d in (alice, bob, group_a):
            dm.add_dialogue(d)

        servers._record_say_utterance(
            Interlocutor(person_id='alice'), 'Hi Alice'
        )

        for d in (alice, bob, group_a):
            assert len(d.history) == 1
            assert d.history[0].text == 'Hi Alice'
            assert d.history[0].speaker_id == ROBOT_SPEAKER_ID

    def test_addressed_say_skips_absent_co_member(self):
        """An addressed Say skips co-member dialogues whose interlocutor is absent.

        Group dialogues remain — only person dialogues participate in
        presence-driven filtering.
        """
        from dialogue_manager.dialogue import ROBOT_SPEAKER_ID
        gh = MagicMock()
        gh.groups_containing.side_effect = (
            lambda pid: ['group_a'] if pid in ('alice', 'bob') else []
        )
        gh.co_members_of.side_effect = (
            lambda pid: {'bob'} if pid == 'alice'
            else {'alice'} if pid == 'bob' else set()
        )
        dm, servers = self._build(group_handler=gh)
        alice = Dialogue(
            role=DialogueRole(name='__default__'),
            interlocutor=Interlocutor(person_id='alice'),
            state=DialogueState.ACTIVE,
        )
        bob = Dialogue(
            role=DialogueRole(name='__default__'),
            interlocutor=Interlocutor(person_id='bob'),
            state=DialogueState.ACTIVE,
        )
        group_a = Dialogue(
            role=DialogueRole(name='__default__'),
            interlocutor=Interlocutor(group_id='group_a'),
            state=DialogueState.ACTIVE,
        )
        # Bob has left.
        bob.interlocutor_present = False
        for d in (alice, bob, group_a):
            dm.add_dialogue(d)

        servers._record_say_utterance(
            Interlocutor(person_id='alice'), 'Hi Alice'
        )

        assert len(alice.history) == 1
        assert alice.history[0].speaker_id == ROBOT_SPEAKER_ID
        assert len(group_a.history) == 1
        # Bob is absent; his dialogue must not collect the utterance.
        assert bob.history == []

    def test_no_active_dialogue_falls_back_to_synthetic(self):
        """When nothing is active for the target, a synthetic gets archived."""
        dm, servers = self._build(
            group_resolver=lambda gid: ['alice'] if gid == 'group_a' else []
        )
        # No active dialogues at all.
        servers._record_say_utterance(
            Interlocutor(group_id='group_a'), 'Hello'
        )
        # The synthetic was archived under group_a and fanned out to alice.
        archived_group = servers._conversations_store.history_for(
            Interlocutor(group_id='group_a')
        )
        archived_alice = servers._conversations_store.history_for(
            Interlocutor(person_id='alice')
        )
        assert len(archived_group) == 1
        assert len(archived_alice) == 1


class TestSkillServersCancelCallback:
    """Tests for cancel callback."""

    def test_cancel_callback_accepts(self):
        """Cancel callback accepts cancellation requests."""
        servers = SkillServers(
            node=MagicMock(),
            dialogue_manager=MagicMock(),
            chatbot_client=MagicMock(),
            say_client=MagicMock(),
            conversations_store=ConversationsHistoryStore(),
            group_resolver=_empty_group_resolver,
            closed_captions_pub=MagicMock()
        )

        mock_goal_handle = MagicMock()

        from rclpy.action import CancelResponse
        result = servers._cancel_callback(mock_goal_handle)

        assert result == CancelResponse.ACCEPT


def _build_servers(
    dialogue_manager=None,
    chatbot_client=None,
    conversations_store=None,
    summarizer=None,
):
    """Build a SkillServers wired with mocks for the tests below."""
    mock_node = MagicMock()
    mock_node.get_clock.return_value.now.return_value.nanoseconds = int(1e9)
    return SkillServers(
        node=mock_node,
        dialogue_manager=dialogue_manager or DialogueManager(),
        chatbot_client=chatbot_client,
        say_client=MagicMock(),
        conversations_store=conversations_store or ConversationsHistoryStore(),
        group_resolver=_empty_group_resolver,
        closed_captions_pub=MagicMock(),
        summarizer=summarizer,
    )


class TestDefaultSummarizer:
    """The built-in fallback summarizer must run without an LLM."""

    def test_renders_session_utterances(self):
        """default_summarizer returns the session's utterances as text."""
        import asyncio
        d = Dialogue(role=DialogueRole(name='test'))
        d.add_utterance('alice', 'hi', 1.0)
        d.add_utterance('__myself__', 'hello', 2.0)
        out = asyncio.run(default_summarizer(d, []))
        assert 'alice: hi' in out
        assert '__myself__: hello' in out


class TestPreloadSummary:
    """SkillServers._preload_summary uses the most recent prior summary."""

    def test_no_prior_dialogues_is_noop(self):
        servers = _build_servers()
        d = Dialogue(
            role=DialogueRole(name='test'),
            interlocutor=Interlocutor(person_id='alice'),
        )
        servers._preload_summary(d)
        assert d.session_start_index == 0
        assert d.history == []

    def test_prior_without_summary_skips_silently(self):
        store = ConversationsHistoryStore()
        prior = Dialogue(
            role=DialogueRole(name='test'),
            interlocutor=Interlocutor(person_id='alice'),
            state=DialogueState.COMPLETED,
            ended_at=100.0,
        )
        prior.add_utterance('alice', 'hi', 99.0)
        store.archive(prior)

        servers = _build_servers(conversations_store=store)
        d = Dialogue(
            role=DialogueRole(name='test'),
            interlocutor=Interlocutor(person_id='alice'),
        )
        servers._preload_summary(d)
        assert d.session_start_index == 0
        assert d.history == []

    def test_prior_with_summary_pre_fills_history(self):
        store = ConversationsHistoryStore()
        prior = Dialogue(
            role=DialogueRole(name='test'),
            interlocutor=Interlocutor(person_id='alice'),
            state=DialogueState.COMPLETED,
            ended_at=100.0,
        )
        prior.add_utterance('alice', 'hi', 99.0)
        prior.summary = 'we said hi'
        store.archive(prior)

        servers = _build_servers(conversations_store=store)
        d = Dialogue(
            role=DialogueRole(name='test'),
            interlocutor=Interlocutor(person_id='alice'),
        )
        servers._preload_summary(d)

        assert d.session_start_index == 2
        assert len(d.history) == 2
        assert d.history[0].speaker_id == SUMMARY_SPEAKER_ID
        assert d.history[0].text == 'we said hi'
        assert d.history[1].speaker_id == SESSION_BREAK_SPEAKER_ID
        # New utterances live in the session range.
        d.add_utterance('alice', 'new', 200.0)
        assert d.session_utterances[-1].text == 'new'

    def test_uses_most_recent_prior(self):
        store = ConversationsHistoryStore()
        for name, ended in (('older', 50.0), ('newer', 200.0)):
            p = Dialogue(
                role=DialogueRole(name='test'),
                interlocutor=Interlocutor(person_id='alice'),
                state=DialogueState.COMPLETED,
                ended_at=ended,
            )
            p.add_utterance('alice', 'x', ended - 1)
            p.summary = name
            store.archive(p)

        servers = _build_servers(conversations_store=store)
        d = Dialogue(
            role=DialogueRole(name='test'),
            interlocutor=Interlocutor(person_id='alice'),
        )
        servers._preload_summary(d)
        assert d.history[0].text == 'newer'


class TestFinalizeAndArchive:
    """Async summarizer fires from _finalize_and_archive."""

    def test_archives_and_invokes_summarizer(self):
        store = ConversationsHistoryStore()
        captured = {}

        async def my_summarizer(dialogue, prior_dialogues):
            captured['prior_count'] = len(prior_dialogues)
            captured['utt_count'] = len(dialogue.session_utterances)
            return 'cumulative summary'

        servers = _build_servers(
            conversations_store=store,
            summarizer=my_summarizer,
        )

        d = Dialogue(
            role=DialogueRole(name='test'),
            interlocutor=Interlocutor(person_id='alice'),
            state=DialogueState.ACTIVE,
        )
        d.add_utterance('alice', 'hello', 1.0)
        servers._dialogue_manager.add_dialogue(d)

        # The dialogue is a dataclass — summary is a regular attribute. Spin
        # until the daemon thread populates it.
        servers._finalize_and_archive(d)

        # Archive happened synchronously.
        assert store.history_for(Interlocutor(person_id='alice')) == [d]
        assert d.state == DialogueState.COMPLETED

        # Summary lands asynchronously — give it up to 2s.
        deadline = time.monotonic() + 2.0
        while d.summary is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert d.summary == 'cumulative summary'
        assert captured['utt_count'] == 1
        # No prior dialogues for alice before this one.
        assert captured['prior_count'] == 0

    def test_default_summarizer_runs_when_none_provided(self):
        store = ConversationsHistoryStore()
        servers = _build_servers(conversations_store=store)

        d = Dialogue(
            role=DialogueRole(name='test'),
            interlocutor=Interlocutor(person_id='alice'),
            state=DialogueState.ACTIVE,
        )
        d.add_utterance('alice', 'hi', 1.0)
        servers._dialogue_manager.add_dialogue(d)

        servers._finalize_and_archive(d)

        deadline = time.monotonic() + 2.0
        while d.summary is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert d.summary is not None
        assert 'alice: hi' in d.summary
