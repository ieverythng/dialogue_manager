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

"""Unit tests for DialogueLifecycle (finalize + archive + summarize)."""

import asyncio
import time
from unittest.mock import MagicMock

from chatbot_msgs.msg import DialogueRole
from dialogue_manager.conversations_history import ConversationsHistoryStore
from dialogue_manager.dialogue import (
    Dialogue,
    DialogueManager,
    DialogueState,
    Interlocutor,
)
from dialogue_manager.dialogue_lifecycle import (
    DialogueLifecycle,
    default_summarizer,
)


def _build_lifecycle(
    dialogue_manager=None,
    conversations_store=None,
    summarizer=None,
    group_handler=None,
):
    """Construct a DialogueLifecycle wired with mocks."""
    mock_node = MagicMock()
    mock_node.get_clock.return_value.now.return_value.nanoseconds = int(1e9)
    return DialogueLifecycle(
        node=mock_node,
        dialogue_manager=dialogue_manager or DialogueManager(),
        conversations_store=conversations_store or ConversationsHistoryStore(),
        group_handler=group_handler,
        summarizer=summarizer,
    )


class TestDefaultSummarizer:
    """The built-in fallback summarizer must run without an LLM."""

    def test_renders_session_utterances(self):
        """default_summarizer returns the session's utterances as text."""
        d = Dialogue(role=DialogueRole(name='test'))
        d.add_utterance('alice', 'hi', 1.0)
        d.add_utterance('__myself__', 'hello', 2.0)
        out = asyncio.run(default_summarizer(d, []))
        assert 'alice: hi' in out
        assert '__myself__: hello' in out


class TestFinalizeAndArchive:
    """Async summarizer fires from finalize_and_archive."""

    def test_archives_and_invokes_summarizer(self):
        store = ConversationsHistoryStore()
        captured = {}

        async def my_summarizer(dialogue, prior_dialogues):
            captured['prior_count'] = len(prior_dialogues)
            captured['utt_count'] = len(dialogue.session_utterances)
            return 'cumulative summary'

        dm = DialogueManager()
        lifecycle = _build_lifecycle(
            dialogue_manager=dm,
            conversations_store=store,
            summarizer=my_summarizer,
        )

        d = Dialogue(
            role=DialogueRole(name='test'),
            interlocutor=Interlocutor(person_id='alice'),
            state=DialogueState.ACTIVE,
        )
        d.add_utterance('alice', 'hello', 1.0)
        dm.add_dialogue(d)

        lifecycle.finalize_and_archive(d)

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
        dm = DialogueManager()
        lifecycle = _build_lifecycle(
            dialogue_manager=dm,
            conversations_store=store,
        )

        d = Dialogue(
            role=DialogueRole(name='test'),
            interlocutor=Interlocutor(person_id='alice'),
            state=DialogueState.ACTIVE,
        )
        d.add_utterance('alice', 'hi', 1.0)
        dm.add_dialogue(d)

        lifecycle.finalize_and_archive(d)

        deadline = time.monotonic() + 2.0
        while d.summary is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert d.summary is not None
        assert 'alice: hi' in d.summary

    def test_group_dialogue_fans_out_to_members(self):
        """Group dialogues archive into each member's personal history."""
        gh = MagicMock()
        gh.members_of.side_effect = (
            lambda gid: ['alice', 'bob'] if gid == 'group_a' else []
        )
        store = ConversationsHistoryStore()
        dm = DialogueManager()
        lifecycle = _build_lifecycle(
            dialogue_manager=dm,
            conversations_store=store,
            group_handler=gh,
        )

        d = Dialogue(
            role=DialogueRole(name='test'),
            interlocutor=Interlocutor(group_id='group_a'),
            state=DialogueState.ACTIVE,
        )
        d.add_utterance('alice', 'hi everyone', 1.0)
        dm.add_dialogue(d)

        lifecycle.finalize_and_archive(d)

        assert store.history_for(Interlocutor(group_id='group_a')) == [d]
        assert store.history_for(Interlocutor(person_id='alice')) == [d]
        assert store.history_for(Interlocutor(person_id='bob')) == [d]

    def test_empty_session_skips_summarization(self):
        """A dialogue with no session utterances still archives but doesn't summarize."""
        store = ConversationsHistoryStore()
        dm = DialogueManager()
        called = {'n': 0}

        async def tracker(dialogue, prior_dialogues):
            called['n'] += 1
            return 'should not happen'

        lifecycle = _build_lifecycle(
            dialogue_manager=dm,
            conversations_store=store,
            summarizer=tracker,
        )

        d = Dialogue(
            role=DialogueRole(name='test'),
            interlocutor=Interlocutor(person_id='alice'),
            state=DialogueState.ACTIVE,
        )
        dm.add_dialogue(d)

        lifecycle.finalize_and_archive(d)

        # Empty dialogues are skipped by archive() too — no fan-out, no summary.
        assert store.history_for(Interlocutor(person_id='alice')) == []
        time.sleep(0.1)  # give a potential thread time to misbehave
        assert called['n'] == 0
        assert d.state == DialogueState.COMPLETED
