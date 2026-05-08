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

"""Unit tests for ConversationsHistoryStore."""

from chatbot_msgs.msg import DialogueRole
from dialogue_manager.conversations_history import (
    ConversationsHistoryStore,
    EXCLUDE_ASK,
)
from dialogue_manager.dialogue import (
    Dialogue,
    DialogueState,
    Interlocutor,
    ROBOT_SPEAKER_ID,
)


def _make_dialogue(
    role_name: str = 'test_role',
    person_id: str = '',
    group_id: str = '',
    utterances: list[tuple[str, str, float]] | None = None,
    started_at: float | None = None,
    ended_at: float | None = None,
) -> Dialogue:
    interlocutor = Interlocutor(person_id=person_id, group_id=group_id)
    d = Dialogue(
        role=DialogueRole(name=role_name),
        interlocutor=interlocutor,
        state=DialogueState.COMPLETED,
    )
    for speaker_id, text, ts in utterances or []:
        d.add_utterance(speaker_id, text, timestamp=ts)
    if started_at is not None:
        d.started_at = started_at
    if ended_at is not None:
        d.ended_at = ended_at
    return d


class TestArchive:
    """Tests for archiving dialogues into per-interlocutor history."""

    def test_archive_person_dialogue(self):
        """Person dialogues end up under the person's key."""
        store = ConversationsHistoryStore()
        d = _make_dialogue(
            person_id='alice',
            utterances=[('alice', 'hi', 1.0), (ROBOT_SPEAKER_ID, 'hello', 2.0)],
        )
        store.archive(d)
        assert store.history_for(Interlocutor(person_id='alice')) == [d]

    def test_archive_group_dialogue_with_members(self):
        """Group dialogues fan out to each member's personal history."""
        store = ConversationsHistoryStore()
        d = _make_dialogue(
            group_id='visitors',
            utterances=[(ROBOT_SPEAKER_ID, 'welcome', 1.0)],
        )
        store.archive(d, group_members=['alice', 'bob'])

        assert store.history_for(Interlocutor(group_id='visitors')) == [d]
        assert store.history_for(Interlocutor(person_id='alice')) == [d]
        assert store.history_for(Interlocutor(person_id='bob')) == [d]

    def test_archive_skips_unbound(self):
        """Empty (anonymous) interlocutor is not archived."""
        store = ConversationsHistoryStore()
        d = _make_dialogue(utterances=[('alice', 'hi', 1.0)])
        store.archive(d)
        assert store.history_for(Interlocutor()) == []

    def test_archive_skips_empty_history(self):
        """Dialogues with no utterances are not archived."""
        store = ConversationsHistoryStore()
        d = _make_dialogue(person_id='alice')
        store.archive(d)
        assert store.history_for(Interlocutor(person_id='alice')) == []


class TestBuildContext:
    """Tests for the LLM context construction."""

    def test_excludes_ask_by_default(self):
        """ASK-role dialogues are filtered out by the default filter."""
        store = ConversationsHistoryStore()
        chat = _make_dialogue(
            role_name=DialogueRole.DEFAULT_ROLE,
            person_id='alice',
            utterances=[('alice', 'tell me about yourself', 1.0)],
            ended_at=10.0,
        )
        ask = _make_dialogue(
            role_name=DialogueRole.ASK_ROLE,
            person_id='alice',
            utterances=[('alice', '42', 5.0)],
            ended_at=20.0,
        )
        store.archive(chat)
        store.archive(ask)

        ctx = store.build_context(Interlocutor(person_id='alice'), now=30.0)
        assert 'tell me about yourself' in ctx
        assert '42' not in ctx

    def test_custom_filter_pipeline(self):
        """Multiple filters are AND-combined."""
        store = ConversationsHistoryStore()
        d = _make_dialogue(
            person_id='alice',
            utterances=[('alice', 'hi', 1.0)],
            ended_at=10.0,
        )
        store.archive(d)

        keep_none = lambda _d: False  # noqa: E731
        ctx = store.build_context(
            Interlocutor(person_id='alice'),
            now=30.0,
            filters=[EXCLUDE_ASK, keep_none],
        )
        assert ctx == ''

    def test_summary_used_for_old_dialogues(self):
        """Older-than-threshold dialogues use the summary when present."""
        store = ConversationsHistoryStore()
        old = _make_dialogue(
            person_id='alice',
            utterances=[('alice', 'something verbose', 1.0)],
            ended_at=0.0,
        )
        old.summary = 'alice said something'
        old.summary_generated_at = 0.0
        store.archive(old)

        ctx = store.build_context(
            Interlocutor(person_id='alice'),
            now=10_000.0,
            summary_age_sec=300.0,
        )
        assert 'alice said something' in ctx
        assert 'verbose' not in ctx

    def test_summarizer_called_and_cached(self):
        """An old dialogue with no cached summary triggers the summarizer once."""
        store = ConversationsHistoryStore()
        d = _make_dialogue(
            person_id='alice',
            utterances=[('alice', 'long chat', 1.0)],
            ended_at=0.0,
        )
        store.archive(d)

        calls = {'n': 0}

        def summarizer(dialogue):
            calls['n'] += 1
            return 'summary text'

        ctx1 = store.build_context(
            Interlocutor(person_id='alice'),
            now=10_000.0,
            summarize=summarizer,
        )
        ctx2 = store.build_context(
            Interlocutor(person_id='alice'),
            now=10_000.0,
            summarize=summarizer,
        )
        assert 'summary text' in ctx1
        assert 'summary text' in ctx2
        assert calls['n'] == 1  # Cached on the dialogue, not regenerated
        assert d.summary == 'summary text'

    def test_recent_dialogue_renders_verbatim(self):
        """Dialogues newer than `summary_age_sec` are rendered in full."""
        store = ConversationsHistoryStore()
        d = _make_dialogue(
            person_id='alice',
            utterances=[('alice', 'fresh content', 1.0)],
            ended_at=100.0,
        )
        d.summary = 'should not be used'
        store.archive(d)

        ctx = store.build_context(
            Interlocutor(person_id='alice'),
            now=200.0,  # 100s old < 300s threshold
            summary_age_sec=300.0,
        )
        assert 'fresh content' in ctx
        assert 'should not be used' not in ctx


class TestPersistence:
    """Tests for JSON load/save round-tripping."""

    def test_save_and_load_round_trip(self, tmp_path):
        """save() then load() restores dialogues for both kinds of interlocutor."""
        store = ConversationsHistoryStore(storage_dir=tmp_path)
        chat = _make_dialogue(
            person_id='alice',
            utterances=[
                ('alice', 'hi', 1.0),
                (ROBOT_SPEAKER_ID, 'hello alice', 2.0),
            ],
            ended_at=2.0,
        )
        chat.summary = 'cached summary'
        chat.summary_generated_at = 5.0
        group = _make_dialogue(
            group_id='visitors',
            utterances=[(ROBOT_SPEAKER_ID, 'welcome', 1.0)],
            ended_at=1.0,
        )
        store.archive(chat)
        store.archive(group)

        store.save()

        restored = ConversationsHistoryStore(storage_dir=tmp_path)
        restored.load()

        alice_history = restored.history_for(Interlocutor(person_id='alice'))
        assert len(alice_history) == 1
        d = alice_history[0]
        assert d.dialogue_id == chat.dialogue_id
        assert d.interlocutor == Interlocutor(person_id='alice')
        assert [(u.speaker_id, u.text) for u in d.history] == [
            ('alice', 'hi'),
            (ROBOT_SPEAKER_ID, 'hello alice'),
        ]
        assert d.summary == 'cached summary'
        assert d.summary_generated_at == 5.0

        group_history = restored.history_for(Interlocutor(group_id='visitors'))
        assert len(group_history) == 1
        assert group_history[0].interlocutor == Interlocutor(group_id='visitors')

    def test_in_memory_only_when_no_storage_dir(self):
        """save() and load() are no-ops when no storage_dir is given."""
        store = ConversationsHistoryStore()
        d = _make_dialogue(
            person_id='alice',
            utterances=[('alice', 'hi', 1.0)],
        )
        store.archive(d)
        store.save()  # must not raise
        store.load()  # must not raise
        assert store.history_for(Interlocutor(person_id='alice')) == [d]
