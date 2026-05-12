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

"""Execute a multi-party dialogue scenario from a small DSL.

The DSL (see `test/dialogues/dialogue_1.md`) lets us describe a
multi-party conversation as a sequence of speaker turns and stage
directions, with inline `# CHECK:` assertions verifying the
dialogue-manager state at that point.

Grammar
-------
- `<ID>:` followed by indented `- <text>` lines:
      the speaker utters those lines, in order. `R` is the robot
      (broadcast Say); any other ID is a person whose voice is
      auto-joined on first appearance and forwarded through the
      `SpeechHandler._on_speech` callback.
- `[<ID> joins]` / `[<ID> leaves]`:
      adds / removes the person from the "present" set. When at
      least two are present, a group is formed automatically and
      published on the GroupHandler; when fewer than two remain
      the group is dispersed.
- `# CHECK: <assertion>; <assertion>; ...`:
      one or more semicolon-separated invariants. Supported forms:
        * `X history contains N utterances`
        * `X is unknown`
        * `X is known but not active`
        * `X is active`
        * `X and Y are in the same group`
        * `"<word>" appears in X history`
        * `"<word>" does not appear in X history`
        * `{A,B,…} group history contains N utterances`
        * `"<word>" does not appear in {A,B,…} group history`
- `# <anything else>` is ignored.
"""

from pathlib import Path
import re
from unittest.mock import MagicMock

from dialogue_manager.conversations_history import ConversationsHistoryStore
from dialogue_manager.dialogue import (
    Dialogue,
    DialogueManager,
    DialogueState,
    Interlocutor,
)
from dialogue_manager.group_handler import GroupHandler
from dialogue_manager.skill_servers import SkillServers
from dialogue_manager.speech_handler import SpeechHandler
from hri_msgs.msg import Group, LiveSpeech
import pytest


DSL_DIR = Path(__file__).parent / 'dialogues'


def _empty_group_resolver(_gid: str) -> list[str]:
    return []


class _Scenario:
    """Run a single DSL scenario and accumulate assertion failures."""

    def __init__(self) -> None:
        node = self._make_node()
        self.dialogue_manager = DialogueManager()
        self.store = ConversationsHistoryStore()
        self.group_handler = GroupHandler(
            node=node,
            on_group_dispersed=self._on_group_dispersed,
        )
        self.speech_handler = SpeechHandler(
            node=node,
            dialogue_manager=self.dialogue_manager,
            chatbot_client=None,
            conversations_store=self.store,
            closed_captions_pub=MagicMock(),
            intents_pub=MagicMock(),
            group_handler=self.group_handler,
        )
        self.speech_handler.set_default_chat('__default__')
        self.skill_servers = SkillServers(
            node=node,
            dialogue_manager=self.dialogue_manager,
            chatbot_client=None,
            say_client=MagicMock(),
            conversations_store=self.store,
            group_resolver=self.group_handler.members_of,
            closed_captions_pub=MagicMock(),
            group_handler=self.group_handler,
        )

        self.present: set[str] = set()
        self._current_group_id: str | None = None
        self._line_no = 0
        self._failures: list[tuple[int, str, str]] = []

    @staticmethod
    def _make_node() -> MagicMock:
        node = MagicMock()
        node.get_clock.return_value.now.return_value.nanoseconds = int(1e9)
        return node

    # ----------------------------------------------- group dispersal hook

    def _on_group_dispersed(self, group_id: str) -> None:
        """Finalize and archive a dispersed group's dialogue (test-local)."""
        dialogue = self.dialogue_manager.get_dialogue_for_interlocutor(
            Interlocutor(group_id=group_id)
        )
        if dialogue is None:
            return
        dialogue.state = DialogueState.COMPLETED
        # NOTE: by the time this callback fires GroupHandler has already
        # removed the group from its map, so members_of() returns []. This
        # mirrors the production behaviour today (manager_node has the
        # same limitation) — fan-out into per-member personal histories
        # therefore doesn't happen on dispersal in the test either.
        self.store.archive(dialogue, group_members=None)
        self.dialogue_manager.remove_dialogue(dialogue.dialogue_id)

    # ----------------------------------------------- presence + groups

    def join(self, person_id: str) -> None:
        self.present.add(person_id)
        self._refresh_group()

    def leave(self, person_id: str) -> None:
        self.present.discard(person_id)
        self._refresh_group()

    def _refresh_group(self) -> None:
        members = sorted(self.present)
        new_id = 'group_' + '_'.join(members) if len(members) >= 2 else None
        if new_id == self._current_group_id:
            return
        if self._current_group_id is not None:
            self._publish_group(self._current_group_id, [])
        if new_id is not None:
            self._publish_group(new_id, members)
        self._current_group_id = new_id

    def _publish_group(self, group_id: str, members: list[str]) -> None:
        msg = Group()
        msg.group_id = group_id
        msg.members = list(members)
        self.group_handler._on_group(msg)

    # ----------------------------------------------- utterances

    def speak(self, speaker: str, text: str) -> None:
        if speaker == 'R':
            self.skill_servers._broadcast_say_utterance(text)
            return
        if speaker not in self.present:
            self.join(speaker)
        msg = LiveSpeech()
        msg.final = text
        msg.locale = 'en'
        msg.confidence = 1.0
        self.speech_handler._on_speech(speaker, msg)

    # ----------------------------------------------- assertions

    def check(self, text: str) -> None:
        try:
            self._run_assertion(text.strip())
        except AssertionError as exc:
            self._failures.append((self._line_no, text.strip(), str(exc)))
        except ValueError as exc:
            self._failures.append((self._line_no, text.strip(), str(exc)))

    def _run_assertion(self, text: str) -> None:
        m = re.fullmatch(r'(\w+) history contains (\d+) utterances?', text)
        if m:
            person, n = m.group(1), int(m.group(2))
            actual = self._count_history(Interlocutor(person_id=person))
            assert actual == n, f'expected {n}, got {actual}'
            return

        m = re.fullmatch(r'(\w+) is unknown', text)
        if m:
            person = m.group(1)
            actual = self._count_history(Interlocutor(person_id=person))
            assert actual == 0, f'{person} has {actual} utterance(s) on record'
            return

        m = re.fullmatch(r'(\w+) and (\w+) are in the same group', text)
        if m:
            a, b = m.group(1), m.group(2)
            shared = (
                set(self.group_handler.groups_containing(a))
                & set(self.group_handler.groups_containing(b))
            )
            assert shared, f'{a} and {b} share no group'
            return

        m = re.fullmatch(r'(\w+) is known but not active', text)
        if m:
            person = m.group(1)
            count = self._count_history(Interlocutor(person_id=person))
            assert count > 0, f'{person} is unknown (no history)'
            assert person not in self.present, \
                f'{person} is still present'
            return

        m = re.fullmatch(r'(\w+) is active', text)
        if m:
            person = m.group(1)
            assert person in self.present, f'{person} is not currently present'
            return

        m = re.fullmatch(r'"([^"]+)" appears in (\w+) history', text)
        if m:
            word, person = m.group(1), m.group(2)
            blob = self._history_text(Interlocutor(person_id=person))
            assert word in blob, f'{word!r} not in history of {person}'
            return

        m = re.fullmatch(
            r'"([^"]+)" does not appears? in (\w+) history', text
        )
        if m:
            word, person = m.group(1), m.group(2)
            blob = self._history_text(Interlocutor(person_id=person))
            assert word not in blob, \
                f'{word!r} unexpectedly present in history of {person}'
            return

        m = re.fullmatch(
            r'\{([^}]+)\} group history contains (\d+) utterances?', text
        )
        if m:
            members_str, n = m.group(1), int(m.group(2))
            group_id = self._group_id_for(members_str)
            actual = self._count_history(Interlocutor(group_id=group_id))
            assert actual == n, f'expected {n}, got {actual}'
            return

        m = re.fullmatch(
            r'"([^"]+)" does not appears? in \{([^}]+)\} group history',
            text,
        )
        if m:
            word, members_str = m.group(1), m.group(2)
            group_id = self._group_id_for(members_str)
            blob = self._history_text(Interlocutor(group_id=group_id))
            assert word not in blob, \
                f'{word!r} unexpectedly in group {group_id} history'
            return

        raise ValueError(f'unrecognised assertion: {text!r}')

    @staticmethod
    def _group_id_for(members_str: str) -> str:
        members = sorted(s.strip() for s in members_str.split(','))
        return 'group_' + '_'.join(members)

    def _all_dialogues_for(self, interlocutor: Interlocutor) -> list[Dialogue]:
        """Return every Dialogue (active or archived) matching `interlocutor`.

        "Matching" means the dialogue's *own* interlocutor equals
        `interlocutor` — we do not pick up group dialogues that happen
        to have been fanned out under a member's archive key, since
        the DSL's "A history" refers to A's own conversational record,
        not group conversations A took part in.
        """
        seen: set = set()
        result: list[Dialogue] = []
        for d in self.dialogue_manager.active_dialogues.values():
            if d.interlocutor == interlocutor and d.dialogue_id not in seen:
                seen.add(d.dialogue_id)
                result.append(d)
        for d in self.store.history_for(interlocutor):
            if d.interlocutor != interlocutor:
                continue
            if d.dialogue_id in seen:
                continue
            seen.add(d.dialogue_id)
            result.append(d)
        return result

    def _count_history(self, interlocutor: Interlocutor) -> int:
        return sum(
            len(d.session_utterances)
            for d in self._all_dialogues_for(interlocutor)
        )

    def _history_text(self, interlocutor: Interlocutor) -> str:
        return '\n'.join(
            u.text
            for d in self._all_dialogues_for(interlocutor)
            for u in d.session_utterances
        )

    # ----------------------------------------------- parser

    def run(self, source: str) -> None:
        lines = source.splitlines()
        i = 0
        while i < len(lines):
            self._line_no = i + 1
            line = lines[i]
            stripped = line.strip()

            if not stripped:
                i += 1
                continue

            if stripped.startswith('#'):
                comment = stripped.lstrip('#').strip()
                if comment.upper().startswith('CHECK:'):
                    rest = comment[len('CHECK:'):]
                    for piece in rest.split(';'):
                        if piece.strip():
                            self.check(piece)
                i += 1
                continue

            if stripped.startswith('['):
                m = re.fullmatch(r'\[(\w+) (joins|leaves)\]', stripped)
                if m is None:
                    raise ValueError(
                        f'line {self._line_no}: bad stage direction: '
                        f'{stripped!r}'
                    )
                pid, op = m.group(1), m.group(2)
                if op == 'joins':
                    self.join(pid)
                else:
                    self.leave(pid)
                i += 1
                continue

            m = re.fullmatch(r'(\w+):', stripped)
            if m is None:
                raise ValueError(
                    f'line {self._line_no}: unrecognised line: {line!r}'
                )
            speaker = m.group(1)
            i += 1
            # Collect indented '- <text>' lines.
            while i < len(lines):
                self._line_no = i + 1
                nxt = lines[i]
                nxt_stripped = nxt.strip()
                if not nxt_stripped:
                    i += 1
                    continue
                if not nxt_stripped.startswith('-'):
                    break
                text = nxt_stripped[1:].strip()
                self.speak(speaker, text)
                i += 1

        if self._failures:
            details = '\n'.join(
                f'  line {n}: {a!r} → {msg}' for n, a, msg in self._failures
            )
            pytest.fail(
                f'{len(self._failures)} DSL check failure(s):\n{details}',
                pytrace=False,
            )


def test_dialogue_1_dsl():
    """Execute test/dialogues/dialogue_1.md and verify every CHECK assertion."""
    source = (DSL_DIR / 'dialogue_1.md').read_text()
    _Scenario().run(source)
