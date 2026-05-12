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

"""Per-interlocutor conversations history with optional disk persistence."""

from collections.abc import Callable, Iterable
import json
import logging
from pathlib import Path
from uuid import UUID

from chatbot_msgs.msg import DialogueRole

from .dialogue import (
    Dialogue,
    DialogueState,
    Interlocutor,
    SESSION_BREAK_SPEAKER_ID,
    SUMMARY_SPEAKER_ID,
    Utterance,
)


ContextFilter = Callable[[Dialogue], bool]
"""Predicate returning True when the dialogue should be included in context."""


def exclude_ask(dialogue: Dialogue) -> bool:
    """Built-in filter: exclude ASK-role dialogues from the context."""
    return dialogue.role.name != DialogueRole.ASK_ROLE


EXCLUDE_ASK: ContextFilter = exclude_ask
"""Default filter list entry — excludes ASK-role dialogues from context."""


class ConversationsHistoryStore:
    """
    Persistent per-interlocutor history of past dialogues.

    Group dialogues are also fanned out into the personal history of each
    participating member when the group→members mapping is supplied.

    Parameters
    ----------
    storage_dir
        Directory under which `person/<id>.json` and `group/<id>.json` are
        written. If ``None``, the store is in-memory only.

    """

    def __init__(
        self,
        storage_dir: Path | None = None,
        logger: logging.Logger | None = None,
    ):
        """Initialize the store, optionally rooted at `storage_dir`."""
        self._storage_dir = (
            Path(storage_dir).expanduser() if storage_dir is not None else None
        )
        self._logger = logger or logging.getLogger(__name__)
        self._by_interlocutor: dict[str, list[Dialogue]] = {}

    @property
    def storage_dir(self) -> Path | None:
        """Return the on-disk storage directory, if any."""
        return self._storage_dir

    # =========================================================================
    # Archival
    # =========================================================================

    def archive(
        self,
        dialogue: Dialogue,
        group_members: Iterable[str] | None = None,
    ) -> None:
        """
        Append a finished dialogue to its interlocutor's history.

        For group dialogues, the same dialogue object is also appended to each
        provided member's personal history (as per DIALOGUE_FLOW.md).
        Empty dialogues (no utterances) and unbound interlocutors are skipped.
        """
        if not dialogue.interlocutor.is_bound:
            self._logger.debug(
                'Skipping archival: dialogue %s has no interlocutor',
                dialogue.dialogue_id,
            )
            return
        if not dialogue.session_utterances:
            self._logger.debug(
                'Skipping archival: dialogue %s has no session utterances',
                dialogue.dialogue_id,
            )
            return

        self._append(dialogue.interlocutor, dialogue)

        if dialogue.interlocutor.is_group and group_members:
            for member_id in group_members:
                if not member_id:
                    continue
                self._append(Interlocutor(person_id=member_id), dialogue)

    def _append(self, interlocutor: Interlocutor, dialogue: Dialogue) -> None:
        bucket = self._by_interlocutor.setdefault(interlocutor.key, [])
        # Idempotent: if this Dialogue is already tracked under the key
        # (typical when periodic snapshotting re-archives an
        # in-progress dialogue), don't duplicate it. The latest history
        # is on the same Dialogue object — bucket entries are
        # references, not copies — so subsequent save() reflects new
        # utterances without us having to re-insert.
        if any(d.dialogue_id == dialogue.dialogue_id for d in bucket):
            return
        bucket.append(dialogue)

    def history_for(self, interlocutor: Interlocutor) -> list[Dialogue]:
        """Return a copy of the past dialogues for an interlocutor."""
        return list(self._by_interlocutor.get(interlocutor.key, []))

    def preload_into(self, dialogue: Dialogue, now: float) -> bool:
        """Pre-fill `dialogue.history` with the most recent prior summary.

        For a known interlocutor with at least one archived prior dialogue
        carrying a non-empty summary, prepend a SUMMARY utterance followed by
        a SESSION_BREAK marker and advance `session_start_index` so these
        pre-fill entries are excluded from archival.

        Returns True if a summary was pre-filled, False otherwise.
        """
        if not dialogue.interlocutor.is_bound:
            return False
        prior = self.history_for(dialogue.interlocutor)
        if not prior:
            return False
        latest = max(prior, key=lambda d: d.ended_at or d.started_at or 0.0)
        if not latest.summary:
            return False

        timestamp = latest.ended_at or latest.started_at or now
        dialogue.history.append(Utterance(
            timestamp=timestamp,
            speaker_id=SUMMARY_SPEAKER_ID,
            text=latest.summary,
        ))
        dialogue.history.append(Utterance(
            timestamp=now,
            speaker_id=SESSION_BREAK_SPEAKER_ID,
            text='',
        ))
        dialogue.session_start_index = len(dialogue.history)
        return True

    # =========================================================================
    # Context construction
    # =========================================================================

    def build_context(
        self,
        interlocutor: Interlocutor,
        now: float,
        filters: list[ContextFilter] | None = None,
        summarize: Callable[[Dialogue], str] | None = None,
        summary_age_sec: float = 300.0,
    ) -> str:
        """
        Build the conversation context string to inject into the LLM backend.

        Each archived dialogue is rendered after applying every predicate in
        `filters` (a dialogue is kept only if every filter returns True). The
        default filter list excludes ASK-role dialogues.

        For dialogues older than `summary_age_sec`, a cached summary is used if
        present. Otherwise, when `summarize` is supplied, it is called once and
        the result cached on the dialogue (the cache is invalidated whenever a
        new utterance is appended).
        """
        if filters is None:
            filters = [EXCLUDE_ASK]

        chunks: list[str] = []
        for dialogue in self.history_for(interlocutor):
            if not all(f(dialogue) for f in filters):
                continue

            reference_ts = (
                dialogue.ended_at
                if dialogue.ended_at is not None
                else dialogue.started_at if dialogue.started_at is not None
                else now
            )
            age = now - reference_ts
            if age > summary_age_sec:
                if dialogue.summary is None and summarize is not None:
                    try:
                        dialogue.summary = summarize(dialogue)
                        dialogue.summary_generated_at = now
                    except Exception as exc:
                        self._logger.warning(
                            'Summarizer failed for dialogue %s: %s',
                            dialogue.dialogue_id, exc,
                        )
                if dialogue.summary:
                    chunks.append(f'[summary] {dialogue.summary}')
                    continue

            chunks.append(self._render_dialogue(dialogue))

        return '\n\n'.join(chunks)

    @staticmethod
    def _render_dialogue(dialogue: Dialogue) -> str:
        lines = [f'[dialogue role={dialogue.role.name}]']
        for utt in dialogue.history:
            lines.append(f'  {utt.speaker_id}: {utt.text}')
        return '\n'.join(lines)

    # =========================================================================
    # Persistence
    # =========================================================================

    def load(self) -> None:
        """Load all per-interlocutor JSON files from `storage_dir`."""
        if self._storage_dir is None or not self._storage_dir.exists():
            return
        for sub in ('person', 'group'):
            d = self._storage_dir / sub
            if not d.is_dir():
                continue
            for path in sorted(d.glob('*.json')):
                try:
                    raw = json.loads(path.read_text())
                except Exception as exc:
                    self._logger.warning('Failed to load %s: %s', path, exc)
                    continue
                key_id = path.stem
                interlocutor = (
                    Interlocutor(person_id=key_id) if sub == 'person'
                    else Interlocutor(group_id=key_id)
                )
                self._by_interlocutor[interlocutor.key] = [
                    self._dialogue_from_dict(d) for d in raw
                ]

    def save(self) -> None:
        """Persist all in-memory histories to `storage_dir`."""
        if self._storage_dir is None:
            return
        (self._storage_dir / 'person').mkdir(parents=True, exist_ok=True)
        (self._storage_dir / 'group').mkdir(parents=True, exist_ok=True)

        for key, bucket in self._by_interlocutor.items():
            kind, _, key_id = key.partition(':')
            if kind not in ('person', 'group') or not key_id:
                continue
            path = self._storage_dir / kind / f'{key_id}.json'
            try:
                path.write_text(json.dumps(
                    [self._dialogue_to_dict(d) for d in bucket],
                    indent=2,
                    ensure_ascii=False,
                ))
            except Exception as exc:
                self._logger.warning('Failed to save %s: %s', path, exc)

    # =========================================================================
    # Serialization
    # =========================================================================

    @staticmethod
    def _dialogue_to_dict(d: Dialogue) -> dict:
        return {
            'dialogue_id': str(d.dialogue_id),
            'role': {
                'name': d.role.name,
                'configuration': d.role.configuration,
            },
            'interlocutor': {
                'person_id': d.interlocutor.person_id,
                'group_id': d.interlocutor.group_id,
            },
            'priority': d.priority,
            'started_at': d.started_at,
            'ended_at': d.ended_at,
            # Only persist the actual session's utterances. Pre-filled summary
            # entries and session-break markers (before session_start_index)
            # are reconstructed at load time from the prior dialogue's stored
            # summary, so we must not serialize them here.
            'history': [
                {
                    'timestamp': u.timestamp,
                    'speaker_id': u.speaker_id,
                    'text': u.text,
                }
                for u in d.session_utterances
            ],
            'summary': d.summary,
            'summary_generated_at': d.summary_generated_at,
        }

    @staticmethod
    def _dialogue_from_dict(data: dict) -> Dialogue:
        role = DialogueRole(
            name=data['role']['name'],
            configuration=data['role'].get('configuration', '{}'),
        )
        interlocutor = Interlocutor(
            person_id=data['interlocutor'].get('person_id', ''),
            group_id=data['interlocutor'].get('group_id', ''),
        )
        return Dialogue(
            role=role,
            interlocutor=interlocutor,
            priority=data.get('priority', 128),
            state=DialogueState.COMPLETED,
            dialogue_id=UUID(data['dialogue_id']),
            started_at=data.get('started_at'),
            ended_at=data.get('ended_at'),
            history=[
                Utterance(
                    timestamp=u['timestamp'],
                    speaker_id=u['speaker_id'],
                    text=u['text'],
                )
                for u in data.get('history', [])
            ],
            summary=data.get('summary'),
            summary_generated_at=data.get('summary_generated_at'),
        )
