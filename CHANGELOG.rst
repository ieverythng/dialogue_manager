^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package dialogue_manager
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Forthcoming
-----------
* typing: replace typing.Optional with PEP 604 X | None
  Drop the typing.Optional / typing.Dict / typing.Set imports across
  chatbot_client, tts_client and speech_handler; use the PEP 604 union
  syntax and the lower-case builtin generics.
  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
* dialogue: fix init bug, docstring style, and Optional in skill_servers
  Found while running the test suite for the first time:
  - four init lines (_chat_server, _ask_server, _say_server, _is_active)
  had been orphaned inside _inject_initial_context; move them back
  into __init_\_ so SkillServers() actually initializes them
  - pep257: docstrings on Interlocutor.is_group / is_bound now use
  imperative mood ('Return True ...')
  - pep257: trailing blank line on the Parameters block in
  ConversationsHistoryStore docstring
  - modernize the one Optional[...] annotation in skill_servers to
  the | None syntax
  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
* test: document new conversations-history tests in test README
  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
* docs: cover conversations history in README, add TODO.md
  Brief subsection in README pointing to DIALOGUE_FLOW.md for the model,
  plus the new conversations_storage_dir parameter.
  TODO.md tracks the two follow-ups left open by the conversations-history
  work: per-turn context delivery (currently a one-shot __system_\_ prime)
  and pyhri group support (currently a stub, blocking member fan-out).
  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
* dialogue: track per-interlocutor conversations history
  Brings the implementation in line with doc/DIALOGUE_FLOW.md:
  - model: replace person_id/group_id pair with Interlocutor union; add
  Utterance, ROBOT_SPEAKER_ID, lazy started_at on first utterance,
  ended_at on completion, in-dialogue history list, and a cached
  summary slot
  - store: new ConversationsHistoryStore archives finished dialogues
  per-person/per-group with member fan-out, builds the LLM context
  with a pluggable filter pipeline (default excludes ASK), and caches
  summaries on disk so the LLM is not re-queried per turn; JSON files
  under ~/.ros/dialogue_manager/conversations
  - recording: chatbot_client appends user and robot utterances to the
  active dialogue; skill_servers archives Chat/Ask on completion and
  records Say utterances against the targeted interlocutor (active
  dialogue or one-shot __say_\_ synthetic)
  - context delivery: at dialogue start, push the built context to the
  chatbot as a __system_\_ DialogueInteraction (no response). A more
  durable mechanism is captured for follow-up
  - node: load history on configure, persist on shutdown; group->members
  resolver is a stub until pyhri exposes groups
  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
* [minor] update typing syntax for optionals
* enable a default dialogue by default
* Contributors: Séverin Lemaignan

0.4.0 (2026-04-02)
------------------
* markup: properly handle nested ROS2 fields
* resolve markup action list relative to package share/ folder
* remove PAL's module
* document and implement markup processing
* Contributors: Séverin Lemaignan

0.3.1 (2026-02-10)
------------------
* mark pkg architecture_independent
* test: make sure integration tests properly shutdown
* extend tests to chatbot interactions
* add integration tests
* Contributors: Séverin Lemaignan

0.3.0 (2026-02-09)
------------------
* add tests
* linting
* fix dialogue ID. First interactions with chatbot_llm are now working
* refactoring -- splitting manager_core.py into smaller files
* Contributors: Séverin Lemaignan

0.2.0 (2026-02-05)
------------------
* initial implementation, done by Claude Opus 4.5
* initial scaffolding, using rpk
* Contributors: Séverin Lemaignan
