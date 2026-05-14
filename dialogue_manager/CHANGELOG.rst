^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package dialogue_manager
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

2.2.0 (2026-05-14)
------------------
* Inject __system_\_ membership note into group dialogues at spawn
  When a group dialogue is spawned (auto-spawn in speech_handler or
  explicit Chat goal with group_id in skill_servers), prepend a
  __system_\_ utterance listing the other participants — e.g.
  "You are now in a group conversation with: alice, bob, carol." —
  so the LLM has explicit context for multi-party turns instead of
  having to infer membership from interleaved user_ids.
  Single snapshot at spawn time; mid-session membership churn does
  not currently re-emit. That would belong to the active-dialogue
  tracker work that's still outstanding.
  Skipped for groups of <=1 (degenerate; not a real group).
  DSL test helpers (_count_history, _history_text) now skip
  SYSTEM_SPEAKER_ID utterances when counting "spoken" content —
  the DSL is asserting fan-out of user/robot speech, not internal
  metadata events.
  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
* manager_node: expose ~/{get,set}_logger_levels services
  Pass enable_logger_service=True to the LifecycleNode constructor so
  log levels can be flipped at runtime via the standard rclpy logger
  services. Lets callers turn the DEBUG dump in chatbot_client.interact
  on/off without restarting the node.
  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
* chatbot_client: DEBUG-level dump of outbound history + summary
  Adds a verbose dump of the exact Utterance[] history and prior-session
  summary about to ship in a dialogue_interaction call. Gated at DEBUG
  so a long conversation doesn't bloat normal runs; enable with
  `--ros-args --log-level dialogue_manager:=debug`.
  Useful for diagnosing "the LLM didn't see my Say utterance" type
  failures: pair this with the matching dump on the chatbot_llm side
  to see (a) what dialogue_manager shipped and (b) what chatbot_llm
  forwarded to the LLM after its system-prompt rendering.
  _debug_enabled() wraps the LoggingSeverity comparison so the
  MagicMock-based unit tests don't trip on the type check.
  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
* Port to stateless chatbot_msgs v4 contract
  The chatbot backend is no longer a long-lived action goal; it is a
  stateless service that receives the full dialogue history on every
  turn. This commit ports the dialogue_manager side to the new
  contract.
  - chatbot_client: complete rewrite.
  * Replace the start_dialogue action handle with a fire-and-forget
  `prepare(dialogue)` call against the optional PrepareDialogue
  service.
  * Replace per-event send_input / inject_context / echo plumbing
  with a single `interact(dialogue)` method that snapshots
  `dialogue.session_utterances` into a chatbot_msgs/Utterance[]
  and ships it (alongside the role and the prior-session summary
  pulled from the conversations store) as one DialogueInteraction
  call. Speaker mapping: ROBOT_SPEAKER_ID -> Utterance.ASSISTANT,
  SYSTEM_SPEAKER_ID -> Utterance.SYSTEM, anything else verbatim.
  * On response, record the robot's utterance into dialogue.history
  (so the next turn carries it), publish intents, speak, and
  propagate the new `dialogue_terminal` flag (with `results`) by
  transitioning the dialogue to COMPLETED.
  - dialogue: drop `chatbot_goal_id` (no separate handle anymore). Add
  SYSTEM_SPEAKER_ID and a `results: str` field used by ChatbotClient
  to surface role-driven terminal results back to skill execution
  coroutines.
  - speech_handler: drop the pending-attach queue and the
  attach_to_dialogue handshake — the chatbot has no per-dialogue
  state to attach to. Forwarding becomes one line:
  `chatbot_client.interact(speaker_dialogue)`. Spawn paths call
  `prepare()` instead.
  - skill_servers: Chat / Ask no longer await an action goal handle.
  Ask waits on `dialogue.state == COMPLETED` + `dialogue.results`
  (set by the chatbot's dialogue_terminal response). Chat with
  `initiate=true` and no `initial_input` appends a __system\_\_
  directive ("greet user X") to the history and calls interact() —
  the chatbot sees the directive in the stream and produces the
  opener. Drop _inject_initial_context (now folded into
  DialogueInteraction.summary).
  - debug_publisher: replace `chatbot_goal_id` field with `results`
  in the JSON snapshot.
  - tests: rewrite test_chatbot_client around prepare / interact /
  _on_response. Update test_dialogue and test_speech_handler for
  the new chatbot_goal_id-less Dialogue. Rework MockChatbotNode in
  test_integration to expose the stateless services.
  Pairs with the chatbot_msgs 4.0.0 contract change and the chatbot_llm
  stateless port.
  NOTE: the active-dialogue tracker + group-aware fan-out simplification
  discussed in the design conversation are NOT in this commit. The
  existing speech_handler fan-out (one utterance into multiple per-person
  + per-group dialogues) is preserved here, just adapted to call
  interact() instead of send_input(). The single-active-dialogue
  invariant is a follow-up.
  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
* Contributors: Séverin Lemaignan

2.1.0 (2026-05-13)
------------------
* chatbot: queue first-utterance speech across async attach
  attach_to_dialogue() dispatches the chatbot goal asynchronously, so on
  the first utterance from a new speaker chatbot_goal_id was still None
  when SpeechHandler checked it — the speech fell through to
  RAW_USER_INPUT and the chatbot never saw the first user input. The
  second utterance worked because by then the goal had been accepted.
  SpeechHandler now keeps a per-dialogue pending queue (lock-guarded
  because callbacks run on the MultiThreadedExecutor + ReentrantCallbackGroup).
  Utterances received while attach is in flight are queued; on acceptance
  they are flushed via send_input(). On rejection the queue is
  republished as RAW_USER_INPUT and an error fires — rejection should not
  happen under normal operation.
  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
* Contributors: Séverin Lemaignan

2.0.0 (2026-05-13)
------------------

Major changes:

This release introduces a significant refactor of the dialogue_manager's
internal architecture, with a focus on supporting multi-party dialogues.

Independent conversations are now tracked for each person or group.

Detailed changes:

* style: make the linter tests pass
  The ament_pep257 / ament_flake8 / ament_copyright tests were red on
  this branch (predating any of the recent refactor work). Bringing
  them green without any \`# noqa\`:
  * **D213 (multi-line docstring summary on second line)** — convention
  the codebase already used in most places (see e.g. \`Dialogue\` and
  \`DialogueState\` docstrings). 30+ stragglers across both packages
  rewritten to match. No semantic change to the docs themselves.
  * **D400 / D415 (first line ends with a period)** — \`fanout.py\`
  module docstring's first line ended with \`."\` because the closing
  quote was inside the prose. Reworded.
  * **D401 (imperative mood)** — \`is_dialogue_present\` and
  \`_group_realised\` started with "True if..."; lead with "Return".
  * **I101 (import order)** — \`test_dialogue_lifecycle.py\` imports
  reordered to satisfy flake8-import-order.
  * **Apache license tail** — \`test_dialogue_dsl.py\`,
  \`test_group_handler.py\`, and 8 files in rqt_dialogues had the
  truncated 4-line header instead of the full 13-line one.
  ament_copyright treats short headers as license=<unknown>; restored.
  376 tests pass (dialogue_manager: 359, rqt_dialogues: 17) including
  all linter checks. No new noqa markers introduced.
* fanout: extract interlocutor graph walk into a shared helper
  SpeechHandler._recipient_dialogues_for_speaker and
  SkillServers._active_recipients_for both walked the same
  interlocutor relationship graph — primary → groups containing the
  person (or members of the group) → co-members — with subtle
  variations (auto-spawn vs fetch-only, what to do with absent
  persons). The graph traversal itself was duplicated. The variations
  weren't.
  New module \`fanout.py\` carries just the shared bit: a
  \`related_interlocutors(interlocutor, group_handler)\` iterator that
  yields the related interlocutors in canonical order, deduplicated.
  Each caller decides what to do with each yielded interlocutor (spawn
  or fetch, presence-filter or not). The broadcast Say path is
  structurally different ("every active dialogue" rather than "fan-out
  from one interlocutor") and stays as-is.
  Side cleanup: \`is_dialogue_present(dialogue, presence_query)\`
  moves alongside it — the "groups always present; persons defer to
  the query" rule is the same in both call sites, so it lives next to
  the traversal helper. \`SkillServers._is_present\` is now a one-line
  delegation.
  The SpeechHandler refactor also unifies the per-kind spawn dispatch
  through a new \`_get_or_spawn_dialogue_for(interlocutor)\` so the
  fan-out loop doesn't need to branch on group vs person.
  New test_fanout.py covers the helper directly (primary-only,
  person/group cases, dedup, empty member skip, presence rules). All
  356 functional tests pass.
* lifecycle: extract DialogueLifecycle from SkillServers
  The end-of-session sequence — set state to COMPLETED, archive into
  the conversations store, kick off async summarization — lived as a
  private method on SkillServers (\`_finalize_and_archive\`). But it's
  not skill-specific: \`manager_node._on_group_dispersed\` and
  \`on_deactivate\` were already reaching across the abstraction
  boundary to call it. That's the smell that motivates this lift.
  Move the lifecycle endgame into a new \`dialogue_lifecycle.py\`
  module:
  * \`DialogueLifecycle.finalize_and_archive(dialogue)\` is now the
  public entry point. Same behaviour as before — state COMPLETED,
  ended_at stamped, archive + member fan-out, async summarizer in a
  daemon thread.
  * The \`Summarizer\` type alias and \`default_summarizer\` move with
  it; their natural home is next to the helper that drives them.
  * SkillServers accepts an optional \`lifecycle\` parameter. When not
  supplied (e.g. in unit tests that exercise skills in isolation), it
  builds a local instance with the same dependencies it already had.
  In production, manager_node constructs one DialogueLifecycle and
  shares it across SkillServers and its own direct callers.
  * manager_node._on_group_dispersed and on_deactivate now invoke
  \`self._lifecycle.finalize_and_archive(...)\` instead of reaching
  into \`_skill_servers._finalize_and_archive\`.
  Tests:
  * New \`test/test_dialogue_lifecycle.py\` covers the lifecycle in
  isolation (summarizer fan-out, default fallback, group member
  archive fan-out, empty-session no-op) — moved from
  test_skill_servers.py where they were testing through a layer
  that didn't own the logic.
  * The remaining skill_servers tests no longer accept a
  \`summarizer\` kwarg in their \`_build_servers\` helper, since
  SkillServers no longer carries that concern.
  347 functional tests pass.
* skills: drop GroupResolver callable; query GroupHandler directly
  manager_node was passing both `group_resolver=self._resolve_group\_
  members` (a wrapper around `group_handler.members_of`) and the
  GroupHandler itself into SkillServers — the resolver carried no
  additional behaviour. Drop the callable parameter and the
  GroupResolver type alias; SkillServers._resolve_group_members now
  calls self._group_handler.members_of directly (returning [] when no
  handler is configured).
  Tests that previously passed a no-op `_empty_group_resolver` simply
  drop the kwarg; the two fan-out tests that supplied a lambda
  resolver now build a MagicMock GroupHandler via a small
  `_group_handler_with_members({...})` helper. The DSL test
  scaffolding loses its `group_resolver=` line for the same reason.
  Pure plumbing simplification — no behaviour change; all 345
  functional tests pass.
* dialogue: drop Dialogue.interlocutor_present, query presence directly
  The per-person presence flag was derived state that mirrored
  SpeechHandler._tracked_voices. It cost us a mutable field on the
  Dialogue data class, a _set_person_presence side-channel, a dead
  notify_change roundtrip (the flag never surfaced in the debug
  snapshot), and a latent inconsistency: any dialogue spawned for a
  person whose voice wasn't currently tracked defaulted to "present
  = True", which was wrong.
  Replace the flag with a single source of truth:
  * SpeechHandler.is_voice_tracked(voice_id) becomes the canonical
  "is this person here right now" query.
  * SkillServers accepts an optional presence_query: Callable[[str], bool]
  parameter (same pattern as the existing group_resolver). manager_node
  wires it to speech_handler.is_voice_tracked. Tests that don't need
  presence filtering rely on the default ("always present").
  * _is_present(dialogue) helper short-circuits to True for group
  dialogues — group "absence" is signalled by dispersal, not by
  presence — and consults the query for person dialogues.
  * SpeechHandler.\_recipient_dialogues_for_speaker now checks
  is_voice_tracked *before* spawning a co-member dialogue, so we no
  longer materialise dialogues for people who aren't actually
  present (the pre-refactor latent bug).
  Behavioural tests added in the previous commit continue to pass;
  two old tests that asserted on the flag are restated against
  is_voice_tracked, and the group-routing tests now pre-track the
  relevant voices in setup_method (matching the real ROS4HRI flow).
* test: pin fan-out filtering behaviour for absent voices
  Adds direct unit tests for three previously DSL-only paths:
  * SpeechHandler co-member fan-out: an untracked co-member's
  dialogue does not collect utterances from a speaker that the
  group_handler still treats as a co-member.
  * SkillServers broadcast Say: an absent person dialogue is
  skipped; group dialogues remain included.
  * SkillServers addressed Say: an absent co-member is skipped from
  the per-person fan-out while the group dialogue still collects
  the utterance.
  These pin the behaviour we want to preserve while replacing the
  \`Dialogue.interlocutor_present\` flag with a presence-query
  abstraction in the next commit.
* dialogue: per-person presence flag + multi-party DSL stress tests
  Bundles together the group-aware bookkeeping work plus the test
  scaffolding that drove it:
  * `Dialogue.interlocutor_present` (default True) marks a person
  dialogue as "paused" when their voice leaves the tracked set; flips
  back to True on rejoin. Broadcast Says, co-member fan-out, and Say
  recipient resolution all filter on this flag so absent participants
  no longer pick up utterances they couldn't witness. The dialogue's
  history is preserved across the gap (`DialogueState` stays ACTIVE —
  the two notions are orthogonal).
  * `SpeechHandler._on_voices_tracked` now drives the flag: each voice
  added/removed propagates `_set_person_presence(voice_id, True/False)`
  and notifies the debug publisher.
  * `GroupHandler` dispersal callback fires *before* the group is
  removed from the membership map, so callees (manager_node's
  `_on_group_dispersed`, the DSL test runner) can resolve
  `members_of(group_id)` and drive proper per-member archive fan-out.
  * `ConversationsHistoryStore._append` is idempotent and `archive` no
  longer materializes subset entries — group history queries use
  *query-time* superset matching (a 40-person group must not produce
  2^40 archive entries). The principled rule: a group dialogue's
  identity is its exact member set, but its history *view* includes
  any past dialogue whose members were a superset.
  DSL test scaffolding:
  * `_Scenario` now publishes `/humans/voices/tracked` on every join /
  leave so the SpeechHandler's presence logic gets exercised.
  * The `{X,Y} group history` family of assertions is gated on the
  group having actually been realised — querying history for a group
  that never formed now fails with a clear pointer to
  `{X,Y} group does not exist` instead.
  * New `{X,Y} group does not exist` assertion validates that no
  dialogue with exactly those members has ever been spawned.
  * Group queries use superset matching across active + archived
  dialogues; person queries stay exact-match (their speech-time
  co-member fan-out already captures their experience).
  Three new scenarios on top of the original dialogue_1.md:
  * `dialogue_2.md`: {A,B,C} → A leaves → {B,C} continues → A rejoins,
  exercising paused-dialogue history preservation and subgroup
  history aggregation.
  * `dialogue_3.md`: lazy spawn semantics — groups exist in the
  GroupHandler before any dialogue is realised; history can only be
  queried after the first utterance.
  * `dialogue_4.md`: 4-person cascade {A,B,C,D} → {B,C,D} → {C,D} →
  solo D → {A,D}, with 40+ CHECK assertions verifying history
  accumulation, broadcast filtering during absence, and the brand-new
  {A,D} pair forming after A's return.
  Plus two unit tests for the new `_set_person_presence` path and TODO
  entries reframed for the lazy-spawn group-history model (LLM context
  preload for subgroups, canonical member-set group_ids in production).
  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
* test: add DSL-driven multi-party dialogue scenario
  A small markdown DSL lets us describe a multi-party conversation as a
  sequence of speaker turns + stage directions, with inline `# CHECK:`
  assertions that verify dialogue-manager state at that point. Example:
  A:
  - hello!
  R:
  - hi there!
  # CHECK: A history contains 2 utterances; B is unknown
  [B joins]
  # CHECK: A and B are in the same group
  A:
  - Hi B!
  # CHECK: {A,B} group history contains 1 utterance
  The new `test_dialogue_dsl` test parses dialogue_1.md and runs the
  scenario through the real DialogueManager / SpeechHandler / SkillServers
  / GroupHandler stack. Stage directions translate to `_on_voices_tracked`
  (presence) + Group message publishes (so the GroupHandler's auto-spawn,
  co-member fan-out, and dispersal logic all get exercised). Robot turns
  go through `SkillServers._broadcast_say_utterance` (the unaddressed Say
  path). Failures from individual assertions are aggregated and reported
  together so the test surfaces every gap at once instead of stopping at
  the first.
  The scenario currently surfaces 3 failures, all pointing at the same
  unimplemented behaviour: when a person leaves (their voice is no
  longer tracked) their dialogue keeps receiving broadcast Says and
  co-member fan-out, so "movie" leaks into A's history after `[A leaves]`
  and A's utterance count drifts. Useful as a regression target for the
  upcoming "deactivate-on-leave / reactivate-on-rejoin" fix.
  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
* persistence: snapshot active dialogues so periodic save actually writes
  `ConversationsHistoryStore._by_interlocutor` only gets populated when a
  Dialogue is finalised (`_finalize_and_archive`), so long-lived active
  dialogues — like the `__default_\_` ones auto-spawned per speaker — never
  reached disk between activate and deactivate. The periodic save() ran
  every 30s but had nothing to write.
  Two changes fix this:
  1. `ConversationsHistoryStore._append` is now idempotent — appending the
  same Dialogue twice is a no-op. Bucket entries are references to the
  live Dialogue object, so once a dialogue is registered, its history
  keeps growing automatically and subsequent save()s pick up the
  updated state without re-insertion.
  2. `manager_node._persist_conversations` now snapshots every active
  bound dialogue into the store before save(). Each tick:
  - walks `dialogue_manager.active_dialogues`
  - calls `conversations_store.archive(d, group_members=...)` on
  each one with a bound interlocutor and at least one session
  utterance (idempotent on repeats; resolved with members for
  groups so the snapshot also fans out into per-member files)
  - then save() writes everything
  A node running with two default-chat speakers now produces
  `~/.ros/dialogue_manager/conversations/person/<voice>.json` within the
  first persist interval, growing on each tick.
  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
* docs: rewrite DIALOGUE_FLOW + README for the current flow
  DIALOGUE_FLOW.md is restructured with a high-level overview first then
  the details in a logical sequence: At a glance → Core concepts (Dialogue,
  Interlocutor, Role) → How dialogues come into existence (proactive vs
  reactive auto-spawn) → Recording fan-out table (speech & Say, addressed
  & broadcast) → Dialogue lifetime + async summariser → Conversations
  history + pre-fill → Priorities (dialogue vs expression split) → Skills
  (Chat/Ask/Say, with chatbot-less variants) → Introspection (debug topic
  + rqt_dialogues).
  README updated to match: new Mermaid diagram covering the groups
  subscription and debug topic; `say_action` parameter added; topic
  table now includes `/humans/interactions/groups` and `~/debug_state`;
  action client switched from `tts_engine/tts` to the Say sub-skill
  endpoint; priority handling text rewritten to the dialogue-vs-
  expression split; new "Live introspection" section pointing at
  `rqt_dialogues`.
  TODO.md trims the two items the recent work made obsolete (pyhri
  groups resolver, default-chat interlocutor binding — both now real)
  and notes the LLM-backed summariser as the natural next step.
  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
* dialogue: track ROS4HRI groups and fan out utterances to all members
  New GroupHandler subscribes to /humans/interactions/groups (TRANSIENT\_
  LOCAL latched topic). Each hri_msgs/Group message updates the group →
  members map; an empty-members message signals dispersal. Exposes
  members_of(group_id), groups_containing(person_id), co_members_of(
  person_id). On dispersal, the manager_node finalises and archives the
  group's dialogue (running the async summariser).
  SpeechHandler now records every utterance into the speaker's dialogue
  *and* the dialogue of every group containing the speaker *and* every
  co-member's per-person dialogue — auto-spawning missing ones when
  default-chat is enabled. Group dialogues are observation-only: no
  chatbot is attached to them (chatbots are per-person partners).
  `_record_say_utterance` fans out the same way for explicitly addressed
  Say goals:
  - Say with group_id → group + each member's dialogue.
  - Say with person_id → person + groups they belong to + co-members.
  No active dialogues at all still falls back to a synthetic __say\_\_
  dialogue archived directly to disk.
  `_resolve_group_members` (previously a stub) now consults the
  GroupHandler, so archived group dialogues fan out into each member's
  on-disk history per ConversationsHistoryStore.archive().
  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
* say: broadcast unaddressed utterances into every active dialogue
  A Say goal with no person_id/group_id was previously not recorded in
  any dialogue history — the robot spoke but the utterance left no trace.
  That meant /skill/say-style broadcast announcements were invisible to
  the introspection plugin and to per-person history persistence.
  Now an unaddressed Say lands a ROBOT_SPEAKER_ID utterance into every
  active dialogue whose interlocutor is bound (person/group). Unbound
  dialogues are skipped; if no bound active dialogue exists the call is
  a no-op. Addressed Says keep their existing per-interlocutor behaviour.
  Extracted `_strip_markup` so both the addressed and broadcast paths
  share the same markup-stripping step.
  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
* dialogue: split priority gating for dialogues vs expressions
  Previously a single `can_accept_priority` used `p > max(dialogue ∪
  expression)` for all three skills. Two consequences:
  - Any active dialogue (even a passive priority-0 default chat) added
  its priority to the rollup, so a Say at priority N was rejected
  whenever any dialogue at priority ≥ N existed — regardless of
  whether anything was actually speaking.
  - Strict `>` rejected equal-priority goals, which is wrong for Say:
  Say is a cooperative one-shot, not a preempting long-lived dialogue.
  Split into two checks matching what each skill actually competes against:
  - Chat / Ask use `can_start_dialogue(p)` — `p > max active dialogue
  priority`. Preemption: a new dialogue must be strictly higher than
  any existing one. Expression priority is irrelevant.
  - Say uses `can_speak(p)` — `p >= current expression priority`.
  Cooperative: a Say speaks if nothing higher is currently coming out
  of the speaker. Dialogue priority is irrelevant.
  `current_max_priority` is kept as a single-number rollup for the debug
  snapshot / diagnostics.
  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
* dialogue: spawn one __default_\_ dialogue per speaker
  Previously a single shared default Dialogue (chatbot-backed or passive)
  swallowed every speaker's utterance, so person A and person B landed in
  the same history.
  Now each new speaker triggers a fresh per-person Dialogue on their first
  utterance: role = `default_chat_role`, interlocutor.person_id = voice_id,
  prior summary pre-filled from disk when available. Subsequent utterances
  from the same speaker reuse that dialogue.
  The chatbot path is unified: ChatbotClient gains `attach_to_dialogue` —
  sends start_dialogue async and sets `chatbot_goal_id` on acceptance.
  SpeechHandler calls it whenever a chatbot is configured. Speech that
  arrives before the attach completes is recorded but not forwarded (a
  small first-utterance race that's an acceptable cost).
  Drops the shared `DialogueManager.default_dialogue_id` concept and the
  `ChatbotClient.start_default_chat` entry point. `manager_node.on_activate`
  just enables auto-spawn on the SpeechHandler. Preload logic moves into
  `ConversationsHistoryStore.preload_into` so both SkillServers and
  SpeechHandler can share it.
  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
* introspection: add debug-state topic and rqt_dialogues plugin
  Two related changes that make the dialogue_manager's internal state
  visible at runtime:
  1. New `~/debug_state` topic (std_msgs/String JSON, latched
  TRANSIENT_LOCAL) on dialogue_manager. Published on-change only —
  triggered by add/remove/utterance/state-change hooks fed through
  DialogueManager.set_change_callback. Snapshot includes chatbot
  status, active dialogues (with full history), archived dialogues
  grouped by interlocutor, and per-dialogue `last_updated_at` so
  tools can highlight the box that just changed.
  2. New rqt_dialogues ROS package. Two-pane Qt UI subscribed to the
  debug topic: left tree lists Active dialogues plus an Archived
  section grouped by interlocutor; right pane shows the selected
  dialogue's role/state/priority and a scrollable utterance history
  (pre-session entries dimmed). Updated boxes briefly flash warm
  yellow. Pure rendering is factored into render.py so it's
  testable without a display.
  Drive-by: when `enable_default_chat=True` but no chatbot is configured
  or reachable, manager_node now creates a chatbot-less passive default
  Dialogue (formerly the path silently warned and left the manager with
  no default dialogue, so chatbot-less speech never landed in any
  dialogue and the debug topic stayed silent). Default-dialogue ownership
  moves from ChatbotClient to DialogueManager so SpeechHandler can route
  to it regardless of chatbot presence.
  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
* repo: restructure into multi-package monorepo
  Move the existing `dialogue_manager` ROS package into a subdirectory of
  the same name so additional packages (rqt_dialogues, ...) can sit
  alongside it under the same repo. No code changes — strictly a
  file-system relocation.
  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
* dialogue: route TTS via Say sub-skill; decouple history from chatbot
  Two related refactors:
  1. Replace direct tts_msgs/action/TTS calls with calls to a Say
  sub-skill action (communication_skills/action/Say) so the TTS engine
  lives behind a composable skill boundary. SayClient replaces
  TTSClient; the action endpoint is configurable via the new
  `say_action` ROS parameter (default `/tts/say`). Per-word feedback
  is read from std_skills/Feedback.data_str.
  2. Treat Chat/Ask as first-class dialogues regardless of chatbot
  availability. Without a chatbot the Dialogue becomes a passive
  container: SpeechHandler records user utterances against it,
  RAW_USER_INPUT intents drive the controlling script, and Say goals
  speak back. Ask without a chatbot waits up to 5s for the next
  utterance and returns it as a free-text answer.
  Session lifecycle adds an async summarizer (default = render session
  utterances as text; LLM-backed implementation is a future plug-in)
  that fires at finalize-and-archive time. The summary lives on the
  finished dialogue and is used to pre-fill the next dialogue's history
  with a SUMMARY + SESSION_BREAK pair. Pre-fill content is excluded
  from archival via Dialogue.session_start_index, so summaries cannot
  bleed into the next session's persisted record.
  Persistence is no longer tied to shutdown only: the in-memory store
  is flushed every 30s while active, and any in-flight dialogues are
  finalized (and summarized) on on_deactivate before clear_all.
  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
* Contributors: Séverin Lemaignan

1.0.0 (2026-05-08)
------------------
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
