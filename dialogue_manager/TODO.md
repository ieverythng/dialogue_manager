# TODO

Follow-up work tracked outside of issues. See also `doc/DIALOGUE_FLOW.md`
for the conceptual model these items refine.

## LLM context delivery — redesign needed

`doc/DIALOGUE_FLOW.md` specifies that the *current conversation context*
is "provided to the LLM backend as the dialogue context every time a
DialogueInteraction takes place".

Today we approximate this with a single `__system__`
`DialogueInteraction` injected at dialogue start
(`ChatbotClient.inject_context`). This works for backends that keep
their own per-`dialogue_id` state, but it has two known limitations:

1. The context is delivered **once**, not on every turn — so
   summarization-driven changes mid-dialogue do not propagate.
2. LLM-based backends typically maintain their own chat history and may
   end up with two overlapping copies (their internal one plus our
   prefix).

Two viable redesigns:

- **(A)** Add an optional `string context` field to
  `chatbot_msgs/DialogueInteraction.srv`, populated on every call.
  Backends that already track their own history would ignore it.
- **(B)** Add a dedicated `chatbot/set_context` service so context can
  be pushed as a separate concern from input-driven turns.

Decision blocked on input from the chatbot backend authors.

## LLM-backed summariser

The session-end summariser hook (`SkillServers.__init__(summarizer=…)`)
defaults to a trivial fallback that renders the session's utterances as
text. An LLM-backed implementation that calls `chatbot/summarize` (when
the chatbot exposes it) would give meaningfully cumulative summaries
and improve next-session pre-fill quality.

## Lazy subgroup history pre-fill for LLM context

The test infrastructure already computes "{B,C}'s history" as the
query-time superset of past dialogues that include B and C
(`test_dialogue_dsl.py::_all_dialogues_for`). The runtime LLM
context-builder doesn't yet do the same: when a new `{B,C}` group
dialogue spawns, it currently only carries forward the per-person
summaries via `ConversationsHistoryStore.preload_into`.

What's missing for full parity with `DIALOGUE_FLOW.md`: at spawn
time for a new group `{S}`, lazily pre-fill the new dialogue's
history with utterances from every archived dialogue whose member
set is a superset of `{S}` (the conversations they were both part
of). This is the lazy analogue of the per-person preload, and
scales — no archive-time combinatorial fan-out, just a one-time
scan when the subgroup actually forms.

Done lazily it stays cheap even for large gatherings: a 40-person
crowd doesn't materialize 2^40 subset archive entries; subgroup
dialogues are created only when ROS4HRI actually reports them.

## Canonical group_id from member set

Group dialogues currently inherit ROS4HRI's `group_id` from
`/humans/interactions/groups`, which is opaque ("g1", "tracking_42",
…). A canonical convention `'group_<sorted_members_joined_by_
underscore>'` would make subgroup-history queries straightforward
and predictable (a frozenset of members maps to exactly one
group_id). manager_node would need a small canonicalisation step
when spawning group dialogues. The DSL tests already use this
convention.
