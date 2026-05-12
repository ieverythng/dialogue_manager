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
