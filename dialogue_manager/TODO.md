# TODO

Follow-up work tracked outside of issues. See also `doc/DIALOGUE_FLOW.md`
for the conceptual model these items refine.

## LLM context delivery — redesign needed

`doc/DIALOGUE_FLOW.md` specifies that the *current conversation context* is
"provided to the LLM backend as the dialogue context every time a
DialogueInteraction takes place".

Today we approximate this with a single `__system__`
`DialogueInteraction` injected at dialogue start
(`ChatbotClient.inject_context`). This works for backends that keep their
own per-`dialogue_id` state, but it has two known limitations:

1. The context is delivered **once**, not on every turn — so
   summarization-driven changes mid-dialogue do not propagate.
2. LLM-based backends typically maintain their own chat history and may
   end up with two overlapping copies (their internal one plus our prefix).

Two viable redesigns:

- **(A)** Add an optional `string context` field to
  `chatbot_msgs/DialogueInteraction.srv`, populated on every call.
  Backends that already track their own history would ignore it.
- **(B)** Add a dedicated `chatbot/set_context` service so context can be
  pushed as a separate concern from input-driven turns.

Decision blocked on input from the chatbot backend authors.

## ROS4HRI groups in pyhri

`pyhri` does not currently expose person groups. As a result,
`DialogueManagerNode._resolve_group_members` returns `[]` and group
dialogues are archived only under their group key — they are *not*
fanned out into each member's personal conversations history, which the
spec says should happen.

Once `pyhri` exposes groups (e.g. an `HRIListener.groups` mapping with a
`members` accessor), wire it into `_resolve_group_members` in
`dialogue_manager/manager_node.py`. The `ConversationsHistoryStore.archive`
already accepts a `group_members` iterable for fan-out — only the
resolver needs to be filled in.

## Default-chat interlocutor binding

The default chat dialogue is created with an empty `Interlocutor()` and
waits for someone to speak. Today its history accumulates against the
unbound dialogue and is therefore not archived to any specific person on
completion.

Once `voice_id` → `person_id` mapping is available (likely via pyhri),
the speech handler should rebind the default dialogue's interlocutor on
the first utterance so the conversation lands in the correct personal
history. `Interlocutor` is a frozen dataclass but
`Dialogue.interlocutor` is mutable, so rebinding is a single assignment.
