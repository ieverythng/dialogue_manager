TERMINOLOGY AND CONVERSATIONAL FLOW
===================================


A *dialogue* represents an on-going conversation between the robot and a
person/group of persons, and is defined as a immutable tuple (UUID, role,
interlocutor). An interlocutor is either a ROS4HRI person ID or a ROS4HRI group
ID (each group ID uniquely maps to a set of person IDs). A dialogue has a
priority that can dynamically change (the dialogue with the highest priorty at
a given point is the *active* dialogue), as well as start and end timestamps.

A *role* is a named configuration that captures what the robot is doing in that
conversation. It is made of a *prompt*, which conveys the persona, the task,
and the conversational style the robot should adopt for that role (e.g. a
front-desk receptionist greeting visitors, a tour guide explaining exhibits, a
clinical companion supporting an elderly user); and a set of *semantic filters*
that govern which symbolic facts from the situated world model are injected
into the LLM context at each turn (e.g. which ontological classes of entities
to include: humans, furniture,  rooms, etc.) Two special roles are predefined:
DEFAULT_ROLE, used for generic open-ended interaction, and an ASK role,
primarily used for closed-form questions ('what is your age?')

Dialogues can be started by the users if they start talking to the robot
unprompted (in that case, the special DEFAULT_ROLE is used), or can be
pro-actively started by the robot (with a pre-defined role).

A dialogue ends either when the chatbot backend explicitely closes it (the
on-going 'Dialogue' action completes -- typically because the dialogue has
reached its proper conclusion), or when the caller of the 'Chat' or 'Ask'
skills cancel the action (for instance because the caller has determined that
the person has left).

The *dialogue history* is the chronological list of utterances associated to a specific
dialogue, with each utterance a tuple (timestamp, ID of the speaker -- the
robot: __myself__ or a person_id --,  text of the utterance).

For every person and group, we also maintain (and persist) a *conversations
history*, which consists in the chronological list of all past dialogue histories
involving that person or group. Importantly, if a person takes part to a
group conversation, that group conversation will appear *both* in the group's
history itself *and in the personal conversation history of each of the persons
involved in the interaction*. The conversations histories are persisted on
disk, so that past interactions between a given person and the robot are not
forgotten.

Finally, for each person/group, we maintain a *current conversation context*.
It is provided to the LLM backend as the dialogue context every time a
DialogueInteraction takes place, and can be understood as the 'digested'
version of the exhaustive conversation history. While a naive implementation
can simply 'copy/paste' the full history of conversations into the context,
refined strategies can by applied, like ignoring some irrelevant dialogues (eg
dialogues with the role ASK), or calling a LLM to summarize at regular interval
older dialogue histories to avoid filling the context with old and irrelevant
details, etc. In dialogue_manager, dialogues with the ASK role are removed from
the context, and, *if the chatbot backend exposes the action
'chatbot/summarize'*, it automatically summarizes all past conversations older
than 5 minutes.

