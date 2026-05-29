# Stress test: lazy spawn semantics.
#
# Group dialogues are spawned only when an utterance actually lands in
# them — not when ROS4HRI starts tracking the group. So a freshly-
# formed group is "in the same group" per the GroupHandler, but its
# dialogue doesn't exist until someone speaks. We can only query the
# history of a group that has actually been realised.

[A joins]
[B joins]
# CHECK: A and B are in the same group
# CHECK: A is unknown; B is unknown
# CHECK: {A,B} group does not exist
[C joins]
# CHECK: A and B are in the same group; A and C are in the same group
# CHECK: A is unknown; B is unknown; C is unknown
# CHECK: {A,B,C} group does not exist; {A,B} group does not exist
A:
    - finally someone speaks
# Now {A,B,C} is realised; subgroup pairs are still ghost.
# CHECK: A history contains 1 utterances
# CHECK: B history contains 1 utterances; C history contains 1 utterances
# CHECK: {A,B,C} group history contains 1 utterances
# CHECK: {A,B} group does not exist
# CHECK: {A,C} group does not exist
[B leaves]
# {A,B,C} is archived; {A,C} is tracked but no dialogue yet.
# CHECK: B is known but not active
# CHECK: {A,B,C} group history contains 1 utterances
# CHECK: {A,C} group does not exist
C:
    - now just A and me
# C's utterance realises {A,C}. Its history now aggregates its own
# utterance plus the superset content from the archived {A,B,C}.
# CHECK: A history contains 2 utterances; C history contains 2 utterances
# CHECK: B history contains 1 utterances
# CHECK: {A,C} group history contains 2 utterances
# CHECK: "just A and me" appears in C history
# CHECK: "just A and me" does not appear in B history
