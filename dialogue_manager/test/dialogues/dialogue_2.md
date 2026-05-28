[A joins]
[B joins]
[C joins]
A:
    - hello!
C:
    - hi A
B:
    - hi!
R:
    - hello everyone!
    - let's play a game
# CHECK: A history contains 5 utterances
# CHECK: B history contains 5 utterances
# CHECK: C history contains 5 utterances
# CHECK: A and B are in the same group
# CHECK: A and C are in the same group
[A leaves]
# CHECK: A is known but not active; A history contains 5 utterances
# CHECK: B and C are in the same group
B:
    - sure, sounds fun
C:
    - yeah, let's do it!
# CHECK: A history contains 5 utterances; B history contains 7 utterances; C history contains 7 utterances
# CHECK: {A,B,C} group history contains 5 utterances
# CHECK: {B,C} group history contains 7 utterances
# CHECK: {A,B} group does not exist
# CHECK: {A,C} group does not exist
# CHECK: "fun" appears in B history; "fun" does not appear in A history; "fun" appears in C history
# CHECK: "fun" appears in {B,C} group history
# CHECK: "fun" does not appear in {A,B,C} group history
[A joins]
A:
    - I'm back!
# CHECK: A is active; A history contains 6 utterances; B history contains 8 utterances; C history contains 8 utterances
# CHECK: {A,B,C} group history contains 6 utterances
# CHECK: {B,C} group history contains 8 utterances
# CHECK: {A,B} group does not exist
# CHECK: {A,C} group does not exist
