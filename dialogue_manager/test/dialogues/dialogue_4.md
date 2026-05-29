# Stress test: 4-person group with cascading downsizing.
#
# Sequence of memberships: {A,B,C,D} → {B,C,D} → {C,D} → solo D.
# History queries only happen against groups that have actually been
# realised; ghost subgroups are checked with "does not exist".

[A joins]
[B joins]
[C joins]
[D joins]
A:
    - greetings team
B:
    - hi all
C:
    - hello!
D:
    - hey there
R:
    - welcome everyone
# CHECK: A history contains 5 utterances; B history contains 5 utterances
# CHECK: C history contains 5 utterances; D history contains 5 utterances
# CHECK: {A,B,C,D} group history contains 5 utterances
# CHECK: A and B are in the same group; C and D are in the same group
# CHECK: A and C are in the same group; B and D are in the same group
# No proper subset dialogues were ever spawned.
# CHECK: {A,B} group does not exist; {A,C} group does not exist
# CHECK: {B,C} group does not exist; {C,D} group does not exist
# CHECK: {A,B,C} group does not exist; {B,C,D} group does not exist
[A leaves]
# {A,B,C,D} has been archived. {B,C,D} is tracked but not yet realised
# (no one has spoken in it).
# CHECK: A is known but not active
# CHECK: {B,C,D} group does not exist
B:
    - looks like we lost A
C:
    - poor A
# Now {B,C,D} is realised. Its history view = own 2 utterances + the
# archived {A,B,C,D}'s 5 (B and C were both there).
# CHECK: A history contains 5 utterances
# CHECK: B history contains 7 utterances; C history contains 7 utterances; D history contains 7 utterances
# CHECK: {B,C,D} group history contains 7 utterances
# CHECK: "lost" does not appear in A history
# CHECK: "lost" appears in {B,C,D} group history
[B leaves]
# {B,C,D} archived. {C,D} tracked but not yet realised.
# CHECK: A is known but not active; B is known but not active
# CHECK: {C,D} group does not exist
C:
    - and now B is gone
D:
    - just us two
# {C,D} realised. History view = own 2 utterances + {A,B,C,D} (5) +
# {B,C,D} (2) — all superset archives containing both C and D.
# CHECK: A history contains 5 utterances; B history contains 7 utterances
# CHECK: C history contains 9 utterances; D history contains 9 utterances
# CHECK: {C,D} group history contains 9 utterances
# CHECK: "us two" appears in C history; "us two" appears in D history
# CHECK: "us two" does not appear in A history; "us two" does not appear in B history
# {B,C,D}'s history was closed before C and D spoke alone.
# CHECK: "us two" does not appear in {B,C,D} group history
# CHECK: {A,D} group does not exist; {A,C} group does not exist
[C leaves]
# CHECK: A is known but not active; B is known but not active
# CHECK: C is known but not active
D:
    - alone now
# D solo; no current group. The utterance lands only in D's personal
# dialogue.
# CHECK: D history contains 10 utterances
# CHECK: A history contains 5 utterances; B history contains 7 utterances; C history contains 9 utterances
# CHECK: "alone now" appears in D history
# CHECK: "alone now" does not appear in A history
# CHECK: "alone now" does not appear in B history
# CHECK: "alone now" does not appear in C history
[A joins]
# CHECK: A is active; A history contains 5 utterances; D history contains 10 utterances
A:
    - I'm back, but looks like I missed a lot
# CHECK: A history contains 6 utterances; D history contains 11 utterances
# CHECK: "missed a lot" appears in {A,D} group history
# CHECK: "missed a lot" does not appear in {B,C,D} group history
