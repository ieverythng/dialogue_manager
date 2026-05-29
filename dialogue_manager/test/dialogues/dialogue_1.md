[A joins]
A:
    - hello!
R:
    - hi there!
A:
    - how are you?
R:
    - I'm doing well, thanks for asking! How about you?
# CHECK: A history contains 4 utterances; B is unknown
[B joins]
# CHECK: A and B are in the same group
A:
    - I'm good too, thanks for asking!
    - Hi B! Long time no see!
R:
    - Hi B!
B:
    - Hi guys. What's up?
[A leaves]
# CHECK: A is known but not active; A history contains 8 utterances; B history contains 4 utterances
R:
    - Oh, A left. I wanted to tell him about this movie project I'm working on.
# CHECK: "movie" appears in B history; "movie" does not appear in A history
[A joins]
# CHECK: A is active; A history contains 8 utterances; B history contains 5 utterances
A:
    - I'm back!
# CHECK: "movie" appears in B history; "movie" does not appear in A history
# CHECK: "Long time no see" appears in B history
# CHECK: {A,B} group history contains 5 utterances; "movie" does not appear in {A,B} group history
