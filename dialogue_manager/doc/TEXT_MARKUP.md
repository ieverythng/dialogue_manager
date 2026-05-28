# Multi-modal expression markup language

The multi-modal expression markup language is a feature added on top of
TTS synthesis. Using markups inserted in the text to be synthesized, it
integrates the speech synthesis with other robot functionalities.

The full markup action format is `<verb name(arguments) timeout>`. The
arguments and timeout are optional, and the minimal markup action format
is `<verb name>`.

The verbs must be one of:

-   `set`: 'start and forget' the action; useful when you do not need
    to know if/when the action is completed
-   `start`: start an action
-   `wait`: wait for a previously started action to finish (the first
    one found backwards with the same name)
-   `stop`: stop an on-going action (the first one found backwards with
    the same name)
-   `do`: equivalent to `start` immediately followed by `wait` (i.e.,
    blocks until the action is completed)

Markup action which are started (not set) and are not waited nor stopped
explicitly, are implicitly waited at the end of the multi-modal
expression.

Available actions are defined in `config/00-default_actions.yaml`.

The currently supported actions are:

- `motion(name)` : perform the `name` pre-recorded motion
- `expression(name="neutral")` : set the `name` predefined facial expressions
- `led(groups, effect="solid_color", color, alpha=1.0, secondary_color, secondary_alpha=1.0, cycle=1.0, partition=1.0)`: perform the specific led
    effect on a specific led group. RGB values are in the range [0, 255]
- `look_at(x=1.0,y,z,frame="base_link",policy="glance")`: look at a specific target

The timeout specifies the maximum number of seconds to wait for the
execution of an markup action.

Using markup action, one can synchronize the speech with a facial
expression or a gesture. For instance, the expression:
`<set expression(happy) <start motion(wave)> Hello! <wait motion timeout=1> <set expression(neutral)>`.
will make the robot say "Hello!" while waving and with a happy
expression, wait until the waving motion is finished (or 1 second has
passed since the motion start), and then return to a neutral expression.

**Note**: `dialogue_manager` provides the
parameter `disabled_markup_actions` to disable the execution of specific
markup actions.

### Built-in action

These actions resembles the standard markup actions semantically and
synthetically, but are exceptions to the rules.

Currently, the only built-in action are:

- `<pause(time)>` : the utterance is paused for the specified time in seconds

For instance, the expression:
`<start motion(wave)> Hello! <pause(2)> Anything new going on? <wait motion>`.
will make the robot stay silent for 2 seconds after saying "Hello!".
