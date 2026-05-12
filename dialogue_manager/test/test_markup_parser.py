# Copyright (c) 2026 IIIA-CSIC. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the multi-modal expression markup parser."""

import os

from dialogue_manager.markup.ast_nodes import (
    Expression,
    MarkupAction,
    PauseAction,
    TextSegment,
    VariableRef,
    VariableText,
)
from dialogue_manager.markup.parser import (
    extract_plain_text,
    MarkupParseError,
    parse_expression,
)
import pytest

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_VALID_EXAMPLES = os.path.join(_TEST_DIR, 'markup_valid_examples')
_INVALID_EXAMPLES = os.path.join(_TEST_DIR, 'markup_invalid_examples')


def _load_examples(path):
    """Load non-empty lines from an example file."""
    with open(path) as f:
        return [line for line in f.read().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Validate against example files
# ---------------------------------------------------------------------------

class TestValidExamples:
    """Every line in markup_valid_examples must parse without error."""

    @pytest.fixture(params=_load_examples(_VALID_EXAMPLES),
                    ids=[f'valid_{i+1}' for i in
                         range(len(_load_examples(_VALID_EXAMPLES)))])
    def valid_line(self, request):
        """Parametrize over valid example lines."""
        return request.param

    def test_parses_without_error(self, valid_line):
        """Valid example parses successfully."""
        result = parse_expression(valid_line)
        assert isinstance(result, Expression)
        assert len(result.segments) > 0


class TestInvalidExamples:
    """Every line in markup_invalid_examples must raise MarkupParseError."""

    @pytest.fixture(params=_load_examples(_INVALID_EXAMPLES),
                    ids=[f'invalid_{i+1}' for i in
                         range(len(_load_examples(_INVALID_EXAMPLES)))])
    def invalid_line(self, request):
        """Parametrize over invalid example lines."""
        return request.param

    def test_raises_parse_error(self, invalid_line):
        """Invalid example raises MarkupParseError."""
        with pytest.raises(MarkupParseError):
            parse_expression(invalid_line)


# ---------------------------------------------------------------------------
# Detailed AST structure tests
# ---------------------------------------------------------------------------

class TestPlainText:
    """Tests for plain text parsing."""

    def test_plain_text(self):
        """Plain text becomes a single TextSegment."""
        expr = parse_expression("Hello, I'm a robot.")
        assert expr.segments == (TextSegment(text="Hello, I'm a robot."),)

    def test_empty_input(self):
        """Empty input produces empty expression."""
        expr = parse_expression('')
        assert expr.segments == ()


class TestActions:
    """Tests for action parsing."""

    def test_set_with_positional_arg(self):
        """<set expression(happy)> parses correctly."""
        expr = parse_expression('<set expression(happy)>')
        assert len(expr.segments) == 1
        action = expr.segments[0]
        assert isinstance(action, MarkupAction)
        assert action.verb == 'set'
        assert action.name == 'expression'
        assert action.positional_args == ('happy',)
        assert action.keyword_args == {}
        assert action.timeout is None

    def test_start_with_keyword_arg(self):
        """<start motor(speed=42)> parses correctly."""
        expr = parse_expression('<start motor(speed=42)>')
        action = expr.segments[0]
        assert action.verb == 'start'
        assert action.name == 'motor'
        assert action.positional_args == ()
        assert action.keyword_args == {'speed': 42}

    def test_do_action(self):
        """<do something> parses correctly."""
        expr = parse_expression('<do something>')
        action = expr.segments[0]
        assert action.verb == 'do'
        assert action.name == 'something'

    def test_stop_action(self):
        """<stop animation> parses correctly."""
        expr = parse_expression('<stop animation>')
        action = expr.segments[0]
        assert action.verb == 'stop'
        assert action.name == 'animation'

    def test_wait_with_timeout(self):
        """<wait motor timeout=5> parses correctly."""
        expr = parse_expression('<wait motor timeout=5>')
        action = expr.segments[0]
        assert action.verb == 'wait'
        assert action.name == 'motor'
        assert action.timeout == 5.0

    def test_mixed_positional_and_keyword_args(self):
        """<start motor(100, direction=forward)> parses correctly."""
        expr = parse_expression('<start motor(100, direction=forward)>')
        action = expr.segments[0]
        assert action.positional_args == (100,)
        assert action.keyword_args == {'direction': 'forward'}

    def test_multiple_positional_args(self):
        """<set tts(hello, world)> parses correctly."""
        expr = parse_expression('<set tts(hello, world)>')
        action = expr.segments[0]
        assert action.name == 'tts'
        assert action.positional_args == ('hello', 'world')

    def test_literal_values(self):
        """<start effect(true, false, null)> parses correctly."""
        expr = parse_expression('<start effect(true, false, null)>')
        action = expr.segments[0]
        assert action.positional_args == (True, False, None)

    def test_quoted_string_values(self):
        """Quoted strings are unquoted in the AST."""
        expr = parse_expression('<set config(name="quoted value")>')
        action = expr.segments[0]
        assert action.keyword_args == {'name': 'quoted value'}

    def test_single_quoted_string(self):
        """Single-quoted strings work too."""
        expr = parse_expression("<set config(name='single quoted')>")
        action = expr.segments[0]
        assert action.keyword_args == {'name': 'single quoted'}

    def test_array_value(self):
        """<start led(color=[1,2,3])> parses correctly."""
        expr = parse_expression('<start led(color=[1,2,3])>')
        action = expr.segments[0]
        assert action.keyword_args == {'color': [1, 2, 3]}

    def test_array_trailing_comma(self):
        """Trailing comma in array is allowed."""
        expr = parse_expression('<start led(color=[1, 2, 3,])>')
        action = expr.segments[0]
        assert action.keyword_args == {'color': [1, 2, 3]}

    def test_empty_array(self):
        """Empty arrays parse correctly."""
        expr = parse_expression('<start led(colors=[], fallback=[1])>')
        action = expr.segments[0]
        assert action.keyword_args == {'colors': [], 'fallback': [1]}

    def test_tts_as_identifier(self):
        """'tts' can be used as an action name (identifier)."""
        expr = parse_expression('<set tts(hello, world)>')
        assert expr.segments[0].name == 'tts'

    def test_timeout_as_identifier(self):
        """'timeout' can be used as a keyword arg name."""
        expr = parse_expression('<set config(timeout="5s", tts="enabled")>')
        action = expr.segments[0]
        assert action.keyword_args == {'timeout': '5s', 'tts': 'enabled'}


class TestNumbers:
    """Tests for number parsing."""

    def test_decimal(self):
        """Decimal number parses as float."""
        expr = parse_expression('<set volume(0.5)>')
        assert expr.segments[0].positional_args == (0.5,)

    def test_negative_integer(self):
        """Negative integer parses correctly."""
        expr = parse_expression('<set gain(-3)>')
        assert expr.segments[0].positional_args == (-3,)

    def test_scientific_notation(self):
        """Scientific notation parses correctly."""
        expr = parse_expression('<set gain(+1.5e2)>')
        assert expr.segments[0].positional_args == (150.0,)

    def test_hex(self):
        """Hex number parses correctly."""
        expr = parse_expression('<set addr(0xFF)>')
        assert expr.segments[0].positional_args == (0xFF,)


class TestPause:
    """Tests for pause action."""

    def test_pause_zero(self):
        """<pause(0)> parses correctly."""
        expr = parse_expression('<pause(0)>')
        assert expr.segments == (PauseAction(duration=0.0),)

    def test_pause_integer(self):
        """<pause(10)> parses correctly."""
        expr = parse_expression('<pause(10)>')
        assert expr.segments == (PauseAction(duration=10.0),)

    def test_pause_float(self):
        """<pause(1)> parses correctly."""
        expr = parse_expression('<pause(1)>')
        assert expr.segments == (PauseAction(duration=1.0),)


class TestVariableText:
    """Tests for variable text in default mode."""

    def test_variable_with_default(self):
        """@name|"default" parses correctly."""
        expr = parse_expression('Hello @name|"default"! How are you?')
        assert len(expr.segments) == 3
        assert expr.segments[0] == TextSegment(text='Hello ')
        assert expr.segments[1] == VariableText(
            query=('name',), default='default'
        )
        assert expr.segments[2] == TextSegment(text='! How are you?')

    def test_dotted_variable(self):
        """@user.name|"stranger" parses correctly."""
        expr = parse_expression('@user.name|"stranger", welcome!')
        assert expr.segments[0] == VariableText(
            query=('user', 'name'), default='stranger'
        )
        assert expr.segments[1] == TextSegment(text=', welcome!')


class TestVariableValues:
    """Tests for variable references inside actions."""

    def test_variable_positional_arg(self):
        """Variable reference as positional arg."""
        expr = parse_expression('<start tts(@msg|"hi")>')
        action = expr.segments[0]
        assert action.positional_args == (
            VariableRef(query=('msg',), default='hi'),
        )

    def test_variable_keyword_arg(self):
        """Variable reference as keyword arg value."""
        expr = parse_expression('<set config(key=@setting|"fallback")>')
        action = expr.segments[0]
        assert action.keyword_args == {
            'key': VariableRef(query=('setting',), default='fallback')
        }

    def test_variable_no_default(self):
        """Variable reference without default value."""
        expr = parse_expression('<set config(key=@setting)>')
        action = expr.segments[0]
        assert action.keyword_args == {
            'key': VariableRef(query=('setting',), default=None)
        }

    def test_variable_numeric_default(self):
        """Variable reference with numeric default."""
        expr = parse_expression('<start motor(speed=@user.speed|100)>')
        action = expr.segments[0]
        assert action.keyword_args == {
            'speed': VariableRef(query=('user', 'speed'), default=100)
        }


class TestComplexExpressions:
    """Tests for multi-segment expressions."""

    def test_mixed_text_and_actions(self):
        """Text interleaved with actions."""
        expr = parse_expression(
            '<set expression(happy)> <start motion(wave)> Hello! '
            '<wait motion timeout=1> <set expression(neutral)>'
        )
        assert len(expr.segments) == 7
        assert isinstance(expr.segments[0], MarkupAction)
        assert isinstance(expr.segments[1], TextSegment)  # space
        assert isinstance(expr.segments[2], MarkupAction)
        assert isinstance(expr.segments[3], TextSegment)  # ' Hello! '
        assert isinstance(expr.segments[4], MarkupAction)  # wait
        assert isinstance(expr.segments[5], TextSegment)  # space
        assert isinstance(expr.segments[6], MarkupAction)

    def test_text_with_pause(self):
        """Text with pause between sentences."""
        expr = parse_expression(
            '<start motion(wave)> Hello! <pause(2)> '
            'Anything new going on? <wait motion>'
        )
        segments = expr.segments
        assert isinstance(segments[0], MarkupAction)
        assert segments[0].verb == 'start'
        assert isinstance(segments[2], PauseAction)
        assert segments[2].duration == 2.0


class TestExtractPlainText:
    """Tests for extract_plain_text helper."""

    def test_plain_text_only(self):
        """Plain text expression returns its text."""
        expr = parse_expression('Hello world')
        assert extract_plain_text(expr) == 'Hello world'

    def test_with_actions(self):
        """Actions are stripped, text preserved."""
        expr = parse_expression(
            '<set expression(happy)> Hello! <pause(2)> Goodbye.'
        )
        assert extract_plain_text(expr) == ' Hello!  Goodbye.'

    def test_with_variables(self):
        """Variable text uses default value."""
        expr = parse_expression('Hello @name|"friend"!')
        assert extract_plain_text(expr) == 'Hello friend!'

    def test_empty(self):
        """Empty expression returns empty string."""
        expr = parse_expression('')
        assert extract_plain_text(expr) == ''
