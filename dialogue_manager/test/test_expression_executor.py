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

"""Unit tests for the expression executor."""

import threading
from unittest.mock import MagicMock, patch

from dialogue_manager.markup.action_library import ActionDefinition, ActionLibrary
from dialogue_manager.markup.ast_nodes import (
    Expression,
    MarkupAction,
    PauseAction,
    TextSegment,
    VariableText,
)
from dialogue_manager.markup.executor import ExpressionExecutor


def _make_executor(say_client=None, action_library=None):
    """Create an ExpressionExecutor with mocked dependencies."""
    node = MagicMock()
    if say_client is None:
        say_client = MagicMock()
        say_client.speak_and_wait.return_value = True
    if action_library is None:
        action_library = MagicMock(spec=ActionLibrary)
        action_library.get_definition.return_value = None
        action_library.default_timeout = 10.0
    return ExpressionExecutor(
        node=node,
        say_client=say_client,
        action_library=action_library,
        expression_timeout=60.0,
    )


class TestExecutorPlainText:
    """Tests for plain text execution."""

    def test_plain_text_speaks(self):
        """Plain text is sent to TTS."""
        tts = MagicMock()
        tts.speak_and_wait.return_value = True
        executor = _make_executor(say_client=tts)

        expr = Expression(segments=(TextSegment(text='Hello world'),))
        result = executor.execute(expr)

        assert result is True
        tts.speak_and_wait.assert_called_once()
        call_text = tts.speak_and_wait.call_args[0][0]
        assert call_text == 'Hello world'

    def test_empty_expression_succeeds(self):
        """Empty expression succeeds without calling TTS."""
        tts = MagicMock()
        executor = _make_executor(say_client=tts)

        expr = Expression(segments=())
        result = executor.execute(expr)

        assert result is True
        tts.speak_and_wait.assert_not_called()

    def test_whitespace_only_text_not_spoken(self):
        """Whitespace-only text segments are not spoken."""
        tts = MagicMock()
        tts.speak_and_wait.return_value = True
        executor = _make_executor(say_client=tts)

        expr = Expression(segments=(TextSegment(text='  '),))
        result = executor.execute(expr)

        assert result is True
        tts.speak_and_wait.assert_not_called()


class TestExecutorVariableText:
    """Tests for variable text resolution."""

    def test_variable_resolved(self):
        """Variable text is resolved from variables dict."""
        tts = MagicMock()
        tts.speak_and_wait.return_value = True
        executor = _make_executor(say_client=tts)

        expr = Expression(segments=(
            TextSegment(text='Hello '),
            VariableText(query=('name',), default='stranger'),
            TextSegment(text='!'),
        ))
        result = executor.execute(
            expr, variables={'name': 'Alice'}
        )

        assert result is True
        call_text = tts.speak_and_wait.call_args[0][0]
        assert call_text == 'Hello Alice!'

    def test_variable_default_used(self):
        """Default is used when variable not in dict."""
        tts = MagicMock()
        tts.speak_and_wait.return_value = True
        executor = _make_executor(say_client=tts)

        expr = Expression(segments=(
            VariableText(query=('missing',), default='friend'),
        ))
        result = executor.execute(expr, variables={})

        assert result is True
        call_text = tts.speak_and_wait.call_args[0][0]
        assert call_text == 'friend'


class TestExecutorPause:
    """Tests for pause action."""

    def test_pause_returns_true(self):
        """Pause completes successfully."""
        executor = _make_executor()

        expr = Expression(segments=(PauseAction(duration=0.0),))
        result = executor.execute(expr)

        assert result is True

    def test_pause_respects_cancellation(self):
        """Pause is interrupted by cancel event."""
        executor = _make_executor()
        cancel = threading.Event()
        cancel.set()  # Pre-cancelled

        expr = Expression(segments=(PauseAction(duration=10.0),))
        result = executor.execute(expr, cancel_event=cancel)

        assert result is False


class TestExecutorActions:
    """Tests for markup action dispatch."""

    def test_unknown_action_skipped(self):
        """Unknown actions are logged and skipped."""
        action_lib = MagicMock(spec=ActionLibrary)
        action_lib.get_definition.return_value = None
        action_lib.default_timeout = 10.0
        executor = _make_executor(action_library=action_lib)

        expr = Expression(segments=(
            MarkupAction(verb='set', name='unknown_action'),
        ))
        result = executor.execute(expr)

        assert result is True

    def test_set_publishes_topic(self):
        """Set verb on a msg type publishes to the topic."""
        mock_publisher = MagicMock()
        mock_msg_class = MagicMock()
        mock_msg_instance = MagicMock()
        mock_msg_class.return_value = mock_msg_instance

        defn = ActionDefinition(
            name='expression',
            type_str='interaction_skills/msg/SetExpression',
            path='/skill/set_expression',
            fields={'expression': '#1$name|neutral'},
            interface_kind='msg',
            interface_class=mock_msg_class,
        )

        action_lib = MagicMock(spec=ActionLibrary)
        action_lib.get_definition.return_value = defn
        action_lib.get_client.return_value = mock_publisher
        action_lib.default_timeout = 10.0

        executor = _make_executor(action_library=action_lib)

        expr = Expression(segments=(
            MarkupAction(
                verb='set', name='expression',
                positional_args=('happy',)
            ),
        ))

        with patch(
            'dialogue_manager.markup.executor.set_message_fields'
        ):
            result = executor.execute(expr)

        assert result is True
        mock_publisher.publish.assert_called_once()


class TestExecutorNestedFields:
    """Tests for dispatch with nested field structures."""

    def test_set_publishes_nested_fields(self):
        """Set verb correctly resolves nested field templates."""
        mock_publisher = MagicMock()
        mock_msg_class = MagicMock()
        mock_msg_instance = MagicMock()
        mock_msg_class.return_value = mock_msg_instance

        # Nested fields: expression.expression = "#1$name|neutral"
        defn = ActionDefinition(
            name='expression',
            type_str='interaction_skills/msg/SetExpression',
            path='/skill/set_expression',
            fields={'expression': {'expression': '#1$name|neutral'}},
            interface_kind='msg',
            interface_class=mock_msg_class,
        )

        action_lib = MagicMock(spec=ActionLibrary)
        action_lib.get_definition.return_value = defn
        action_lib.get_client.return_value = mock_publisher
        action_lib.default_timeout = 10.0

        executor = _make_executor(action_library=action_lib)

        expr = Expression(segments=(
            MarkupAction(
                verb='set', name='expression',
                positional_args=('happy',)
            ),
        ))

        with patch(
            'dialogue_manager.markup.executor.set_message_fields'
        ) as mock_set:
            result = executor.execute(expr)

        assert result is True
        mock_publisher.publish.assert_called_once()
        # Verify set_message_fields received the nested dict
        mock_set.assert_called_once_with(
            mock_msg_instance,
            {'expression': {'expression': 'happy'}},
        )

    def test_set_nested_fields_with_default(self):
        """Nested fields use defaults when no args provided."""
        mock_publisher = MagicMock()
        mock_msg_class = MagicMock()
        mock_msg_instance = MagicMock()
        mock_msg_class.return_value = mock_msg_instance

        defn = ActionDefinition(
            name='expression',
            type_str='interaction_skills/msg/SetExpression',
            path='/skill/set_expression',
            fields={'expression': {'expression': '#1$name|neutral'}},
            interface_kind='msg',
            interface_class=mock_msg_class,
        )

        action_lib = MagicMock(spec=ActionLibrary)
        action_lib.get_definition.return_value = defn
        action_lib.get_client.return_value = mock_publisher
        action_lib.default_timeout = 10.0

        executor = _make_executor(action_library=action_lib)

        # No positional args -> default "neutral"
        expr = Expression(segments=(
            MarkupAction(verb='set', name='expression'),
        ))

        with patch(
            'dialogue_manager.markup.executor.set_message_fields'
        ) as mock_set:
            result = executor.execute(expr)

        assert result is True
        mock_set.assert_called_once_with(
            mock_msg_instance,
            {'expression': {'expression': 'neutral'}},
        )

    def test_field_structure_mismatch_gives_informative_error(self):
        """Mismatched field structure logs a warning with helpful context."""
        mock_publisher = MagicMock()
        mock_msg_class = MagicMock()
        mock_msg_instance = MagicMock()
        mock_msg_class.return_value = mock_msg_instance

        # Flat field — will cause set_message_fields to fail if the
        # actual message expects a nested structure
        defn = ActionDefinition(
            name='expression',
            type_str='interaction_skills/msg/SetExpression',
            path='/skill/set_expression',
            fields={'expression': '#1$name|neutral'},
            interface_kind='msg',
            interface_class=mock_msg_class,
        )

        action_lib = MagicMock(spec=ActionLibrary)
        action_lib.get_definition.return_value = defn
        action_lib.get_client.return_value = mock_publisher
        action_lib.default_timeout = 10.0

        executor = _make_executor(action_library=action_lib)

        expr = Expression(segments=(
            MarkupAction(
                verb='set', name='expression',
                positional_args=('happy',)
            ),
        ))

        with patch(
            'dialogue_manager.markup.executor.set_message_fields',
            side_effect=AttributeError(
                "Value 'happy' is expected to be a dictionary but is a str"
            ),
        ):
            result = executor.execute(expr)

        # Expression continues (unknown/failed actions don't abort)
        assert result is True
        # Verify the warning includes helpful guidance
        logger = executor._node.get_logger()
        warn_msg = logger.warn.call_args[0][0]
        assert 'YAML action definition' in warn_msg
        assert 'nested fields' in warn_msg

    def test_text_with_nested_set_action(self):
        """Full expression: text + nested set action dispatches correctly."""
        tts = MagicMock()
        tts.speak_and_wait.return_value = True

        mock_publisher = MagicMock()
        mock_msg_class = MagicMock()
        mock_msg_instance = MagicMock()
        mock_msg_class.return_value = mock_msg_instance

        defn = ActionDefinition(
            name='expression',
            type_str='interaction_skills/msg/SetExpression',
            path='/skill/set_expression',
            fields={'expression': {'expression': '#1$name|neutral'}},
            interface_kind='msg',
            interface_class=mock_msg_class,
        )

        action_lib = MagicMock(spec=ActionLibrary)
        action_lib.get_definition.return_value = defn
        action_lib.get_client.return_value = mock_publisher
        action_lib.default_timeout = 10.0

        executor = _make_executor(say_client=tts, action_library=action_lib)

        # Mimics: "Hello<set expression(happy)>what are you doing"
        expr = Expression(segments=(
            TextSegment(text='Hello'),
            MarkupAction(
                verb='set', name='expression',
                positional_args=('happy',)
            ),
            TextSegment(text='what are you doing'),
        ))

        with patch(
            'dialogue_manager.markup.executor.set_message_fields'
        ) as mock_set:
            result = executor.execute(expr)

        assert result is True
        assert tts.speak_and_wait.call_count == 2
        mock_publisher.publish.assert_called_once()
        mock_set.assert_called_once_with(
            mock_msg_instance,
            {'expression': {'expression': 'happy'}},
        )


class TestExecutorTextParsing:
    """Tests for execute_text (parsing + execution)."""

    def test_execute_text_plain(self):
        """Plain text string is parsed and spoken."""
        tts = MagicMock()
        tts.speak_and_wait.return_value = True
        executor = _make_executor(say_client=tts)

        result = executor.execute_text('Hello world')

        assert result is True
        tts.speak_and_wait.assert_called_once()

    def test_execute_text_invalid_markup_fallback(self):
        """Invalid markup falls back to speaking raw text."""
        tts = MagicMock()
        tts.speak_and_wait.return_value = True
        executor = _make_executor(say_client=tts)

        result = executor.execute_text('<invalid>')

        assert result is True
        tts.speak_and_wait.assert_called_once()
        call_text = tts.speak_and_wait.call_args[0][0]
        assert call_text == '<invalid>'


class TestExecutorCancellation:
    """Tests for cancellation handling."""

    def test_cancel_before_execution(self):
        """Pre-set cancel event stops execution immediately."""
        executor = _make_executor()
        cancel = threading.Event()
        cancel.set()

        expr = Expression(segments=(TextSegment(text='Hello'),))
        result = executor.execute(expr, cancel_event=cancel)

        assert result is False


class TestExecutorGetPlainText:
    """Tests for get_plain_text helper."""

    def test_strips_markup(self):
        """Markup is stripped, text preserved."""
        executor = _make_executor()
        result = executor.get_plain_text(
            '<set expression(happy)> Hello! <pause(2)> Goodbye.'
        )
        assert result == ' Hello!  Goodbye.'

    def test_invalid_markup_returns_raw(self):
        """Invalid markup returns the raw input."""
        executor = _make_executor()
        result = executor.get_plain_text('<invalid>')
        assert result == '<invalid>'


class TestExecutorMixedExpression:
    """Tests for expressions with mixed text and actions."""

    def test_text_between_actions(self):
        """Text between actions is spoken; actions are dispatched."""
        tts = MagicMock()
        tts.speak_and_wait.return_value = True

        action_lib = MagicMock(spec=ActionLibrary)
        action_lib.get_definition.return_value = None
        action_lib.default_timeout = 10.0

        executor = _make_executor(say_client=tts, action_library=action_lib)

        expr = Expression(segments=(
            MarkupAction(verb='set', name='expression',
                         positional_args=('happy',)),
            TextSegment(text=' Hello! '),
            MarkupAction(verb='set', name='expression',
                         positional_args=('neutral',)),
        ))
        result = executor.execute(expr)

        assert result is True
        assert tts.speak_and_wait.call_count == 1
        call_text = tts.speak_and_wait.call_args[0][0]
        assert call_text == ' Hello! '

    def test_consecutive_text_merged(self):
        """Consecutive text and variable segments are merged into one TTS call."""
        tts = MagicMock()
        tts.speak_and_wait.return_value = True
        executor = _make_executor(say_client=tts)

        expr = Expression(segments=(
            TextSegment(text='Hello '),
            VariableText(query=('name',), default='friend'),
            TextSegment(text=', welcome!'),
        ))
        result = executor.execute(expr, variables={'name': 'Bob'})

        assert result is True
        assert tts.speak_and_wait.call_count == 1
        call_text = tts.speak_and_wait.call_args[0][0]
        assert call_text == 'Hello Bob, welcome!'
