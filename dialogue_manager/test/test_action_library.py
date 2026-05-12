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

"""Unit tests for the markup action library."""

import os
import tempfile

from unittest.mock import MagicMock

from dialogue_manager.markup.action_library import (
    ActionLibrary,
    resolve_field_template,
    resolve_fields,
    resolve_variable,
)
from dialogue_manager.markup.ast_nodes import VariableRef


class TestResolveFieldTemplate:
    """Tests for field template resolution."""

    def test_positional_arg(self):
        """#1 resolves to first positional arg."""
        result = resolve_field_template('#1', ('happy',), {})
        assert result == 'happy'

    def test_named_arg(self):
        """$name resolves to named arg."""
        result = resolve_field_template('$name', (), {'name': 'happy'})
        assert result == 'happy'

    def test_positional_then_named(self):
        """#1$name tries positional first, then named."""
        result = resolve_field_template('#1$name', ('positional',), {'name': 'named'})
        assert result == 'positional'

    def test_positional_fallback_to_named(self):
        """#1$name falls back to named when no positional."""
        result = resolve_field_template('#1$name', (), {'name': 'named'})
        assert result == 'named'

    def test_default_value(self):
        """$name|default uses default when arg missing."""
        result = resolve_field_template('$name|neutral', (), {})
        assert result == 'neutral'

    def test_full_template(self):
        """#1$name|default resolves with all options."""
        # No positional, no named -> default
        result = resolve_field_template('#1$name|neutral', (), {})
        assert result == 'neutral'

    def test_numeric_default(self):
        """Numeric default is coerced to number."""
        result = resolve_field_template('$x|1.0', (), {})
        assert result == 1.0

    def test_boolean_default(self):
        """Boolean default is coerced to bool."""
        result = resolve_field_template('$flag|false', (), {})
        assert result is False

    def test_plain_string(self):
        """Non-template string passes through unchanged."""
        result = resolve_field_template('literal', (), {})
        assert result == 'literal'

    def test_no_match_returns_none(self):
        """Template with no match and no default returns None."""
        result = resolve_field_template('$missing', (), {})
        assert result is None


class TestResolveFields:
    """Tests for recursive field resolution."""

    def test_simple_fields(self):
        """Flat fields are resolved."""
        template = {'expression': '#1$name|neutral'}
        result = resolve_fields(template, ('happy',), {})
        assert result == {'expression': 'happy'}

    def test_nested_fields(self):
        """Nested dicts are resolved recursively."""
        template = {
            'target': {
                'point': {
                    'x': '$x|1.0',
                    'y': '#1$y',
                }
            }
        }
        result = resolve_fields(template, (2.0,), {'x': 3.0})
        assert result == {'target': {'point': {'x': 3.0, 'y': 2.0}}}

    def test_literal_values_pass_through(self):
        """Non-string values in template pass through."""
        template = {'skip_planning': False, 'name': '#1$name'}
        result = resolve_fields(template, ('wave',), {})
        assert result == {'skip_planning': False, 'name': 'wave'}


class TestResolveVariable:
    """Tests for variable reference resolution."""

    def test_simple_variable(self):
        """Simple variable reference."""
        ref = VariableRef(query=('name',), default='stranger')
        assert resolve_variable(ref, {'name': 'Alice'}) == 'Alice'

    def test_dotted_variable(self):
        """Dotted variable reference."""
        ref = VariableRef(query=('user', 'name'), default='stranger')
        result = resolve_variable(ref, {'user': {'name': 'Bob'}})
        assert result == 'Bob'

    def test_missing_variable_uses_default(self):
        """Missing variable uses default."""
        ref = VariableRef(query=('missing',), default='fallback')
        assert resolve_variable(ref, {'other': 'value'}) == 'fallback'

    def test_none_variables_uses_default(self):
        """None variables dict uses default."""
        ref = VariableRef(query=('name',), default='default')
        assert resolve_variable(ref, None) == 'default'


class TestActionLibraryLoad:
    """Tests for loading action definitions from YAML."""

    def test_load_yaml_config(self):
        """Loading a valid YAML config populates definitions."""
        config = """
expression:
  type: interaction_skills/msg/SetExpression
  path: /skill/set_expression
  fields:
    expression: "#1$name|neutral"
motion:
  type: motion_skills/action/ReplayMotion
  path: /skill/replay_motion
  fields:
    name: "#1$name"
"""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False
        ) as f:
            f.write(config)
            f.flush()
            path = f.name

        try:
            node = MagicMock()
            lib = ActionLibrary(
                _node=node,
                _disabled_actions=[],
                _default_timeout=10.0,
            )
            lib.load([path])

            defn = lib.get_definition('expression')
            assert defn is not None
            assert defn.interface_kind == 'msg'
            assert defn.path == '/skill/set_expression'

            defn2 = lib.get_definition('motion')
            assert defn2 is not None
            assert defn2.interface_kind == 'action'
        finally:
            os.unlink(path)

    def test_disabled_actions_skipped(self):
        """Disabled actions are not loaded."""
        config = """
motion:
  type: motion_skills/action/ReplayMotion
  path: /skill/replay_motion
  fields:
    name: "#1$name"
"""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False
        ) as f:
            f.write(config)
            f.flush()
            path = f.name

        try:
            node = MagicMock()
            lib = ActionLibrary(
                _node=node,
                _disabled_actions=['motion'],
                _default_timeout=10.0,
            )
            lib.load([path])
            assert lib.get_definition('motion') is None
        finally:
            os.unlink(path)

    def test_service_interface_kind(self):
        """Service type string is correctly classified."""
        config = """
custom_srv:
  type: std_srvs/srv/Trigger
  path: /some_service
  fields: {}
"""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False
        ) as f:
            f.write(config)
            f.flush()
            path = f.name

        try:
            node = MagicMock()
            lib = ActionLibrary(
                _node=node,
                _disabled_actions=[],
                _default_timeout=10.0,
            )
            lib.load([path])
            defn = lib.get_definition('custom_srv')
            assert defn.interface_kind == 'srv'
        finally:
            os.unlink(path)

    def test_missing_file_logs_warning(self):
        """Missing config file logs a warning but doesn't crash."""
        node = MagicMock()
        lib = ActionLibrary(
            _node=node,
            _disabled_actions=[],
            _default_timeout=10.0,
        )
        lib.load(['/nonexistent/path.yaml'])
        node.get_logger().warn.assert_called()
