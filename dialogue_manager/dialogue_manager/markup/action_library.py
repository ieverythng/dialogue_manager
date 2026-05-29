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

"""Action library: loads markup action definitions and creates ROS2 clients."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.publisher import Publisher
from rosidl_runtime_py.utilities import get_action, get_message, get_service
import yaml

from .ast_nodes import VariableRef


# ---------------------------------------------------------------------------
# Field template resolution
# ---------------------------------------------------------------------------

_FIELD_TEMPLATE_RE = re.compile(
    r'^(?:#(\d+))?'   # optional positional index (#N)
    r'(?:\$(\w+))?'   # optional named key ($name)
    r'(?:\|(.+))?$'   # optional default value (|default)
)


def resolve_field_template(
    template: str,
    positional_args: tuple,
    keyword_args: dict,
) -> Any:
    """Resolve a field template like '#1$name|default' against arguments."""
    m = _FIELD_TEMPLATE_RE.match(template)
    if not m:
        return template  # Not a template, return as-is

    idx_str, name, default = m.groups()

    # Try positional arg (1-based index)
    if idx_str is not None:
        idx = int(idx_str) - 1
        if 0 <= idx < len(positional_args):
            return positional_args[idx]

    # Try named arg
    if name is not None and name in keyword_args:
        return keyword_args[name]

    # Use default
    if default is not None:
        return _coerce_default(default)

    return None


def _coerce_default(value: str) -> Any:
    """Coerce a string default value to the appropriate Python type."""
    if value.lower() == 'true':
        return True
    if value.lower() == 'false':
        return False
    if value.lower() == 'null':
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def resolve_fields(
    fields_template: dict,
    positional_args: tuple,
    keyword_args: dict,
    variables: dict | None = None,
) -> dict:
    """Recursively resolve a fields template dict against arguments."""
    result = {}
    for key, tmpl in fields_template.items():
        result[key] = _resolve_value(tmpl, positional_args, keyword_args,
                                     variables)
    return result


def _resolve_value(
    tmpl: Any,
    positional_args: tuple,
    keyword_args: dict,
    variables: dict | None,
) -> Any:
    """Resolve a single template value."""
    if isinstance(tmpl, dict):
        return {
            k: _resolve_value(v, positional_args, keyword_args, variables)
            for k, v in tmpl.items()
        }
    if isinstance(tmpl, str):
        return resolve_field_template(tmpl, positional_args, keyword_args)
    if isinstance(tmpl, VariableRef):
        return resolve_variable(tmpl, variables)
    # Literal values (bool, int, float, None) pass through
    return tmpl


def resolve_variable(ref: VariableRef, variables: dict | None) -> Any:
    """Resolve a VariableRef against a variables dict."""
    if variables:
        value = variables
        for key in ref.query:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return ref.default
        return value
    return ref.default


# ---------------------------------------------------------------------------
# Action definitions
# ---------------------------------------------------------------------------

@dataclass
class ActionDefinition:
    """Definition of a markup action loaded from config."""

    name: str
    type_str: str
    path: str
    fields: dict
    interface_kind: str  # 'msg', 'action', or 'srv'
    interface_class: Any = None


def _derive_interface_kind(type_str: str) -> str:
    """Derive the interface kind from a ROS2 type string."""
    parts = type_str.split('/')
    if len(parts) >= 2:
        kind = parts[1].lower()
        if kind in ('msg', 'action', 'srv'):
            return kind
    return 'msg'  # default fallback


# ---------------------------------------------------------------------------
# Action Library
# ---------------------------------------------------------------------------

@dataclass
class ActionLibrary:
    """Registry of markup actions with their ROS2 clients."""

    _node: Node
    _disabled_actions: list[str]
    _default_timeout: float
    _callback_group: ReentrantCallbackGroup | None = None
    _definitions: dict[str, ActionDefinition] = field(default_factory=dict)
    _clients: dict[str, Any] = field(default_factory=dict)

    def load(self, config_paths: list[str]) -> None:
        """Load action definitions from YAML config files."""
        for path in config_paths:
            try:
                with open(path) as f:
                    data = yaml.safe_load(f)
            except Exception as e:
                self._node.get_logger().warn(
                    f'[MARKUP] Failed to load config {path}: {e}'
                )
                continue

            if not isinstance(data, dict):
                self._node.get_logger().warn(
                    f'[MARKUP] Invalid config format in {path}'
                )
                continue

            for name, definition in data.items():
                if name in self._disabled_actions:
                    self._node.get_logger().info(
                        f'[MARKUP] Skipping disabled action: {name}'
                    )
                    continue

                type_str = definition.get('type', '')
                kind = _derive_interface_kind(type_str)

                self._definitions[name] = ActionDefinition(
                    name=name,
                    type_str=type_str,
                    path=definition.get('path', ''),
                    fields=definition.get('fields', {}),
                    interface_kind=kind,
                )
                self._node.get_logger().debug(
                    f'[MARKUP] Loaded action: {name} ({kind} on {definition.get("path", "")})'
                )

    def create_clients(self) -> None:
        """Create ROS2 clients/publishers for all loaded definitions."""
        for name, defn in self._definitions.items():
            try:
                cls = self._load_interface_class(defn.type_str, defn.interface_kind)
                defn.interface_class = cls
            except Exception as e:
                self._node.get_logger().warn(
                    f'[MARKUP] Cannot load type for {name} '
                    f'({defn.type_str}): {e}'
                )
                continue

            try:
                if defn.interface_kind == 'msg':
                    client = self._node.create_publisher(
                        cls, defn.path, 10,
                    )
                elif defn.interface_kind == 'action':
                    client = ActionClient(
                        self._node, cls, defn.path,
                        callback_group=self._callback_group,
                    )
                elif defn.interface_kind == 'srv':
                    client = self._node.create_client(
                        cls, defn.path,
                        callback_group=self._callback_group,
                    )
                else:
                    self._node.get_logger().warn(
                        f'[MARKUP] Unknown interface kind for {name}: '
                        f'{defn.interface_kind}'
                    )
                    continue

                self._clients[name] = client
                self._node.get_logger().info(
                    f'[MARKUP] Created {defn.interface_kind} client: '
                    f'{name} on {defn.path}'
                )
            except Exception as e:
                self._node.get_logger().warn(
                    f'[MARKUP] Failed to create client for {name}: {e}'
                )

    def get_definition(self, action_name: str) -> ActionDefinition | None:
        """Return the definition for an action name, or None."""
        return self._definitions.get(action_name)

    def get_client(self, action_name: str) -> Any:
        """Return the ROS2 client for an action name, or None."""
        return self._clients.get(action_name)

    @property
    def default_timeout(self) -> float:
        """Return the default action timeout in seconds."""
        return self._default_timeout

    def destroy(self) -> None:
        """Destroy all created clients and publishers."""
        for name, client in self._clients.items():
            try:
                if isinstance(client, ActionClient):
                    client.destroy()
                elif isinstance(client, Publisher):
                    self._node.destroy_publisher(client)
                else:
                    self._node.destroy_client(client)
            except Exception as e:
                self._node.get_logger().warn(
                    f'[MARKUP] Failed to destroy client for {name}: {e}'
                )
        self._clients.clear()
        self._definitions.clear()

    @staticmethod
    def _load_interface_class(type_str: str, kind: str) -> Any:
        """Dynamically load a ROS2 interface class from its type string."""
        if kind == 'msg':
            return get_message(type_str)
        elif kind == 'action':
            return get_action(type_str)
        elif kind == 'srv':
            return get_service(type_str)
        raise ValueError(f'Unknown interface kind: {kind}')
