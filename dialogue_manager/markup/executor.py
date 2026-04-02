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

"""Expression executor: walks an AST and dispatches TTS and ROS2 calls."""

from __future__ import annotations

import threading
import time
from typing import Any

from rclpy.action import ActionClient
from rclpy.node import Node
from rosidl_runtime_py import set_message_fields

from .action_library import (
    ActionLibrary,
    resolve_fields,
    resolve_variable,
)
from .ast_nodes import (
    Expression,
    MarkupAction,
    PauseAction,
    TextSegment,
    VariableRef,
    VariableText,
)
from .parser import extract_plain_text, MarkupParseError, parse_expression


class ExpressionExecutor:
    """Execute parsed multi-modal expressions."""

    def __init__(
        self,
        node: Node,
        tts_client: Any,  # TTSClient - avoid circular import
        action_library: ActionLibrary,
        expression_timeout: float = 60.0,
    ):
        """Initialize the executor."""
        self._node = node
        self._tts_client = tts_client
        self._action_library = action_library
        self._expression_timeout = expression_timeout

    def execute_text(
        self,
        text: str,
        priority: int = 128,
        variables: dict | None = None,
        cancel_event: threading.Event | None = None,
    ) -> bool:
        """Parse and execute a markup expression string (blocking)."""
        try:
            expression = parse_expression(text)
        except MarkupParseError as e:
            self._node.get_logger().warn(f'[MARKUP] Parse error: {e}')
            # Fallback: speak the raw text
            return self._tts_client.speak_and_wait(
                text, priority, cancel_event
            )

        return self.execute(expression, priority, variables, cancel_event)

    def execute(
        self,
        expression: Expression,
        priority: int = 128,
        variables: dict | None = None,
        cancel_event: threading.Event | None = None,
    ) -> bool:
        """Execute a parsed expression (blocking)."""
        if cancel_event is None:
            cancel_event = threading.Event()

        ctx = _ExecutionContext(
            tracked_actions={},
            cancel_event=cancel_event,
            priority=priority,
            variables=variables or {},
            deadline=time.monotonic() + self._expression_timeout,
        )

        try:
            # Group consecutive text/variable segments into TTS chunks,
            # then process each segment
            segments = list(expression.segments)
            i = 0
            while i < len(segments):
                if ctx.cancel_event.is_set():
                    self._cancel_all(ctx)
                    return False
                if time.monotonic() > ctx.deadline:
                    self._node.get_logger().warn(
                        '[MARKUP] Expression timeout exceeded'
                    )
                    self._cancel_all(ctx)
                    return False

                seg = segments[i]

                # Collect consecutive text/variable text for a single
                # TTS utterance
                if isinstance(seg, (TextSegment, VariableText)):
                    text_parts: list[str] = []
                    while i < len(segments) and isinstance(
                        segments[i], (TextSegment, VariableText)
                    ):
                        s = segments[i]
                        if isinstance(s, TextSegment):
                            text_parts.append(s.text)
                        else:
                            text_parts.append(
                                resolve_variable(
                                    VariableRef(s.query, s.default),
                                    ctx.variables,
                                )
                                or s.default
                            )
                        i += 1
                    tts_text = ''.join(text_parts)
                    if tts_text.strip():
                        if not self._speak(tts_text, ctx):
                            return False
                    continue

                if isinstance(seg, PauseAction):
                    if not self._pause(seg.duration, ctx):
                        return False
                    i += 1
                    continue

                if isinstance(seg, MarkupAction):
                    if not self._execute_action(seg, ctx):
                        return False
                    i += 1
                    continue

                i += 1

            # Implicit wait: wait for all remaining tracked actions
            if not self._wait_all_tracked(ctx):
                return False

            return True

        except Exception as e:
            self._node.get_logger().error(
                f'[MARKUP] Execution error: {e}'
            )
            self._cancel_all(ctx)
            return False

    def get_plain_text(self, text: str) -> str:
        """Extract plain text from a markup string for closed captions."""
        try:
            return extract_plain_text(parse_expression(text))
        except MarkupParseError:
            return text

    # -- Internal execution methods -----------------------------------------

    def _speak(self, text: str, ctx: _ExecutionContext) -> bool:
        """Send text to TTS and wait for completion."""
        return self._tts_client.speak_and_wait(
            text, ctx.priority, ctx.cancel_event
        )

    def _pause(self, duration: float, ctx: _ExecutionContext) -> bool:
        """Sleep for duration, checking for cancellation."""
        end = time.monotonic() + duration
        while time.monotonic() < end:
            if ctx.cancel_event.is_set():
                return False
            if time.monotonic() > ctx.deadline:
                return False
            time.sleep(min(0.05, end - time.monotonic()))
        return True

    def _execute_action(
        self, action: MarkupAction, ctx: _ExecutionContext
    ) -> bool:
        """Dispatch a markup action based on its verb."""
        defn = self._action_library.get_definition(action.name)
        if defn is None:
            self._node.get_logger().warn(
                f'[MARKUP] Unknown action: {action.name}'
            )
            return True  # Skip unknown actions, don't abort expression

        client = self._action_library.get_client(action.name)
        if client is None:
            self._node.get_logger().warn(
                f'[MARKUP] No client for action: {action.name}'
            )
            return True

        # Resolve positional/keyword args, including any VariableRefs
        pos_args = tuple(
            resolve_variable(a, ctx.variables) if isinstance(a, VariableRef)
            else a
            for a in action.positional_args
        )
        kw_args = {
            k: (resolve_variable(v, ctx.variables)
                if isinstance(v, VariableRef) else v)
            for k, v in action.keyword_args.items()
        }

        # Resolve field templates
        resolved = resolve_fields(
            defn.fields, pos_args, kw_args, ctx.variables
        )

        timeout = action.timeout or self._action_library.default_timeout

        verb = action.verb

        if verb == 'set':
            return self._dispatch_fire_and_forget(
                action.name, defn, client, resolved
            )
        elif verb == 'start':
            return self._dispatch_start(
                action.name, defn, client, resolved, ctx
            )
        elif verb == 'wait':
            return self._dispatch_wait(action.name, timeout, ctx)
        elif verb == 'stop':
            return self._dispatch_stop(action.name, ctx)
        elif verb == 'do':
            if not self._dispatch_start(
                action.name, defn, client, resolved, ctx
            ):
                return False
            return self._dispatch_wait(action.name, timeout, ctx)
        else:
            self._node.get_logger().warn(
                f'[MARKUP] Unknown verb: {verb}'
            )
            return True

    def _dispatch_fire_and_forget(
        self, name: str, defn, client, resolved: dict
    ) -> bool:
        """Fire-and-forget: publish topic or send action/service without tracking."""
        try:
            if defn.interface_kind == 'msg':
                msg = defn.interface_class()
                _set_fields(msg, resolved)
                client.publish(msg)
                self._node.get_logger().debug(
                    f'[MARKUP] Published {name} on {defn.path}'
                )
            elif defn.interface_kind == 'action':
                goal = defn.interface_class.Goal()
                _set_fields(goal, resolved)
                client.send_goal_async(goal)
                self._node.get_logger().debug(
                    f'[MARKUP] Sent fire-and-forget action goal: {name}'
                )
            elif defn.interface_kind == 'srv':
                req = defn.interface_class.Request()
                _set_fields(req, resolved)
                client.call_async(req)
                self._node.get_logger().debug(
                    f'[MARKUP] Sent fire-and-forget service call: {name}'
                )
        except Exception as e:
            self._node.get_logger().warn(
                f'[MARKUP] Failed to dispatch {name}: {e}'
            )
        return True

    def _dispatch_start(
        self, name: str, defn, client, resolved: dict,
        ctx: _ExecutionContext
    ) -> bool:
        """Start an action and track it."""
        if defn.interface_kind != 'action':
            # Topics and services can't be tracked; treat as fire-and-forget
            self._node.get_logger().debug(
                f'[MARKUP] {name} is {defn.interface_kind}, '
                f'treating start as set'
            )
            return self._dispatch_fire_and_forget(name, defn, client, resolved)

        try:
            if not isinstance(client, ActionClient):
                return True
            if not client.wait_for_server(timeout_sec=1.0):
                self._node.get_logger().warn(
                    f'[MARKUP] Action server not available: {name}'
                )
                return True

            goal = defn.interface_class.Goal()
            _set_fields(goal, resolved)

            send_future = client.send_goal_async(goal)

            # Wait for goal acceptance
            while not send_future.done():
                if ctx.cancel_event.is_set():
                    return False
                time.sleep(0.05)

            goal_handle = send_future.result()
            if not goal_handle or not goal_handle.accepted:
                self._node.get_logger().warn(
                    f'[MARKUP] Action goal rejected: {name}'
                )
                return True

            result_future = goal_handle.get_result_async()
            ctx.tracked_actions[name] = _TrackedAction(
                goal_handle=goal_handle,
                result_future=result_future,
            )
            self._node.get_logger().debug(
                f'[MARKUP] Started action: {name}'
            )
        except Exception as e:
            self._node.get_logger().warn(
                f'[MARKUP] Failed to start {name}: {e}'
            )
        return True

    def _dispatch_wait(
        self, name: str, timeout: float, ctx: _ExecutionContext
    ) -> bool:
        """Wait for a tracked action to complete."""
        tracked = ctx.tracked_actions.pop(name, None)
        if tracked is None:
            self._node.get_logger().debug(
                f'[MARKUP] No tracked action to wait for: {name}'
            )
            return True

        deadline = min(
            time.monotonic() + timeout,
            ctx.deadline,
        )

        while not tracked.result_future.done():
            if ctx.cancel_event.is_set():
                tracked.goal_handle.cancel_goal_async()
                return False
            if time.monotonic() > deadline:
                self._node.get_logger().warn(
                    f'[MARKUP] Timeout waiting for {name}'
                )
                tracked.goal_handle.cancel_goal_async()
                return True  # Timeout is not a fatal error
            time.sleep(0.05)

        self._node.get_logger().debug(f'[MARKUP] Action completed: {name}')
        return True

    def _dispatch_stop(
        self, name: str, ctx: _ExecutionContext
    ) -> bool:
        """Cancel a tracked action."""
        tracked = ctx.tracked_actions.pop(name, None)
        if tracked is None:
            self._node.get_logger().debug(
                f'[MARKUP] No tracked action to stop: {name}'
            )
            return True

        try:
            tracked.goal_handle.cancel_goal_async()
            self._node.get_logger().debug(f'[MARKUP] Stopped action: {name}')
        except Exception as e:
            self._node.get_logger().warn(
                f'[MARKUP] Failed to stop {name}: {e}'
            )
        return True

    def _wait_all_tracked(self, ctx: _ExecutionContext) -> bool:
        """Wait for all remaining tracked actions (implicit end-of-expression wait)."""
        for name in list(ctx.tracked_actions.keys()):
            timeout = self._action_library.default_timeout
            if not self._dispatch_wait(name, timeout, ctx):
                return False
        return True

    def _cancel_all(self, ctx: _ExecutionContext) -> None:
        """Cancel all tracked actions."""
        for name in list(ctx.tracked_actions.keys()):
            self._dispatch_stop(name, ctx)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _ExecutionContext:
    """Mutable state during expression execution."""

    __slots__ = (
        'tracked_actions', 'cancel_event', 'priority',
        'variables', 'deadline',
    )

    def __init__(
        self,
        tracked_actions: dict[str, _TrackedAction],
        cancel_event: threading.Event,
        priority: int,
        variables: dict,
        deadline: float,
    ):
        """Initialize execution context."""
        self.tracked_actions = tracked_actions
        self.cancel_event = cancel_event
        self.priority = priority
        self.variables = variables
        self.deadline = deadline


class _TrackedAction:
    """A started action whose completion we may wait for."""

    __slots__ = ('goal_handle', 'result_future')

    def __init__(self, goal_handle: Any, result_future: Any):
        """Initialize tracked action."""
        self.goal_handle = goal_handle
        self.result_future = result_future


def _set_fields(msg: Any, fields: dict) -> None:
    """Set fields on a ROS2 message, filtering out None values."""
    filtered = {k: v for k, v in fields.items() if v is not None}
    if filtered:
        try:
            set_message_fields(msg, filtered)
        except AttributeError as e:
            # Provide a helpful error when a field expects a nested message
            # but received a scalar (common misconfiguration in YAML).
            raise TypeError(
                f'Failed to set fields on {type(msg).__name__}: {e}. '
                f'This usually means the YAML action definition needs '
                f'nested fields to match the ROS2 message structure. '
                f'Resolved fields: {filtered}'
            ) from e
