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

"""AST node definitions for multi-modal expression markup."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union


@dataclass
class VariableRef:
    """A variable reference inside an action, e.g. @user.name|"fallback"."""

    query: tuple[str, ...]
    default: object = None


@dataclass
class TextSegment:
    """Plain text to be sent to TTS."""

    text: str


@dataclass
class VariableText:
    """Variable text in default mode: @query|"default"."""

    query: tuple[str, ...]
    default: str


@dataclass
class MarkupAction:
    """A markup action like <set expression(happy)>."""

    verb: str
    name: str
    positional_args: tuple = ()
    keyword_args: dict = field(default_factory=dict)
    timeout: float | None = None


@dataclass
class PauseAction:
    """A <pause(N)> built-in action."""

    duration: float


Segment = Union[TextSegment, VariableText, MarkupAction, PauseAction]


@dataclass
class Expression:
    """A parsed multi-modal expression."""

    segments: tuple[Segment, ...]
