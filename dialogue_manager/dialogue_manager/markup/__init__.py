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

"""Multi-modal expression markup language: parser and executor."""

from .action_library import ActionLibrary  # noqa: F401
from .ast_nodes import (  # noqa: F401
    Expression,
    MarkupAction,
    PauseAction,
    Segment,
    TextSegment,
    VariableRef,
    VariableText,
)
from .executor import ExpressionExecutor  # noqa: F401
from .parser import (  # noqa: F401
    extract_plain_text,
    MarkupParseError,
    parse_expression,
)
