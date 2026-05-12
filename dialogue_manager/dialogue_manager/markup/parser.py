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

"""Hand-written parser for multi-modal expression markup."""

from __future__ import annotations

from dataclasses import dataclass
import re

from .ast_nodes import (
    Expression,
    MarkupAction,
    PauseAction,
    Segment,
    TextSegment,
    VariableRef,
    VariableText,
)


class MarkupParseError(Exception):
    """Raised when markup input cannot be parsed."""


def parse_expression(text: str) -> Expression:
    """Parse a multi-modal expression string into an AST."""
    return _ExpressionParser(text).parse()


def extract_plain_text(expression: Expression) -> str:
    """Extract plain text from a parsed expression, stripping all markup."""
    parts: list[str] = []
    for seg in expression.segments:
        if isinstance(seg, TextSegment):
            parts.append(seg.text)
        elif isinstance(seg, VariableText):
            parts.append(seg.default)
    return ''.join(parts)


# ---------------------------------------------------------------------------
# Token types (action mode only, internal)
# ---------------------------------------------------------------------------

_T_DO = 'DO'
_T_FALSE = 'FALSE'
_T_NULL = 'NULL'
_T_PAUSE = 'PAUSE'
_T_SET = 'SET'
_T_START = 'START'
_T_STOP = 'STOP'
_T_TIMEOUT = 'TIMEOUT'
_T_TRUE = 'TRUE'
_T_TTS = 'TTS'
_T_WAIT = 'WAIT'
_T_LPAREN = 'LPAREN'
_T_RPAREN = 'RPAREN'
_T_LBRACE = 'LBRACE'
_T_RBRACE = 'RBRACE'
_T_DOT = 'DOT'
_T_COMMA = 'COMMA'
_T_ASSIGN = 'ASSIGN'
_T_AT = 'AT'
_T_COLUMN = 'COLUMN'
_T_STRING = 'STRING'
_T_NUMBER = 'NUMBER'
_T_ID = 'ID'
_T_EOF = 'EOF'

_KEYWORDS: dict[str, str] = {
    'do': _T_DO, 'false': _T_FALSE, 'False': _T_FALSE,
    'null': _T_NULL, 'pause': _T_PAUSE, 'set': _T_SET,
    'start': _T_START, 'stop': _T_STOP, 'timeout': _T_TIMEOUT,
    'true': _T_TRUE, 'True': _T_TRUE, 'tts': _T_TTS, 'wait': _T_WAIT,
}

_PUNCT: dict[str, str] = {
    '(': _T_LPAREN, ')': _T_RPAREN, '[': _T_LBRACE, ']': _T_RBRACE,
    ',': _T_COMMA, '=': _T_ASSIGN, '@': _T_AT, '|': _T_COLUMN,
}

_START_VERBS = {_T_SET, _T_START, _T_DO}
_STOP_VERBS = {_T_STOP, _T_WAIT}
_IDENT_TYPES = {_T_ID, _T_TIMEOUT, _T_TTS}
_STRING_TYPES = (
    {_T_STRING, _T_ID, _T_TIMEOUT, _T_TTS}
    | _START_VERBS | _STOP_VERBS
)

_NUMBER_RE = re.compile(
    r'[+-]?(?:'
    r'0[xX][0-9a-fA-F]+'
    r'|(?:0|[1-9][0-9]*)(?:\.[0-9]*)?(?:[eE][+-]?[0-9]*)?'
    r'|\.[0-9]+(?:[eE][+-]?[0-9]*)?'
    r')'
)


@dataclass
class _Token:
    kind: str
    value: object
    pos: int


# ---------------------------------------------------------------------------
# Top-level expression parser (default mode)
# ---------------------------------------------------------------------------

class _ExpressionParser:
    """Scan default-mode text, delegating <actions> and @variables."""

    def __init__(self, text: str):
        self._text = text
        self._pos = 0
        self._len = len(text)

    def parse(self) -> Expression:
        """Parse the full input into an Expression."""
        segments: list[Segment] = []
        while self._pos < self._len:
            ch = self._text[self._pos]
            if ch == '<':
                segments.append(self._parse_action_block())
            elif ch == '@':
                segments.append(self._parse_variable_text())
            else:
                segments.append(self._parse_text())
        return Expression(segments=tuple(segments))

    def _parse_text(self) -> TextSegment:
        start = self._pos
        while self._pos < self._len and self._text[self._pos] not in '<@':
            self._pos += 1
        return TextSegment(text=self._text[start:self._pos])

    def _parse_variable_text(self) -> VariableText:
        self._pos += 1  # skip '@'
        query = self._read_query()
        if self._pos >= self._len or self._text[self._pos] != '|':
            raise MarkupParseError(
                f"Expected '|' after variable name at position {self._pos}"
            )
        self._pos += 1  # skip '|'
        default = self._read_quoted_string()
        return VariableText(query=tuple(query), default=default)

    def _read_query(self) -> list[str]:
        name = self._read_identifier()
        if not name:
            raise MarkupParseError(
                f'Expected identifier at position {self._pos}'
            )
        parts = [name]
        if self._pos < self._len and self._text[self._pos] == '.':
            self._pos += 1
            name2 = self._read_identifier()
            if not name2:
                raise MarkupParseError(
                    f"Expected identifier after '.' at position {self._pos}"
                )
            parts.append(name2)
        return parts

    def _read_identifier(self) -> str:
        start = self._pos
        if self._pos < self._len and self._text[self._pos].isalpha():
            self._pos += 1
            while self._pos < self._len and (
                self._text[self._pos].isalnum() or self._text[self._pos] == '_'
            ):
                self._pos += 1
        return self._text[start:self._pos]

    def _read_quoted_string(self) -> str:
        if self._pos >= self._len or self._text[self._pos] not in '"\'':
            raise MarkupParseError(
                f'Expected quoted string at position {self._pos}'
            )
        quote = self._text[self._pos]
        self._pos += 1
        chars: list[str] = []
        while self._pos < self._len:
            ch = self._text[self._pos]
            if ch == '\\' and self._pos + 1 < self._len:
                if self._text[self._pos + 1] == quote:
                    chars.append(quote)
                    self._pos += 2
                    continue
            if ch == quote:
                self._pos += 1
                return ''.join(chars)
            chars.append(ch)
            self._pos += 1
        raise MarkupParseError('Unterminated string')

    def _parse_action_block(self) -> MarkupAction | PauseAction:
        block_start = self._pos
        self._pos += 1  # skip '<'

        # Scan for matching '>', respecting quoted strings
        content_start = self._pos
        in_string: str | None = None
        while self._pos < self._len:
            ch = self._text[self._pos]
            if in_string:
                if ch == '\\' and self._pos + 1 < self._len:
                    self._pos += 2
                    continue
                if ch == in_string:
                    in_string = None
            else:
                if ch in '"\'':
                    in_string = ch
                elif ch == '>':
                    content = self._text[content_start:self._pos]
                    self._pos += 1  # skip '>'
                    return _parse_action_content(content, block_start + 1)
            self._pos += 1

        raise MarkupParseError(
            f'Unterminated action starting at position {block_start}'
        )


# ---------------------------------------------------------------------------
# Action content tokenizer
# ---------------------------------------------------------------------------

def _parse_number_literal(s: str) -> int | float:
    """Convert a number string to int or float."""
    stripped = s.lstrip('+-')
    if stripped.startswith(('0x', '0X')):
        return int(s, 16)
    if '.' in s or 'e' in s.lower():
        return float(s)
    return int(s)


def _read_action_string(content: str, pos: int) -> tuple[str, int]:
    """Read a quoted string from action content."""
    quote = content[pos]
    pos += 1
    chars: list[str] = []
    while pos < len(content):
        ch = content[pos]
        if ch == '\\' and pos + 1 < len(content):
            if content[pos + 1] == quote:
                chars.append(quote)
                pos += 2
                continue
        if ch == quote:
            return ''.join(chars), pos + 1
        chars.append(ch)
        pos += 1
    raise MarkupParseError('Unterminated string in action')


def _tokenize_action(content: str, offset: int) -> list[_Token]:
    """Tokenize the text between < and > into a list of action-mode tokens."""
    tokens: list[_Token] = []
    pos = 0
    length = len(content)

    while pos < length:
        ch = content[pos]

        if ch.isspace():
            pos += 1
            continue

        # Single-char punctuation
        if ch in _PUNCT:
            tokens.append(_Token(_PUNCT[ch], ch, offset + pos))
            pos += 1
            continue

        # Dot: DOT or start of number like .5
        if ch == '.':
            if pos + 1 < length and content[pos + 1].isdigit():
                m = _NUMBER_RE.match(content, pos)
                if m:
                    tokens.append(_Token(
                        _T_NUMBER, _parse_number_literal(m.group()),
                        offset + pos,
                    ))
                    pos = m.end()
                    continue
            tokens.append(_Token(_T_DOT, '.', offset + pos))
            pos += 1
            continue

        # Sign or digit -> number
        if ch in '+-' or ch.isdigit():
            m = _NUMBER_RE.match(content, pos)
            if m:
                tokens.append(_Token(
                    _T_NUMBER, _parse_number_literal(m.group()),
                    offset + pos,
                ))
                pos = m.end()
                continue
            if ch in '+-':
                raise MarkupParseError(
                    f"Invalid character '{ch}' at position {offset + pos}"
                )

        # Quoted string
        if ch in '"\'':
            val, end = _read_action_string(content, pos)
            tokens.append(_Token(_T_STRING, val, offset + pos))
            pos = end
            continue

        # Identifier or keyword
        if ch.isalpha():
            start = pos
            pos += 1
            while pos < length and (content[pos].isalnum() or content[pos] == '_'):
                pos += 1
            word = content[start:pos]
            tok_type = _KEYWORDS.get(word, _T_ID)
            tokens.append(_Token(tok_type, word, offset + start))
            continue

        raise MarkupParseError(
            f"Unexpected character '{ch}' at position {offset + pos}"
        )

    tokens.append(_Token(_T_EOF, '', offset + length))
    return tokens


# ---------------------------------------------------------------------------
# Action parser (recursive descent over token list)
# ---------------------------------------------------------------------------

def _parse_action_content(
    content: str, offset: int
) -> MarkupAction | PauseAction:
    """Tokenize and parse action content into an AST node."""
    tokens = _tokenize_action(content, offset)
    return _ActionParser(tokens).parse()


class _ActionParser:
    """Recursive descent parser for action content."""

    def __init__(self, tokens: list[_Token]):
        self._tokens = tokens
        self._pos = 0

    def _peek(self) -> str:
        """Return the type of the current token."""
        return self._tokens[self._pos].kind

    def _peek_at(self, offset: int) -> str:
        """Return the kind of a token at an offset from current position."""
        idx = self._pos + offset
        if idx < len(self._tokens):
            return self._tokens[idx].kind
        return _T_EOF

    def _advance(self) -> _Token:
        """Consume and return the current token."""
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def _expect(self, token_type: str) -> _Token:
        """Consume a token of the expected type, or raise."""
        tok = self._tokens[self._pos]
        if tok.kind != token_type:
            raise MarkupParseError(
                f'Expected {token_type}, got {tok.kind} '
                f"('{tok.value}') at position {tok.pos}"
            )
        self._pos += 1
        return tok

    # -- Top-level action parsing -------------------------------------------

    def parse(self) -> MarkupAction | PauseAction:
        """Parse the full action token sequence."""
        if self._peek() == _T_EOF:
            raise MarkupParseError('Empty action')

        if self._peek() == _T_PAUSE:
            return self._parse_pause()

        tok = self._tokens[self._pos]

        if tok.kind in _START_VERBS:
            verb = self._advance().value
            name = self._parse_ident()
            positional, keyword = self._parse_optional_arguments()
            timeout = self._parse_optional_timeout()
            self._expect(_T_EOF)
            return MarkupAction(
                verb=verb,
                name=name,
                positional_args=tuple(positional),
                keyword_args=dict(keyword),
                timeout=timeout,
            )

        if tok.kind in _STOP_VERBS:
            verb = self._advance().value
            name = self._parse_ident()
            timeout = self._parse_optional_timeout()
            self._expect(_T_EOF)
            return MarkupAction(verb=verb, name=name, timeout=timeout)

        raise MarkupParseError(
            f'Expected verb (set/start/do/stop/wait/pause), '
            f"got '{tok.value}' at position {tok.pos}"
        )

    def _parse_pause(self) -> PauseAction:
        self._advance()  # consume PAUSE
        self._expect(_T_LPAREN)
        num_tok = self._expect(_T_NUMBER)
        self._expect(_T_RPAREN)
        self._expect(_T_EOF)
        return PauseAction(duration=float(num_tok.value))

    def _parse_ident(self) -> str:
        """Parse ident: ID | TIMEOUT | TTS."""
        tok = self._tokens[self._pos]
        if tok.kind not in _IDENT_TYPES:
            raise MarkupParseError(
                f"Expected identifier, got '{tok.value}' at position {tok.pos}"
            )
        self._advance()
        return tok.value

    # -- Arguments ----------------------------------------------------------

    def _parse_optional_arguments(self) -> tuple[list, list[tuple]]:
        """Parse arguments if present (starts with '(')."""
        if self._peek() != _T_LPAREN:
            return [], []
        return self._parse_arguments()

    def _parse_arguments(self) -> tuple[list, list[tuple]]:
        """Parse '(' (positionalArgs (',' keywordArgs)? | keywordArgs)? ')'."""
        self._expect(_T_LPAREN)

        if self._peek() == _T_RPAREN:
            self._advance()
            return [], []

        positional: list = []
        keyword: list[tuple] = []
        in_keyword_mode = False

        while True:
            # Keyword arg: ident = value
            if (
                self._peek() in _IDENT_TYPES
                and self._peek_at(1) == _T_ASSIGN
            ):
                in_keyword_mode = True
                name_tok = self._advance()
                self._expect(_T_ASSIGN)
                value = self._parse_value()
                keyword.append((name_tok.value, value))
            elif in_keyword_mode:
                tok = self._tokens[self._pos]
                raise MarkupParseError(
                    f'Expected keyword argument, '
                    f"got '{tok.value}' at position {tok.pos}"
                )
            else:
                positional.append(self._parse_value())

            if self._peek() == _T_COMMA:
                self._advance()
                if self._peek() == _T_RPAREN:
                    tok = self._tokens[self._pos]
                    raise MarkupParseError(
                        f'Trailing comma in arguments at position {tok.pos}'
                    )
            elif self._peek() == _T_RPAREN:
                self._advance()
                break
            else:
                tok = self._tokens[self._pos]
                raise MarkupParseError(
                    f"Expected ',' or ')' at position {tok.pos}"
                )

        return positional, keyword

    def _parse_optional_timeout(self) -> float | None:
        """Parse timeout = NUMBER if present."""
        if self._peek() == _T_TIMEOUT:
            self._advance()
            self._expect(_T_ASSIGN)
            num_tok = self._expect(_T_NUMBER)
            return float(num_tok.value)
        return None

    # -- Values -------------------------------------------------------------

    def _parse_value(self) -> object:
        """Parse value: staticValue | variableValue."""
        if self._peek() == _T_AT:
            return self._parse_variable_value()
        return self._parse_static_value()

    def _parse_variable_value(self) -> VariableRef:
        """Parse @query (| staticValue)?."""
        self._advance()  # consume @
        query = self._parse_query()
        default = None
        if self._peek() == _T_COLUMN:
            self._advance()
            default = self._parse_static_value()
        return VariableRef(query=tuple(query), default=default)

    def _parse_query(self) -> list[str]:
        """Parse ident (. ident)?."""
        parts = [self._parse_ident()]
        if self._peek() == _T_DOT:
            self._advance()
            parts.append(self._parse_ident())
        return parts

    def _parse_static_value(self) -> object:
        """Parse array | literal | string | number."""
        tok_type = self._peek()

        if tok_type == _T_LBRACE:
            return self._parse_array()

        if tok_type == _T_TRUE:
            self._advance()
            return True

        if tok_type == _T_FALSE:
            self._advance()
            return False

        if tok_type == _T_NULL:
            self._advance()
            return None

        if tok_type == _T_NUMBER:
            return self._advance().value

        if tok_type in _STRING_TYPES:
            return self._advance().value

        tok = self._tokens[self._pos]
        raise MarkupParseError(
            f"Expected value, got '{tok.value}' at position {tok.pos}"
        )

    def _parse_array(self) -> list:
        """Parse '[' value (',' value)* ','? ']' | '[' ']'."""
        self._expect(_T_LBRACE)

        if self._peek() == _T_RBRACE:
            self._advance()
            return []

        values = [self._parse_value()]
        while self._peek() == _T_COMMA:
            self._advance()
            if self._peek() == _T_RBRACE:
                break  # trailing comma allowed in arrays
            values.append(self._parse_value())

        self._expect(_T_RBRACE)
        return values
