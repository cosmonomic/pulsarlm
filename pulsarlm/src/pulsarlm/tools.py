"""Tool call bodies in the syntaxes chat templates render them in."""

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

type ToolCallSyntax = Literal["qwen", "gemma"]

GEMMA_QUOTE = '<|"|>'


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]


class ToolCallError(ValueError):
    pass


def parse_tool_call(syntax: ToolCallSyntax, body: str) -> ToolCall:
    """Raises ToolCallError when body is not a call in syntax. A qwen call is a
    JSON object (Qwen2, Qwen3) or XML-like <function=NAME> tags (Qwen3.5)."""
    if syntax == "qwen":
        if body.lstrip().startswith("<function="):
            return parse_function_call(body)
        return parse_json_call(body)
    return GemmaCallParser(body).call()


def parse_json_call(body: str) -> ToolCall:
    try:
        call = json.loads(body)
    except json.JSONDecodeError as error:
        raise ToolCallError(f"not JSON: {error}") from error
    if not isinstance(call, dict) or not isinstance(call.get("name"), str):
        raise ToolCallError("a call is an object with a string name")
    arguments = call.get("arguments", {})
    if not isinstance(arguments, dict):
        raise ToolCallError("arguments must be an object")
    return ToolCall(call["name"], arguments)


FUNCTION_CALL = re.compile(r"\s*<function=([^>\n]+)>(.*)</function>\s*", re.DOTALL)
PARAMETER = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.DOTALL)


def parse_function_call(body: str) -> ToolCall:
    """<function=NAME> then a <parameter=KEY> block per argument, as Qwen3.5's
    template renders them: a value that parses as JSON other than a string is
    that value, any other is its text."""
    call = FUNCTION_CALL.fullmatch(body)
    if call is None:
        raise ToolCallError("a call is <function=NAME>...</function>")
    name, inner = call.groups()
    arguments: dict[str, Any] = {}
    end = 0
    for parameter in PARAMETER.finditer(inner):
        if inner[end : parameter.start()].strip():
            raise ToolCallError(
                f"text between parameters: {inner[end : parameter.start()]!r}"
            )
        key, text = parameter.groups()
        arguments[key] = parameter_value(text)
        end = parameter.end()
    if inner[end:].strip():
        raise ToolCallError(f"text after the parameters: {inner[end:]!r}")
    return ToolCall(name.strip(), arguments)


def parameter_value(text: str) -> Any:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return text
    return text if isinstance(value, str) else value


class GemmaCallParser:
    """call:NAME{key:value,...}, values as Gemma 4's template formats them: strings
    between <|"|> quotes, mappings, sequences, true, false, null and numbers."""

    def __init__(self, body: str) -> None:
        self._text = body.strip()
        self._at = 0

    def call(self) -> ToolCall:
        self._expect("call:")
        brace = self._text.find("{", self._at)
        if brace <= self._at:
            raise ToolCallError("a call is call:NAME{...}")
        name = self._text[self._at : brace]
        self._at = brace
        arguments = self._mapping()
        if self._at != len(self._text):
            raise ToolCallError(f"text after the call: {self._text[self._at :]!r}")
        return ToolCall(name, arguments)

    def _value(self) -> Any:
        if self._text.startswith(GEMMA_QUOTE, self._at):
            return self._string()
        if self._text.startswith("{", self._at):
            return self._mapping()
        if self._text.startswith("[", self._at):
            return self._sequence()
        end = self._at
        while end < len(self._text) and self._text[end] not in ",}]":
            end += 1
        word = self._text[self._at : end].strip()
        self._at = end
        literals = {"true": True, "false": False, "null": None}
        if word in literals:
            return literals[word]
        try:
            return json.loads(word)
        except json.JSONDecodeError as error:
            raise ToolCallError(f"not a value: {word!r}") from error

    def _string(self) -> str:
        self._expect(GEMMA_QUOTE)
        end = self._text.find(GEMMA_QUOTE, self._at)
        if end < 0:
            raise ToolCallError("unterminated string")
        value = self._text[self._at : end]
        self._at = end + len(GEMMA_QUOTE)
        return value

    def _mapping(self) -> dict[str, Any]:
        self._expect("{")
        mapping: dict[str, Any] = {}
        while not self._skip("}"):
            if mapping:
                self._expect(",")
            if self._text.startswith(GEMMA_QUOTE, self._at):
                key = self._string()
            else:
                colon = self._text.find(":", self._at)
                if colon < 0:
                    raise ToolCallError("a key ends with ':'")
                key = self._text[self._at : colon].strip()
                self._at = colon
            self._expect(":")
            mapping[key] = self._value()
        return mapping

    def _sequence(self) -> list[Any]:
        self._expect("[")
        items: list[Any] = []
        while not self._skip("]"):
            if items:
                self._expect(",")
            items.append(self._value())
        return items

    def _skip(self, text: str) -> bool:
        if self._text.startswith(text, self._at):
            self._at += len(text)
            return True
        if self._at >= len(self._text):
            raise ToolCallError(f"expected {text!r} at the end")
        return False

    def _expect(self, text: str) -> None:
        if not self._text.startswith(text, self._at):
            raise ToolCallError(f"expected {text!r} at {self._text[self._at :]!r}")
        self._at += len(text)
