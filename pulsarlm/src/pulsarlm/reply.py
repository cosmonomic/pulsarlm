from collections.abc import Sequence
from typing import Literal, NamedTuple

from tokenizers import Tokenizer
from tokenizers.decoders import DecodeStream

from pulsarlm.tools import ToolCallSyntax

type Part = Literal["reasoning", "content"]


type ThinkSyntax = Literal["qwen", "gemma"]


class ThinkMarker(NamedTuple):
    opening: str
    label: str
    closing: str


THINK_MARKERS: dict[ThinkSyntax, ThinkMarker] = {
    "qwen": ThinkMarker("<think>", "\n", "</think>"),
    "gemma": ThinkMarker("<|channel>", "thought\n", "<channel|>"),
}


class ToolCallMarker(NamedTuple):
    opening: str
    closing: str


TOOL_CALL_MARKERS: dict[ToolCallSyntax, ToolCallMarker] = {
    "qwen_json": ToolCallMarker("<tool_call>", "</tool_call>"),
    "qwen_function": ToolCallMarker("<tool_call>", "</tool_call>"),
    "gemma": ToolCallMarker("<|tool_call>", "<tool_call|>"),
}


def marker_id(tokenizer: Tokenizer, marker: str) -> int:
    token_id = tokenizer.token_to_id(marker)
    if token_id is None:
        raise ValueError(f"the tokenizer has no {marker} token")
    return token_id


class ReplyDecoder:
    """Decodes sampled tokens into reasoning and content text. Stop tokens, think
    markers and the label text after an opening marker decode to nothing. With
    a tool_call_syntax, the text between tool call markers decodes to nothing and
    is collected in tool_call_bodies instead. A prompt ending inside a think
    block starts the reply in it."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        stop_ids: list[int],
        *,
        think_syntax: ThinkSyntax | None,
        tool_call_syntax: ToolCallSyntax | None = None,
        prompt: Sequence[int] = (),
    ) -> None:
        self._tokenizer = tokenizer
        self._stop_ids = stop_ids
        self._decoder = DecodeStream(skip_special_tokens=False)
        think = None if think_syntax is None else THINK_MARKERS[think_syntax]
        self._label = "" if think is None else think.label
        self._think_opening_id = (
            None if think is None else marker_id(tokenizer, think.opening)
        )
        self._think_closing_id = (
            None if think is None else marker_id(tokenizer, think.closing)
        )
        self.tool_call_syntax: ToolCallSyntax | None = tool_call_syntax
        call = None if tool_call_syntax is None else TOOL_CALL_MARKERS[tool_call_syntax]
        self._call_opening_id = (
            None if call is None else marker_id(tokenizer, call.opening)
        )
        self._call_closing_id = (
            None if call is None else marker_id(tokenizer, call.closing)
        )
        self.tool_call_bodies: list[str] = []
        self._calling = False
        self._thinking = False
        self._seen_label: str | None = None
        self._content_started = False
        self._resume_think_block(prompt)

    def _resume_think_block(self, prompt: Sequence[int]) -> None:
        if self._think_opening_id is None:
            return
        opening = max(
            (i for i, token in enumerate(prompt) if token == self._think_opening_id),
            default=-1,
        )
        if opening < 0 or self._think_closing_id in prompt[opening + 1 :]:
            return
        self._thinking = True
        after = self._tokenizer.decode(
            list(prompt[opening + 1 :]), skip_special_tokens=False
        )
        partial = self._label.startswith(after) and after != self._label
        self._seen_label = after if partial else None

    def step(self, token: int) -> tuple[Part, str]:
        if token in self._stop_ids:
            return "content", ""
        text = self._decoder.step(self._tokenizer, token) or ""
        if token == self._call_opening_id:
            self._calling = True
            self.tool_call_bodies.append("")
            return "content", ""
        if token == self._call_closing_id:
            self._calling = False
            return "content", ""
        if self._calling:
            self.tool_call_bodies[-1] += text
            return "content", ""
        if token == self._think_opening_id:
            self._thinking = True
            self._seen_label = ""
            return "reasoning", ""
        if token == self._think_closing_id:
            self._thinking = False
            self._seen_label = None
            return "reasoning", ""
        if self._thinking:
            return "reasoning", self._without_label(text)
        if not self._content_started:
            text = text.lstrip()
            self._content_started = bool(text)
        return "content", text

    def _without_label(self, text: str) -> str:
        if self._seen_label is None:
            return text
        self._seen_label += text
        if self._label.startswith(self._seen_label):
            return ""
        text = self._seen_label.removeprefix(self._label)
        self._seen_label = None
        return text
