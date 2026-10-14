from collections.abc import Sequence
from typing import Literal, NamedTuple

from tokenizers import Tokenizer
from tokenizers.decoders import DecodeStream

from pulsarlm.tools import ToolCallSyntax

type Part = Literal["reasoning", "content"]


class ThinkMarker(NamedTuple):
    opening: str
    label: str
    closing: str


THINK_MARKERS = (
    ThinkMarker("<think>", "\n", "</think>"),
    ThinkMarker("<|channel>", "thought\n", "<channel|>"),
)


class ToolCallMarker(NamedTuple):
    opening: str
    closing: str
    syntax: ToolCallSyntax


TOOL_CALL_MARKERS = (
    ToolCallMarker("<tool_call>", "</tool_call>", "json"),
    ToolCallMarker("<|tool_call>", "<tool_call|>", "gemma"),
)


def tool_call_marker(tokenizer: Tokenizer) -> ToolCallMarker | None:
    for marker in TOOL_CALL_MARKERS:
        if (
            tokenizer.token_to_id(marker.opening) is not None
            and tokenizer.token_to_id(marker.closing) is not None
        ):
            return marker
    return None


class ReplyDecoder:
    """Decodes sampled tokens into reasoning and content text. Stop tokens, think
    markers and the label text after an opening marker decode to nothing. With
    tool_calls, the text between tool call markers decodes to nothing and is
    collected in tool_call_bodies instead. A prompt ending inside a think block
    starts the reply in it."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        stop_ids: list[int],
        *,
        tool_calls: bool = False,
        prompt: Sequence[int] = (),
    ) -> None:
        self._tokenizer = tokenizer
        self._stop_ids = stop_ids
        self._decoder = DecodeStream(skip_special_tokens=False)
        self._labels: dict[int, str] = {}
        self._closing_ids: set[int] = set()
        for marker in THINK_MARKERS:
            opening_id = tokenizer.token_to_id(marker.opening)
            closing_id = tokenizer.token_to_id(marker.closing)
            if opening_id is not None and closing_id is not None:
                self._labels[opening_id] = marker.label
                self._closing_ids.add(closing_id)
        marker = tool_call_marker(tokenizer) if tool_calls else None
        self.tool_call_marker = marker
        self._call_opening_id = (
            None if marker is None else tokenizer.token_to_id(marker.opening)
        )
        self._call_closing_id = (
            None if marker is None else tokenizer.token_to_id(marker.closing)
        )
        self.tool_call_bodies: list[str] = []
        self._calling = False
        self._thinking = False
        self._label = ""
        self._label_seen: str | None = None
        self._content_started = False
        self._resume_think_block(prompt)

    def _resume_think_block(self, prompt: Sequence[int]) -> None:
        opening = max(
            (i for i, token in enumerate(prompt) if token in self._labels), default=-1
        )
        if opening < 0 or any(t in self._closing_ids for t in prompt[opening + 1 :]):
            return
        self._thinking = True
        self._label = self._labels[prompt[opening]]
        after = self._tokenizer.decode(
            list(prompt[opening + 1 :]), skip_special_tokens=False
        )
        partial = self._label.startswith(after) and after != self._label
        self._label_seen = after if partial else None

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
        if token in self._labels:
            self._thinking = True
            self._label = self._labels[token]
            self._label_seen = ""
            return "reasoning", ""
        if token in self._closing_ids:
            self._thinking = False
            self._label_seen = None
            return "reasoning", ""
        if self._thinking:
            return "reasoning", self._without_label(text)
        if not self._content_started:
            text = text.lstrip()
            self._content_started = bool(text)
        return "content", text

    def _without_label(self, text: str) -> str:
        if self._label_seen is None:
            return text
        self._label_seen += text
        if self._label.startswith(self._label_seen):
            return ""
        text = self._label_seen.removeprefix(self._label)
        self._label_seen = None
        return text
