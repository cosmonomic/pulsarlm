from typing import Literal, NamedTuple

from tokenizers import Tokenizer
from tokenizers.decoders import DecodeStream

type Part = Literal["reasoning", "content"]


class ThinkMarker(NamedTuple):
    opening: str
    label: str
    closing: str


THINK_MARKERS = (
    ThinkMarker("<think>", "\n", "</think>"),
    ThinkMarker("<|channel>", "thought\n", "<channel|>"),
)


class ReplyDecoder:
    """Decodes sampled tokens into reasoning and content text. Stop tokens, think
    markers and the label text after an opening marker decode to nothing."""

    def __init__(self, tokenizer: Tokenizer, stop_ids: list[int]) -> None:
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
        self._thinking = False
        self._label = ""
        self._label_seen: str | None = None
        self._content_started = False

    def step(self, token: int) -> tuple[Part, str]:
        if token in self._stop_ids:
            return "content", ""
        text = self._decoder.step(self._tokenizer, token) or ""
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
