import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers

from pulsarlm.reply import ReplyDecoder

SPECIAL = ["<think>", "</think>", "<|channel>", "<channel|>", "<eos>"]
WORDS = SPECIAL + ["thought", "a", "b", "c", "\n", " "]


def tokenizer() -> Tokenizer:
    tok = Tokenizer(
        models.WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="a")
    )
    tok.pre_tokenizer = pre_tokenizers.Split("", "isolated")
    tok.decoder = decoders.Fuse()
    tok.add_special_tokens(SPECIAL)
    return tok


def decode(words: list[str]) -> list[tuple[str, str]]:
    tok = tokenizer()
    decoder = ReplyDecoder(tok, [WORDS.index("<eos>")])
    parts = [decoder.step(WORDS.index(word)) for word in words]
    return [(part, text) for part, text in parts if text]


@pytest.mark.parametrize(
    ("opening", "label", "closing"),
    [("<think>", ["\n"], "</think>"), ("<|channel>", ["thought", "\n"], "<channel|>")],
)
def test_a_think_block_is_reasoning_without_its_markers_or_label(
    opening, label, closing
):
    words = [opening, *label, "a", " ", "b", closing, "\n", "\n", "c", "<eos>"]
    assert decode(words) == [
        ("reasoning", "a"),
        ("reasoning", " "),
        ("reasoning", "b"),
        ("content", "c"),
    ]


def test_reasoning_not_starting_with_the_label_keeps_its_text():
    assert decode(["<|channel>", "a", "<channel|>", "b"]) == [
        ("reasoning", "a"),
        ("content", "b"),
    ]


def test_a_reply_without_a_think_block_is_content():
    assert decode(["a", " ", "b", "<eos>"]) == [
        ("content", "a"),
        ("content", " "),
        ("content", "b"),
    ]
