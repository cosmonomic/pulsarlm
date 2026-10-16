import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers

from pulsarlm.reply import ReplyDecoder

SPECIAL = ["<think>", "</think>", "<|channel>", "<channel|>", "<eos>"]
SPECIAL += ["<tool_call>", "</tool_call>"]
WORDS = SPECIAL + ["thought", "a", "b", "c", "\n", " "]


def tokenizer() -> Tokenizer:
    tok = Tokenizer(
        models.WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="a")
    )
    tok.pre_tokenizer = pre_tokenizers.Split("", "isolated")
    tok.decoder = decoders.Fuse()
    tok.add_special_tokens(SPECIAL)
    return tok


def decode(
    words: list[str], decoder: ReplyDecoder | None = None
) -> list[tuple[str, str]]:
    decoder = decoder or ReplyDecoder(tokenizer(), [WORDS.index("<eos>")])
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


def test_tool_call_bodies_are_collected_and_not_content():
    decoder = ReplyDecoder(tokenizer(), [WORDS.index("<eos>")], tool_calls=True)
    words = ["a", "<tool_call>", "b", " ", "c", "</tool_call>", "<tool_call>", "a"]
    assert decode(words + ["</tool_call>", "<eos>"], decoder) == [("content", "a")]
    assert decoder.tool_call_marker is not None
    assert decoder.tool_call_marker.syntax == "qwen"
    assert decoder.tool_call_bodies == ["b c", "a"]


def test_without_tool_calls_the_markers_are_content():
    decoder = ReplyDecoder(tokenizer(), [WORDS.index("<eos>")])
    assert "".join(text for _, text in decode(["<tool_call>", "b"], decoder)) == (
        "<tool_call>b"
    )
    assert decoder.tool_call_bodies == []


@pytest.mark.parametrize("label", [[], ["thought"], ["thought", "\n"]])
def test_a_prompt_ending_inside_a_think_block_starts_the_reply_in_it(label):
    tok = tokenizer()
    prompt = [WORDS.index(w) for w in ["a", "<|channel>", *label]]
    decoder = ReplyDecoder(tok, [WORDS.index("<eos>")], prompt=prompt)
    rest = ["thought", "\n"][len(label) :]
    assert decode([*rest, "a", "<channel|>", "b"], decoder) == [
        ("reasoning", "a"),
        ("content", "b"),
    ]


def test_a_prompt_with_a_closed_think_block_starts_the_reply_in_content():
    tok = tokenizer()
    prompt = [WORDS.index(w) for w in ["<|channel>", "thought", "\n", "<channel|>"]]
    decoder = ReplyDecoder(tok, [WORDS.index("<eos>")], prompt=prompt)
    assert decode(["a"], decoder) == [("content", "a")]
