"""ChatML prompts, as Qwen2 and Qwen3 are trained on."""

from typing import Literal

from tokenizers import Tokenizer

type Role = Literal["system", "user", "assistant"]

END_OF_TEXT = "<|endoftext|>"
REPLY_OPENING = "<|im_start|>assistant\n"


def stop_ids(tokenizer: Tokenizer, eos_ids: list[int]) -> list[int]:
    """The model's eos tokens and the tokenizer's end of text."""
    end_of_text = tokenizer.token_to_id(END_OF_TEXT)
    if end_of_text is None or end_of_text in eos_ids:
        return eos_ids
    return eos_ids + [end_of_text]


def render_turn(role: Role, content: str) -> str:
    return f"<|im_start|>{role}\n{content}<|im_end|>\n"


def encode_prompt(
    tokenizer: Tokenizer, messages: list[tuple[Role, str]]
) -> tuple[list[int], int]:
    """The tokens of the messages and the opening of the assistant's reply, and
    how many of them the leading system message takes."""
    turns = [render_turn(role, content) for role, content in messages]
    prompt = tokenizer.encode("".join(turns) + REPLY_OPENING).ids
    if messages and messages[0][0] == "system":
        return prompt, len(tokenizer.encode(turns[0]).ids)
    return prompt, 0
