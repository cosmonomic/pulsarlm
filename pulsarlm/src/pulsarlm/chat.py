"""Chat prompts from the checkpoint's own Jinja chat template, rendered as
transformers renders them."""

from datetime import datetime
from typing import Literal, NoReturn

from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

type Role = Literal["system", "user", "assistant"]
type ReasoningEffort = Literal["none", "minimal", "low", "medium", "high"]

END_OF_TEXT = "<|endoftext|>"
SPECIAL_TOKENS = ("bos_token", "eos_token", "unk_token", "pad_token")


def stop_ids(tokenizer: Tokenizer, eos_ids: list[int]) -> list[int]:
    """The model's eos tokens and the tokenizer's end of text."""
    end_of_text = tokenizer.token_to_id(END_OF_TEXT)
    if end_of_text is None or end_of_text in eos_ids:
        return eos_ids
    return eos_ids + [end_of_text]


def raise_exception(message: str) -> NoReturn:
    raise ValueError(message)


def strftime_now(format: str) -> str:
    return datetime.now().strftime(format)


def special_token(value: str | dict | None) -> str | None:
    return value["content"] if isinstance(value, dict) else value


class ChatTemplate:
    def __init__(self, source: str, tokenizer_config: dict) -> None:
        environment = ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols]
        )
        environment.globals["raise_exception"] = raise_exception
        environment.globals["strftime_now"] = strftime_now
        self._template = environment.from_string(source)
        self._special_tokens = {
            name: special_token(tokenizer_config.get(name)) for name in SPECIAL_TOKENS
        }

    def render(
        self,
        messages: list[tuple[Role, str]],
        *,
        add_generation_prompt: bool,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> str:
        """reasoning_effort None leaves reasoning_effort and enable_thinking
        undefined, so the template's own defaults apply."""
        thinking = (
            {}
            if reasoning_effort is None
            else {
                "reasoning_effort": reasoning_effort,
                "enable_thinking": reasoning_effort != "none",
            }
        )
        return self._template.render(
            messages=[{"role": role, "content": content} for role, content in messages],
            add_generation_prompt=add_generation_prompt,
            **thinking,
            **self._special_tokens,
        )


def encode_prompt(
    tokenizer: Tokenizer,
    template: ChatTemplate,
    messages: list[tuple[Role, str]],
    *,
    reasoning_effort: ReasoningEffort | None = None,
) -> tuple[list[int], int]:
    """The tokens of the messages and the opening of the assistant's reply, and
    how many of them the leading system message takes."""
    prompt = template.render(
        messages, add_generation_prompt=True, reasoning_effort=reasoning_effort
    )
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    if messages and messages[0][0] == "system":
        system = template.render(
            messages[:1], add_generation_prompt=False, reasoning_effort=reasoning_effort
        )
        return ids, len(tokenizer.encode(system, add_special_tokens=False).ids)
    return ids, 0
