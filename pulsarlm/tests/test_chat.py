from datetime import datetime

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from pulsarlm.chat import ChatTemplate, encode_prompt

TURNS = (
    "{{ bos_token }}{% for m in messages %}<|turn>{{ m.role }}\n{{ m.content | trim }}"
    "<turn|>\n{% endfor %}{% if add_generation_prompt %}<|turn>model\n{% endif %}"
)
SPECIAL = ["<bos>", "<|turn>", "<turn|>"]


def tokenizer() -> Tokenizer:
    words = SPECIAL + ["system", "user", "model", "a", "b", "c"]
    tok = Tokenizer(
        models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="a")
    )
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens(SPECIAL)
    return tok


def test_the_template_takes_the_tokenizer_configs_special_tokens():
    template = ChatTemplate(TURNS, {"bos_token": {"content": "<bos>"}})
    rendered = template.render(
        [{"role": "user", "content": " a "}], add_generation_prompt=True
    )
    assert rendered == "<bos><|turn>user\na<turn|>\n<|turn>model\n"


def test_the_system_message_counts_the_tokens_it_renders_to():
    tok = tokenizer()
    template = ChatTemplate(TURNS, {"bos_token": "<bos>"})
    ids, system_tokens = encode_prompt(
        tok,
        template,
        [{"role": "system", "content": "a b"}, {"role": "user", "content": "c"}],
    )
    assert system_tokens == len(tok.encode("<bos><|turn>system\na b<turn|>\n").ids)
    assert ids[0] == SPECIAL.index("<bos>")
    assert ids.count(SPECIAL.index("<bos>")) == 1


def test_a_prompt_without_a_system_message_sinks_nothing():
    _, system_tokens = encode_prompt(
        tokenizer(),
        ChatTemplate(TURNS, {"bos_token": "<bos>"}),
        [{"role": "user", "content": "a"}],
    )
    assert system_tokens == 0


THINKING = (
    "{{ reasoning_effort if reasoning_effort is defined else 'undefined' }} "
    "{{ enable_thinking if enable_thinking is defined else 'undefined' }}"
)


@pytest.mark.parametrize(
    ("effort", "rendered"),
    [(None, "undefined undefined"), ("none", "none False"), ("low", "low True")],
)
def test_reasoning_effort_reaches_the_template_with_enable_thinking(effort, rendered):
    template = ChatTemplate(THINKING, {})
    assert (
        template.render([], add_generation_prompt=False, reasoning_effort=effort)
        == rendered
    )


def test_raise_exception_in_a_template_raises():
    template = ChatTemplate("{{ raise_exception('no') }}", {})
    with pytest.raises(ValueError, match="no"):
        template.render([], add_generation_prompt=False)


def test_the_template_can_read_the_date():
    template = ChatTemplate("{{ strftime_now('%Y') }}", {})
    rendered = template.render([], add_generation_prompt=False)
    assert rendered == str(datetime.now().year)
