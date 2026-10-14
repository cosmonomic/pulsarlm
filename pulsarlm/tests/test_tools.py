import pytest

from pulsarlm.tools import ToolCall, ToolCallError, parse_tool_call

Q = '<|"|>'


def test_a_json_call_has_a_name_and_arguments():
    body = '\n{"name": "read", "arguments": {"start": 0, "end": 10}}\n'
    assert parse_tool_call("json", body) == ToolCall("read", {"start": 0, "end": 10})


def test_a_json_call_without_arguments_takes_none():
    assert parse_tool_call("json", '{"name": "now"}') == ToolCall("now", {})


@pytest.mark.parametrize(
    "body", ["not json", '{"arguments": {}}', '{"name": "a", "arguments": [1]}']
)
def test_a_malformed_json_call_is_an_error(body):
    with pytest.raises(ToolCallError):
        parse_tool_call("json", body)


def test_a_gemma_call_reads_every_value_kind():
    body = (
        f"call:search{{query:{Q}a, b: {{c}}{Q},limit:5,ratio:0.5,exact:true,"
        f"filter:null,tags:[{Q}x{Q},2],where:{{{Q}key{Q}:{Q}v{Q}}}}}"
    )
    assert parse_tool_call("gemma", body) == ToolCall(
        "search",
        {
            "query": "a, b: {c}",
            "limit": 5,
            "ratio": 0.5,
            "exact": True,
            "filter": None,
            "tags": ["x", 2],
            "where": {"key": "v"},
        },
    )


def test_a_gemma_call_without_arguments():
    assert parse_tool_call("gemma", "call:now{}") == ToolCall("now", {})


@pytest.mark.parametrize(
    "body",
    [
        "read{start:0}",
        "call:read",
        f"call:read{{path:{Q}a}}",
        "call:read{start:zero}",
        "call:read{start:0} trailing",
        "call:read{start:0",
    ],
)
def test_a_malformed_gemma_call_is_an_error(body):
    with pytest.raises(ToolCallError):
        parse_tool_call("gemma", body)
