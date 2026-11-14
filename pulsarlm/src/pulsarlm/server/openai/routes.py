import json
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRouter
from tokenizers import Tokenizer

from pulsarlm.async_engine import AsyncEngine, EngineStopped
from pulsarlm.chat import ChatTemplate, ChatTemplateError, Message, encode_prompt
from pulsarlm.completion import CompletionError, CompletionUsage, complete
from pulsarlm.core.engine import EvictionPolicy, RequestError, SamplingParams
from pulsarlm.core.engine import State as EngineState
from pulsarlm.eviction_log import EvictionLog
from pulsarlm.server.openai.schema import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    DeltaToolCall,
    FinishReason,
    FunctionCall,
    MessageToolCall,
    ModelCard,
    ModelList,
    PromptTokensDetails,
    Usage,
)
from pulsarlm.paging import sink_tokens
from pulsarlm.reply import TOOL_CALL_MARKERS, Part, ReplyDecoder, ThinkSyntax
from pulsarlm.tools import ToolCallError, ToolCallSyntax, parse_tool_call


@dataclass
class ServerState:
    engine: AsyncEngine
    tokenizer: Tokenizer
    chat_template: ChatTemplate
    model_id: str
    stop_ids: list[int]
    tool_call_syntax: ToolCallSyntax | None
    think_syntax: ThinkSyntax | None
    eviction: EvictionPolicy
    eviction_log: EvictionLog | None

    def eviction_for(self, system_tokens: int) -> EvictionPolicy:
        """The default policy, sinking the system message's tokens, or the first
        token without one."""
        return EvictionPolicy(
            n_sink=sink_tokens(system_tokens),
            n_working=self.eviction.n_working,
            cycle_tokens=self.eviction.cycle_tokens,
            max_context_window=self.eviction.max_context_window,
            eviction_threshold=self.eviction.eviction_threshold,
            eviction_warmup=self.eviction.eviction_warmup,
            attention_mass_decay=self.eviction.attention_mass_decay,
        )


def get_server_state(request: Request) -> ServerState:
    return request.app.state.server_state


type State = Annotated[ServerState, Depends(get_server_state)]


router = APIRouter()


@router.get("/models")
async def list_models(state: State) -> ModelList:
    return ModelList(data=[ModelCard(id=state.model_id, created=int(time.time()))])


@router.post("/chat/completions")
async def create_chat_completion(body: ChatCompletionRequest, state: State):
    if body.model != state.model_id:
        raise HTTPException(
            404, f"model {body.model!r} not found, only {state.model_id} is loaded"
        )
    if not body.messages:
        raise HTTPException(422, "messages must be non-empty")
    tools = (
        [tool.model_dump() for tool in body.tools]
        if body.tools and body.tool_choice == "auto"
        else None
    )
    if tools is not None and state.tool_call_syntax is None:
        raise HTTPException(
            400, f"{state.model_id} has no tool call syntax pulsarlm parses"
        )

    try:
        prompt, system_tokens = encode_prompt(
            state.tokenizer,
            state.chat_template,
            template_messages(body.messages),
            reasoning_effort=body.reasoning_effort,
            tools=tools,
        )
    except ChatTemplateError as error:
        raise HTTPException(400, f"chat template: {error}") from error
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    states: list[EngineState] = []
    reply = _Reply(state, body.max_tokens, prompt, tool_calls=tools is not None)
    tokens = complete(
        state.engine,
        prompt,
        sampling=SamplingParams(
            temperature=body.temperature,
            top_p=body.top_p,
            top_k=body.top_k,
            seed=body.seed,
        ),
        max_tokens=body.max_tokens,
        eviction=state.eviction_for(system_tokens),
        on_state=None if state.eviction_log is None else states.append,
        on_completed=reply.record_usage,
    )
    if state.eviction_log is not None:
        tokens = state.eviction_log.record(completion_id, prompt, tokens, states)
    if body.stream:
        return _stream_completion(body, completion_id, reply, tokens)
    return await _full_completion(body, completion_id, reply, tokens)


def template_messages(messages: list[ChatMessage]) -> list[Message]:
    """Decodes tool call arguments, which chat templates take as mappings."""
    rendered = []
    for m in messages:
        message: Message = {"role": m.role, "content": m.content or ""}
        if m.reasoning is not None:
            message["reasoning"] = message["reasoning_content"] = m.reasoning
        if m.tool_calls:
            try:
                message["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": call.type,
                        "function": {
                            "name": call.function.name,
                            "arguments": json.loads(call.function.arguments),
                        },
                    }
                    for call in m.tool_calls
                ]
            except json.JSONDecodeError as error:
                raise HTTPException(
                    422, f"tool call arguments are not JSON: {error}"
                ) from error
        if m.tool_call_id is not None:
            message["tool_call_id"] = m.tool_call_id
        rendered.append(message)
    return rendered


class _Reply:
    """Decodes sampled tokens until a stop token."""

    def __init__(
        self,
        state: ServerState,
        max_tokens: int | None,
        prompt: list[int],
        *,
        tool_calls: bool,
    ) -> None:
        self._stop_ids = state.stop_ids
        self._max_tokens = max_tokens
        self._decoder = ReplyDecoder(
            state.tokenizer,
            state.stop_ids,
            think_syntax=state.think_syntax,
            tool_call_syntax=state.tool_call_syntax if tool_calls else None,
            prompt=prompt,
        )
        self.token_count = 0
        self.finish_reason: FinishReason | None = None
        self.usage: CompletionUsage | None = None

    def record_usage(self, usage: CompletionUsage) -> None:
        self.usage = usage

    def usage_body(self) -> Usage:
        assert self.usage is not None
        return Usage(
            prompt_tokens=self.usage.prompt_tokens,
            completion_tokens=self.usage.completion_tokens,
            total_tokens=self.usage.prompt_tokens + self.usage.completion_tokens,
            prompt_tokens_details=PromptTokensDetails(
                cached_tokens=self.usage.cached_tokens
            ),
        )

    def step(self, token: int) -> tuple[Part, str]:
        self.token_count += 1
        if token in self._stop_ids:
            self.finish_reason = "stop"
        elif self.token_count == self._max_tokens:
            self.finish_reason = "length"
        return self._decoder.step(token)

    def tool_calls(self) -> tuple[list[MessageToolCall], str]:
        """The tool calls, and the text of those that do not parse."""
        syntax = self._decoder.tool_call_syntax
        if syntax is None:
            return [], ""
        marker = TOOL_CALL_MARKERS[syntax]
        calls: list[MessageToolCall] = []
        unparsed: list[str] = []
        for body in self._decoder.tool_call_bodies:
            try:
                call = parse_tool_call(syntax, body)
            except ToolCallError:
                unparsed.append(marker.opening + body + marker.closing)
                continue
            calls.append(
                MessageToolCall(
                    id=f"call_{uuid.uuid4().hex[:24]}",
                    function=FunctionCall(
                        name=call.name, arguments=json.dumps(call.arguments)
                    ),
                )
            )
        return calls, "".join(unparsed)

    def final_finish_reason(self, calls: list[MessageToolCall]) -> FinishReason:
        if self.finish_reason in (None, "stop") and calls:
            return "tool_calls"
        return self.finish_reason or "stop"


async def _texts(
    reply: _Reply, tokens: AsyncGenerator[int]
) -> AsyncIterator[tuple[Part, str]]:
    async with aclosing(tokens):
        async for token in tokens:
            yield reply.step(token)


def _http_error(error: EngineStopped | CompletionError) -> HTTPException:
    if isinstance(error, EngineStopped):
        return HTTPException(503, f"engine stopped: {error}")
    if isinstance(error, CompletionError) and error.error == RequestError.OOM:
        return HTTPException(400, "out of device memory")
    return HTTPException(500, f"request failed: {error}")


async def _full_completion(
    body: ChatCompletionRequest,
    completion_id: str,
    reply: _Reply,
    tokens: AsyncGenerator[int],
) -> ChatCompletionResponse:
    texts: dict[Part, list[str]] = {"reasoning": [], "content": []}
    try:
        async for part, text in _texts(reply, tokens):
            texts[part].append(text)
    except (EngineStopped, CompletionError) as error:
        raise _http_error(error) from error
    calls, unparsed = reply.tool_calls()
    content = "".join(texts["content"]) + unparsed

    return ChatCompletionResponse(
        id=completion_id,
        created=int(time.time()),
        model=body.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(
                    role="assistant",
                    content=content or (None if calls else ""),
                    reasoning="".join(texts["reasoning"]) or None,
                    tool_calls=calls or None,
                ),
                finish_reason=reply.final_finish_reason(calls),
            )
        ],
        usage=reply.usage_body(),
    )


def _stream_completion(
    body: ChatCompletionRequest,
    completion_id: str,
    reply: _Reply,
    tokens: AsyncGenerator[int],
) -> StreamingResponse:
    created = int(time.time())

    def chunk(
        delta: ChatCompletionChunkDelta, finish_reason: FinishReason | None = None
    ) -> str:
        payload = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=body.model,
            choices=[
                ChatCompletionChunkChoice(delta=delta, finish_reason=finish_reason)
            ],
        )
        return f"data: {payload.model_dump_json()}\n\n"

    async def events() -> AsyncIterator[str]:
        yield chunk(ChatCompletionChunkDelta(role="assistant"))
        try:
            async for part, text in _texts(reply, tokens):
                if text and part == "reasoning":
                    yield chunk(ChatCompletionChunkDelta(reasoning=text))
                elif text:
                    yield chunk(ChatCompletionChunkDelta(content=text))
        except (EngineStopped, CompletionError) as error:
            failure = _http_error(error)
            yield f"data: {json.dumps({'error': {'message': failure.detail, 'code': failure.status_code}})}\n\n"
        else:
            calls, unparsed = reply.tool_calls()
            if unparsed:
                yield chunk(ChatCompletionChunkDelta(content=unparsed))
            if calls:
                yield chunk(
                    ChatCompletionChunkDelta(
                        tool_calls=[
                            DeltaToolCall(index=i, **call.model_dump())
                            for i, call in enumerate(calls)
                        ]
                    )
                )
            yield chunk(ChatCompletionChunkDelta(), reply.final_finish_reason(calls))
            if body.stream_options is not None and body.stream_options.include_usage:
                usage = ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=body.model,
                    choices=[],
                    usage=reply.usage_body(),
                )
                yield f"data: {usage.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
