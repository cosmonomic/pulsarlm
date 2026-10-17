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
from pulsarlm.completion import CompletionError, complete
from pulsarlm.core.engine import EvictionPolicy, RequestError, SamplingParams
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
    Usage,
)
from pulsarlm.paging import sink_pages
from pulsarlm.reply import Part, ReplyDecoder, tool_call_marker
from pulsarlm.store import LocalStore
from pulsarlm.tools import ToolCallError, parse_tool_call


@dataclass
class ServerState:
    engine: AsyncEngine
    store: LocalStore
    tokenizer: Tokenizer
    chat_template: ChatTemplate
    model_id: str
    stop_ids: list[int]
    eviction: EvictionPolicy

    def eviction_for(self, system_tokens: int) -> EvictionPolicy:
        """The default policy, sinking the pages the system message touches, or
        the first page without one."""
        return EvictionPolicy(
            n_sink=sink_pages(system_tokens),
            n_working=self.eviction.n_working,
            cycle_tokens=self.eviction.cycle_tokens,
            max_pages=self.eviction.max_pages,
            eviction_target=self.eviction.eviction_target,
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
    if tools is not None and tool_call_marker(state.tokenizer) is None:
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
    tokens = complete(
        state.engine,
        state.store,
        prompt,
        sampling=SamplingParams(
            temperature=body.temperature,
            top_p=body.top_p,
            top_k=body.top_k,
            seed=body.seed,
        ),
        max_tokens=body.max_tokens,
        eviction=state.eviction_for(system_tokens),
    )
    reply = _Reply(state, body.max_tokens, prompt, tool_calls=tools is not None)
    if body.stream:
        return _stream_completion(body, reply, tokens)
    return await _full_completion(body, reply, tokens, len(prompt))


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
            state.tokenizer, state.stop_ids, tool_calls=tool_calls, prompt=prompt
        )
        self.token_count = 0
        self.finish_reason: FinishReason | None = None

    def step(self, token: int) -> tuple[Part, str]:
        self.token_count += 1
        if token in self._stop_ids:
            self.finish_reason = "stop"
        elif self.token_count == self._max_tokens:
            self.finish_reason = "length"
        return self._decoder.step(token)

    def tool_calls(self) -> tuple[list[MessageToolCall], str]:
        """The tool calls, and the text of those that do not parse."""
        marker = self._decoder.tool_call_marker
        if marker is None:
            return [], ""
        calls: list[MessageToolCall] = []
        unparsed: list[str] = []
        for body in self._decoder.tool_call_bodies:
            try:
                call = parse_tool_call(marker.syntax, body)
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
        return HTTPException(503, "the request does not fit in the KV cache")
    return HTTPException(500, f"request failed: {error}")


async def _full_completion(
    body: ChatCompletionRequest,
    reply: _Reply,
    tokens: AsyncGenerator[int],
    prompt_tokens: int,
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
        id=f"chatcmpl-{uuid.uuid4().hex}",
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
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=reply.token_count,
            total_tokens=prompt_tokens + reply.token_count,
        ),
    )


def _stream_completion(
    body: ChatCompletionRequest,
    reply: _Reply,
    tokens: AsyncGenerator[int],
) -> StreamingResponse:
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
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
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
