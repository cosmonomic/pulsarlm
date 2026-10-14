from typing import Any, Literal

from pydantic import BaseModel, Field

from pulsarlm.chat import ReasoningEffort

type FinishReason = Literal["stop", "length", "tool_calls"]


class FunctionCall(BaseModel):
    name: str
    arguments: str


class MessageToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    reasoning: str | None = None
    tool_calls: list[MessageToolCall] | None = None
    tool_call_id: str | None = None


class FunctionDefinition(BaseModel):
    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )


class Tool(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDefinition


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    max_tokens: int | None = Field(default=None, ge=0)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    seed: int | None = Field(default=None, ge=0)
    reasoning_effort: ReasoningEffort | None = None
    tools: list[Tool] | None = None
    tool_choice: Literal["auto", "none"] = "auto"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: FinishReason


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage


class DeltaToolCall(MessageToolCall):
    index: int


class ChatCompletionChunkDelta(BaseModel):
    role: Literal["assistant"] | None = None
    content: str | None = None
    reasoning: str | None = None
    tool_calls: list[DeltaToolCall] | None = None


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: ChatCompletionChunkDelta
    finish_reason: FinishReason | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "pulsarlm"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]
