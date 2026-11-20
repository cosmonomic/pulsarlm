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


class StreamOptions(BaseModel):
    include_usage: bool = False


class EvictionOptions(BaseModel):
    n_working: int | None = Field(
        default=None,
        ge=0,
        description="The last tokens a cycle's end never evicts; at least 1.",
    )
    cycle_tokens: int | None = Field(
        default=None,
        ge=0,
        description="The tokens between two cycle ends; at least 1.",
    )
    mass_tokens: int | None = Field(
        default=None,
        ge=0,
        description="The last tokens of each cycle whose queries accumulate page "
        "masses, in [1, cycle_tokens]; cycle_tokens takes all of the cycle's.",
    )
    max_context_capacity: int | None = Field(
        default=None,
        description="The view's cap, in tokens: 0 evicts nothing by capacity, and "
        "a negative value takes the server's max_context_window.",
    )
    eviction_threshold: float | None = Field(
        default=None,
        description="The quantile of each layer's page scores below which a "
        "cycle's end evicts a page, in [0, 1]; 0 evicts nothing by score.",
    )
    eviction_warmup: int | None = Field(
        default=None,
        ge=0,
        description="The cycles whose pages a layer samples twice over before it "
        "evicts by score.",
    )
    attention_mass_decay: float | None = Field(
        default=None,
        description="The page density EMA's rate per token, in (0, 1].",
    )


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    stream_options: StreamOptions | None = None
    max_tokens: int | None = Field(default=None, ge=0)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    seed: int | None = Field(default=None, ge=0)
    reasoning_effort: ReasoningEffort | None = None
    tools: list[Tool] | None = None
    tool_choice: Literal["auto", "none"] = "auto"
    eviction: EvictionOptions | None = Field(
        default=None,
        description="The request's eviction policy; a field it leaves out or sets "
        "to null takes the server's default.",
    )


class PromptTokensDetails(BaseModel):
    cached_tokens: int


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: PromptTokensDetails


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
    usage: Usage | None = None


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "pulsarlm"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]
