from typing import NamedTuple

HUGGING_FACE = "https://huggingface.co/"


class ModelSource(NamedTuple):
    url: str
    config_overrides: dict[str, str]


class HuggingFaceRepository(NamedTuple):
    repository: str
    revision: str


QWEN2 = {"tool_call_syntax": "none", "think_syntax": "none"}
QWEN2_5 = {"tool_call_syntax": "qwen_json", "think_syntax": "none"}
QWEN3 = {"tool_call_syntax": "qwen_json", "think_syntax": "qwen"}
QWEN3_INSTRUCT = {"tool_call_syntax": "qwen_json", "think_syntax": "none"}
QWEN3_5 = {"tool_call_syntax": "qwen_function", "think_syntax": "qwen"}
LLAMA = {"tool_call_syntax": "none", "think_syntax": "none"}
GEMMA4 = {"tool_call_syntax": "gemma", "think_syntax": "gemma"}


def hugging_face(repository: str, config_overrides: dict[str, str]) -> ModelSource:
    return ModelSource(f"{HUGGING_FACE}{repository}/tree/main", config_overrides)


MODELS = {
    "qwen2-0.5b-instruct": hugging_face("Qwen/Qwen2-0.5B-Instruct", QWEN2),
    "qwen2-1.5b-instruct": hugging_face("Qwen/Qwen2-1.5B-Instruct", QWEN2),
    "qwen2-7b-instruct": hugging_face("Qwen/Qwen2-7B-Instruct", QWEN2),
    "qwen2-72b-instruct": hugging_face("Qwen/Qwen2-72B-Instruct", QWEN2),
    "qwen2.5-0.5b-instruct": hugging_face("Qwen/Qwen2.5-0.5B-Instruct", QWEN2_5),
    "qwen2.5-1.5b-instruct": hugging_face("Qwen/Qwen2.5-1.5B-Instruct", QWEN2_5),
    "qwen2.5-3b-instruct": hugging_face("Qwen/Qwen2.5-3B-Instruct", QWEN2_5),
    "qwen2.5-7b-instruct": hugging_face("Qwen/Qwen2.5-7B-Instruct", QWEN2_5),
    "qwen2.5-14b-instruct": hugging_face("Qwen/Qwen2.5-14B-Instruct", QWEN2_5),
    "qwen2.5-32b-instruct": hugging_face("Qwen/Qwen2.5-32B-Instruct", QWEN2_5),
    "qwen2.5-72b-instruct": hugging_face("Qwen/Qwen2.5-72B-Instruct", QWEN2_5),
    "qwen3-0.6b": hugging_face("Qwen/Qwen3-0.6B", QWEN3),
    "qwen3-1.7b": hugging_face("Qwen/Qwen3-1.7B", QWEN3),
    "qwen3-4b": hugging_face("Qwen/Qwen3-4B", QWEN3),
    "qwen3-4b-instruct-2507": hugging_face(
        "Qwen/Qwen3-4B-Instruct-2507", QWEN3_INSTRUCT
    ),
    "qwen3-4b-thinking-2507": hugging_face("Qwen/Qwen3-4B-Thinking-2507", QWEN3),
    "qwen3-8b": hugging_face("Qwen/Qwen3-8B", QWEN3),
    "qwen3-14b": hugging_face("Qwen/Qwen3-14B", QWEN3),
    "qwen3-32b": hugging_face("Qwen/Qwen3-32B", QWEN3),
    "qwen3.5-0.8b": hugging_face("Qwen/Qwen3.5-0.8B", QWEN3_5),
    "qwen3.5-2b": hugging_face("Qwen/Qwen3.5-2B", QWEN3_5),
    "qwen3.5-4b": hugging_face("Qwen/Qwen3.5-4B", QWEN3_5),
    "qwen3.5-9b": hugging_face("Qwen/Qwen3.5-9B", QWEN3_5),
    "qwen3.5-35b-a3b": hugging_face("Qwen/Qwen3.5-35B-A3B", QWEN3_5),
    "qwen3.5-122b-a10b": hugging_face("Qwen/Qwen3.5-122B-A10B", QWEN3_5),
    "qwen3.5-397b-a17b": hugging_face("Qwen/Qwen3.5-397B-A17B", QWEN3_5),
    "qwen3.6-35b-a3b": hugging_face("Qwen/Qwen3.6-35B-A3B", QWEN3_5),
    "qwen3.8-27b": hugging_face("Qwen/Qwen3.8-27B", QWEN3_5),
    "qwen3.8-27b-fp8": hugging_face("Qwen/Qwen3.8-27B-FP8", QWEN3_5),
    "qwen3.8-2.4t-a95b": hugging_face("Qwen/Qwen3.8-2.4T-A95B", QWEN3_5),
    "qwen3.8-flash-next": hugging_face("Qwen/Qwen3.8-Flash-Next", QWEN3_5),
    "llama-3-8b-instruct": hugging_face("meta-llama/Meta-Llama-3-8B-Instruct", LLAMA),
    "llama-3-70b-instruct": hugging_face("meta-llama/Meta-Llama-3-70B-Instruct", LLAMA),
    "llama-3.1-8b-instruct": hugging_face("meta-llama/Llama-3.1-8B-Instruct", LLAMA),
    "llama-3.1-70b-instruct": hugging_face("meta-llama/Llama-3.1-70B-Instruct", LLAMA),
    "llama-3.2-1b-instruct": hugging_face("meta-llama/Llama-3.2-1B-Instruct", LLAMA),
    "llama-3.2-3b-instruct": hugging_face("meta-llama/Llama-3.2-3B-Instruct", LLAMA),
    "llama-3.3-70b-instruct": hugging_face("meta-llama/Llama-3.3-70B-Instruct", LLAMA),
    "gemma-4-e2b-it": hugging_face("google/gemma-4-E2B-it", GEMMA4),
    "gemma-4-e4b-it": hugging_face("google/gemma-4-E4B-it", GEMMA4),
    "gemma-4-12b-it": hugging_face("google/gemma-4-12B-it", GEMMA4),
    "gemma-4-26b-a4b-it": hugging_face("google/gemma-4-26B-A4B-it", GEMMA4),
    "gemma-4-31b-it": hugging_face("google/gemma-4-31B-it", GEMMA4),
}


def hugging_face_repository(url: str) -> HuggingFaceRepository:
    """url is https://huggingface.co/<repository>/tree/<revision>."""
    repository, tree, revision = url.removeprefix(HUGGING_FACE).partition("/tree/")
    if not url.startswith(HUGGING_FACE) or not tree or not repository or not revision:
        raise ValueError(f"{url} is not a Hugging Face repository at a revision")
    return HuggingFaceRepository(repository, revision)
