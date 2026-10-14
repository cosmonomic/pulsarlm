REPOSITORIES = {
    "qwen2-0.5b-instruct": "Qwen/Qwen2-0.5B-Instruct",
    "qwen2-1.5b-instruct": "Qwen/Qwen2-1.5B-Instruct",
    "qwen2-7b-instruct": "Qwen/Qwen2-7B-Instruct",
    "qwen2-72b-instruct": "Qwen/Qwen2-72B-Instruct",
    "qwen2.5-0.5b-instruct": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5-1.5b-instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-3b-instruct": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-14b-instruct": "Qwen/Qwen2.5-14B-Instruct",
    "qwen2.5-32b-instruct": "Qwen/Qwen2.5-32B-Instruct",
    "qwen2.5-72b-instruct": "Qwen/Qwen2.5-72B-Instruct",
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-4b-instruct-2507": "Qwen/Qwen3-4B-Instruct-2507",
    "qwen3-4b-thinking-2507": "Qwen/Qwen3-4B-Thinking-2507",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3-14b": "Qwen/Qwen3-14B",
    "qwen3-32b": "Qwen/Qwen3-32B",
    "gemma-4-e2b-it": "google/gemma-4-E2B-it",
    "gemma-4-e4b-it": "google/gemma-4-E4B-it",
    "gemma-4-12b-it": "google/gemma-4-12B-it",
    "gemma-4-26b-a4b-it": "google/gemma-4-26B-A4B-it",
    "gemma-4-31b-it": "google/gemma-4-31B-it",
}


def repository_url(model: str) -> str:
    return f"https://huggingface.co/{REPOSITORIES[model]}"
