
PAGE_SIZE = 16  # the tensor-core paged attention kernel requires 16 or 32


def sink_tokens(system_tokens: int) -> int:
    """The system message's tokens, or the first token without one."""
    return max(system_tokens, 1)
