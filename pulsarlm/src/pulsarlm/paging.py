import math

PAGE_SIZE = 16  # the tensor-core paged attention kernel requires 16 or 32


def trained_view_pages(max_positions: int, cycle_tokens: int) -> int | None:
    """The most full pages a view keeps at a cycle's end so that, with a partial tail
    page and the next cycle's tokens, its positions stay within the model's trained
    context. None when the config states no limit."""
    if max_positions <= 0:
        return None
    return max(1, (max_positions - cycle_tokens - (PAGE_SIZE - 1)) // PAGE_SIZE)


def sink_pages(system_tokens: int) -> int:
    """The pages the system message touches, or the first page without one."""
    return math.ceil(system_tokens / PAGE_SIZE) if system_tokens else 1
