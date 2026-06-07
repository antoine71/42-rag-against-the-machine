import time
from collections.abc import Callable
from typing import Any, TypeVar

R = TypeVar("R")


def measure(
    func: Callable[..., R], *args: Any, **kwargs: Any
) -> tuple[R, int]:
    """Runs a callable and measures its execution time.

    Args:
        func: Callable to execute.
        *args: Positional arguments forwarded to the callable.
        **kwargs: Keyword arguments forwarded to the callable.

    Returns:
        A tuple containing the callable result and elapsed time in
        milliseconds.
    """
    start = time.perf_counter() * 1000
    result = func(*args, **kwargs)
    end = time.perf_counter() * 1000
    delta = int(end - start)
    return result, delta
