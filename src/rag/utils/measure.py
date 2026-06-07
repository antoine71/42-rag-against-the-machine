import time
from collections.abc import Callable
from typing import Any, TypeVar

R = TypeVar("R")


def measure(
    func: Callable[..., R], *args: Any, **kwargs: Any
) -> tuple[R, int]:
    start = time.perf_counter() * 1000
    result = func(*args, **kwargs)
    end = time.perf_counter() * 1000
    delta = int(end - start)
    return result, delta
