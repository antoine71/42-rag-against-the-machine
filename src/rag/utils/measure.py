import time
from collections.abc import Callable
from typing import TypeVar

R = TypeVar("R")


def measure(func: Callable[..., R], *args, **kwargs) -> tuple[R, int]:
    start = time.perf_counter() * 1000
    result = func(*args, **kwargs)
    end = time.perf_counter() * 1000
    delta = int(end - start)
    return result, delta
