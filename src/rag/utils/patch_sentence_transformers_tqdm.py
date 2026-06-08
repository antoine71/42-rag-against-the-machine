# mypy: disable-error-code=attr-defined
import functools
from collections.abc import Callable
from typing import Any

import sentence_transformers.sentence_transformer.model as st_model

from rag.tui import TUI


def patch_tqdm(func: Callable[..., Any]) -> Callable[..., Any]:
    """Patches sentence-transformers progress bars for wrapped calls.

    Args:
        func: Callable that may create sentence-transformers tqdm ranges.

    Returns:
        A wrapper that disables persistent tqdm bars before calling `func`.
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Runs the wrapped callable after patching tqdm behavior.

        Args:
            *args: Positional arguments forwarded to the wrapped callable.
            **kwargs: Keyword arguments forwarded to the wrapped callable.

        Returns:
            The wrapped callable result.
        """
        st_model.trange = functools.partial(
            st_model.trange, leave=False, ncols=TUI.WIDTH
        )
        return func(*args, **kwargs)

    return wrapper
