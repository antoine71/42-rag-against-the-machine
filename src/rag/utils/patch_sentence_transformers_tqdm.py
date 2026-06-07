# mypy: disable-error-code=attr-defined
import functools
from collections.abc import Callable
from typing import Any

import sentence_transformers.sentence_transformer.model as st_model


def patch_tqdm(func: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        st_model.trange = functools.partial(st_model.trange, leave=False)
        return func(*args, **kwargs)

    return wrapper
