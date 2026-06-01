import functools
import inspect
import logging
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ValidationError

from rag.exceptions import RAGException

logger = logging.getLogger(__name__)


def args_to_kwargs(
    func: Callable[[Any], Any], *args: tuple[Any]
) -> dict[str, Any]:
    """Map CLI positional arguments to keyword arguments using signature.

    Args:
        func: The target function to analyze.
        *args: The positional arguments passed to the function.

    Returns:
        A dictionary mapping argument names to their values.
    """
    sig = inspect.signature(func)
    params = list(sig.parameters.values())

    kwargs = {}

    for param, value in zip(params, args):
        kwargs[param.name] = value

    return kwargs


def validate_with(model: type[BaseModel]) -> Callable[[Any], Any]:
    """Decorator to validate a function's arguments against a Pydantic model.

    Converts positional arguments to keyword arguments before validating.

    Args:
        model: The Pydantic BaseModel subclass to validate against.

    Returns:
        A decorator function wrapping the original function.
    """

    def decorator(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
        """Wraps the target function with argument validation."""
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Validate CLI positional arguments against the Pydantic model.
            """
            try:
                params = model(**args_to_kwargs(func, *args))
                logger.info(
                    f"Processing {func.__name__} with parameters "
                    f"{params.model_dump()}."
                )
            except ValidationError as e:
                raise RAGException(f"Command line argument error:\n{e}") from e
            return func(*args, **kwargs)

        return wrapper

    return decorator
