from types import TracebackType

from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner


class Report:
    def __init__(self) -> None:
        self._live = Live()
        self._tasks:
        self._groups: 

    def __enter__(self) -> "Report":
        self._live.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return self._live.__exit__(exc_type, exc, tb)

    def add_dataset(self, dataset: str, title: str) -> None:
        panel = Panel(f"Dataset\t{dataset}", title=title)
        self._live.update(panel)

    def add_task(self, label: str):
        spinner = Spinner("dots", text=label)
        self._live.update(spinner)
