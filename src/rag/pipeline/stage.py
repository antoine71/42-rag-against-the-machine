from typing import Protocol


class Stage(Protocol):
    def run(self) -> None: ...
