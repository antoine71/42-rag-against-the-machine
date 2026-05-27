import itertools
import random
from collections.abc import Callable

import pytest

from rag.models.minimal_source import MinimalSource
from rag.retrieving.retrieving_manager import RetrievingManager


def minimal_source_factory(chunk_id: str) -> MinimalSource:
    return MinimalSource(
        chunk_id=chunk_id,
        file_path="mock",
        first_character_index=random.randint(0, 100),
        last_character_index=random.randint(101, 200),
    )


@pytest.fixture
def retrieving_manager() -> RetrievingManager:
    return RetrievingManager()


@pytest.fixture
def sources_ranks() -> list[list[MinimalSource]]:
    return [
        [
            MinimalSource(
                file_path="B",
                first_character_index=0,
                last_character_index=100,
            ),
            MinimalSource(
                file_path="A",
                first_character_index=0,
                last_character_index=100,
            ),
        ],
        [
            MinimalSource(
                file_path="C",
                first_character_index=0,
                last_character_index=100,
            ),
            MinimalSource(
                file_path="A",
                first_character_index=0,
                last_character_index=100,
            ),
        ],
    ]


@pytest.fixture
def merge_fct() -> Callable[[list[list[MinimalSource]]], list[MinimalSource]]:
    def mock_merge_fct(
        inputs: list[list[MinimalSource]],
    ) -> list[MinimalSource]:
        return sorted(
            set(itertools.chain.from_iterable(inputs)),
            key=lambda x: x.file_path,
        )

    return mock_merge_fct


class TestRetrievingManager:
    def test_process_query(
        self,
        retrieving_manager: RetrievingManager,
        sources_ranks: list[list[MinimalSource]],
        merge_fct: Callable[[list[list[MinimalSource]]], list[MinimalSource]],
    ) -> None:
        pass
