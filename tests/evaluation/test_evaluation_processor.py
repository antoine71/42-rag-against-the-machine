from unittest.mock import MagicMock

import pytest

from rag.evaluation.evaluation_processor import EvaluationProcessor
from rag.models.minimal_source import MinimalSource
from rag.models.question import AnsweredQuestion
from rag.models.rag_dataset import RagDataset
from rag.models.search_result import MinimalSearchResults, StudentSearchResults


@pytest.fixture
def valid_student_answers() -> StudentSearchResults:
    return StudentSearchResults(
        search_results=[
            MinimalSearchResults(
                question_id="id1",
                question="q1",
                retrieved_sources=[
                    MinimalSource(
                        file_path="f1",
                        first_character_index=0,
                        last_character_index=100,
                    )
                ],
            )
        ],
        k=1,
    )


@pytest.fixture
def valid_dataset() -> RagDataset[AnsweredQuestion]:
    return RagDataset(
        rag_questions=[
            AnsweredQuestion(
                question_id="id1",
                question="q1",
                answer="a1",
                sources=[
                    MinimalSource(
                        file_path="f1",
                        first_character_index=0,
                        last_character_index=100,
                    )
                ],
            )
        ]
    )


@pytest.fixture
def evaluation_processor(
    valid_student_answers: StudentSearchResults,
    valid_dataset: RagDataset[AnsweredQuestion],
) -> EvaluationProcessor:
    files_manager = MagicMock()
    files_manager.load_search_results.return_value = valid_student_answers
    files_manager.load_dataset.return_value = valid_dataset
    return EvaluationProcessor(files_manager)


class TestEvaluationProcessor:
    def test_evaluate(self, evaluation_processor: EvaluationProcessor) -> None:
        result = evaluation_processor.evaluate("mock", "mock")
        assert result.recall_1 == 1.0
        assert result.recall_5 == 1.0
        assert result.recall_10 == 1.0

    def test_calculate_metric(
        self,
        evaluation_processor: EvaluationProcessor,
        valid_student_answers: StudentSearchResults,
        valid_dataset: RagDataset[AnsweredQuestion],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        assert (
            evaluation_processor._calculate_metric(
                valid_student_answers, valid_dataset, 1
            )
            == 1.0
        )
        invalid = StudentSearchResults(
            search_results=[
                MinimalSearchResults(
                    question_id="id2",
                    question="q1",
                    retrieved_sources=[
                        MinimalSource(
                            file_path="f1",
                            first_character_index=0,
                            last_character_index=100,
                        )
                    ],
                )
            ],
            k=1,
        )
        assert (
            evaluation_processor._calculate_metric(invalid, valid_dataset, 1)
            == 0.0
        )
        invalid = StudentSearchResults(
            search_results=[
                MinimalSearchResults(
                    question_id="id1",
                    question="q1",
                    retrieved_sources=[
                        MinimalSource(
                            file_path="f2",
                            first_character_index=0,
                            last_character_index=100,
                        )
                    ],
                )
            ],
            k=1,
        )
        assert (
            evaluation_processor._calculate_metric(invalid, valid_dataset, 1)
            == 0.0
        )

    def test_is_data_valid(
        self,
        evaluation_processor: EvaluationProcessor,
        valid_student_answers: StudentSearchResults,
        valid_dataset: RagDataset[AnsweredQuestion],
    ) -> None:
        assert (
            evaluation_processor._is_data_valid(
                valid_student_answers, valid_dataset
            )
            is True
        )

    def test_is_data_valid_ids_mismatch(
        self,
        evaluation_processor: EvaluationProcessor,
        valid_student_answers: StudentSearchResults,
        valid_dataset: RagDataset[AnsweredQuestion],
    ) -> None:
        valid_student_answers.search_results[0].question_id = "id2"
        assert (
            evaluation_processor._is_data_valid(
                valid_student_answers, valid_dataset
            )
            is False
        )

    def test_is_data_valid_ids_invalid_sources(
        self,
        evaluation_processor: EvaluationProcessor,
        valid_dataset: RagDataset[AnsweredQuestion],
    ) -> None:
        invalid = StudentSearchResults(
            search_results=[
                MinimalSearchResults(
                    question_id="id1",
                    question="q1",
                    retrieved_sources=[
                        MinimalSource(
                            file_path="f1",
                            first_character_index=1000,
                            last_character_index=100,
                        )
                    ],
                )
            ],
            k=1,
        )
        assert (
            evaluation_processor._is_data_valid(invalid, valid_dataset)
            is False
        )

    def test_init_evaluation(
        self,
        evaluation_processor: EvaluationProcessor,
        valid_student_answers: StudentSearchResults,
        valid_dataset: RagDataset[AnsweredQuestion],
    ) -> None:
        evaluation = evaluation_processor._init_evaluation(
            valid_student_answers, valid_dataset
        )
        assert evaluation.data_is_valid is True
        assert evaluation.number_of_questions == 1
        assert evaluation.number_of_questions_with_sources == 1
        assert evaluation.number_of_questions_with_student_sources == 1

    @pytest.mark.parametrize(
        "truth_source,student_source,overlap",
        [
            (("f1", 0, 100), ("f1", 0, 100), 1.0),
            (("f1", 50, 50), ("f1", 0, 100), 0.0),
            (("f2", 0, 100), ("f1", 0, 100), 0.0),
            (("f1", 10, 90), ("f1", 0, 100), 1.0),
            (("f1", 0, 100), ("f1", 10, 90), 0.8),
            (("f1", 50, 90), ("f1", 0, 60), 0.25),
            (("f1", 50, 80), ("f1", 0, 40), 0.0),
            (("f1", 0, 50), ("f1", 60, 80), 0.0),
        ],
    )
    def test_calculate_overlap(
        self,
        truth_source: tuple[str, int, int],
        student_source: tuple[str, int, int],
        overlap: float,
        evaluation_processor: EvaluationProcessor,
    ):
        truth_source_obj = MinimalSource(
            file_path=truth_source[0],
            first_character_index=truth_source[1],
            last_character_index=truth_source[2],
        )
        student_source_obj = MinimalSource(
            file_path=student_source[0],
            first_character_index=student_source[1],
            last_character_index=student_source[2],
        )

        assert (
            evaluation_processor._calulate_overlap(
                truth_source_obj, student_source_obj
            )
            == overlap
        )
