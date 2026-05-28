import logging
from dataclasses import dataclass

from rag.models.minimal_source import MinimalSource
from rag.models.question import AnsweredQuestion
from rag.models.rag_dataset import RagDataset
from rag.models.search_result import StudentSearchResults
from rag.utils.files_manager import FilesManager

logger = logging.getLogger(__name__)


@dataclass
class RecallEvaluation:
    data_is_valid: bool = False
    number_of_questions: int = 0
    number_of_questions_with_sources: int = 0
    number_of_questions_with_student_sources: int = 0
    recall_1: float = 0.0
    recall_3: float = 0.0
    recall_5: float = 0.0
    recall_10: float = 0.0


class EvaluationProcessor:
    def __init__(self, files_manager: FilesManager) -> None:
        self._files_manager = files_manager

    def evaluate(
        self, student_answer_path: str, dataset_path: str
    ) -> RecallEvaluation:
        evaluation = RecallEvaluation()
        student_answers = self._files_manager.load_search_results(
            student_answer_path
        )
        dataset: RagDataset[AnsweredQuestion] = (
            self._files_manager.load_dataset(
                dataset_path, "answered_questions"
            )
        )
        if not self._is_data_valid(student_answers, dataset):
            return evaluation
        evaluation = self._init_evaluation(student_answers, dataset)
        evaluation.recall_1 = self._calculate_metric(
            student_answers, dataset, 1
        )
        evaluation.recall_3 = self._calculate_metric(
            student_answers, dataset, 3
        )
        evaluation.recall_5 = self._calculate_metric(
            student_answers, dataset, 5
        )
        evaluation.recall_10 = self._calculate_metric(
            student_answers, dataset, 10
        )
        return evaluation

    def _calculate_metric(
        self,
        student_answers: StudentSearchResults,
        dataset: RagDataset[AnsweredQuestion],
        k: int,
    ) -> float:
        metrics: list[float] = []
        for question in dataset.rag_questions:
            try:
                student_search = next(
                    s
                    for s in student_answers.search_results
                    if s.question_id == question.question_id
                )
            except StopIteration:
                return 0.0
            truth_sources = question.sources
            student_sources = student_search.retrieved_sources[
                : min(k, len(student_search.retrieved_sources))
            ]
            metrics.append(
                self._evaluate_sources(truth_sources, student_sources)
            )
        return sum(metrics) / len(metrics)

    def _evaluate_sources(
        self,
        truth_sources: list[MinimalSource],
        student_sources: list[MinimalSource],
    ) -> int:
        for student_source in student_sources:
            for truth_source in truth_sources:
                overlap = self._calulate_overlap(truth_source, student_source)
                if overlap >= 0.05:
                    return 1
        return 0

    def _calulate_overlap(
        self, truth_source: MinimalSource, student_source: MinimalSource
    ) -> float:
        if truth_source.file_path != student_source.file_path:
            return 0.0
        if (
            truth_source.first_character_index
            == truth_source.last_character_index
        ):
            return 0.0
        overlap_start = max(
            truth_source.first_character_index,
            student_source.first_character_index,
        )
        overlap_end = min(
            truth_source.last_character_index,
            student_source.last_character_index,
        )
        len_overlap = max(0, overlap_end - overlap_start)
        return len_overlap / (
            truth_source.last_character_index
            - truth_source.first_character_index
        )

    def _init_evaluation(
        self,
        student_answers: StudentSearchResults,
        dataset: RagDataset[AnsweredQuestion],
    ) -> RecallEvaluation:
        evaluation = RecallEvaluation()
        evaluation.data_is_valid = True
        evaluation.number_of_questions = len(dataset.rag_questions)
        evaluation.number_of_questions_with_sources = len(
            [q for q in dataset.rag_questions if len(q.sources) > 0]
        )
        evaluation.number_of_questions_with_student_sources = len(
            [
                s
                for s in student_answers.search_results
                if len(s.retrieved_sources) > 0
            ]
        )
        return evaluation

    def _is_data_valid(
        self,
        student_answers: StudentSearchResults,
        dataset: RagDataset[AnsweredQuestion],
    ) -> bool:
        if set(q.question_id for q in dataset.rag_questions) != set(
            s.question_id for s in student_answers.search_results
        ):
            logger.warning("Invalid student question ids.")
            return False
        for answer in student_answers.search_results:
            for source in answer.retrieved_sources:
                if source.last_character_index <= source.first_character_index:
                    logger.warning("Invalid student source characters index.")
                    return False
        for question in dataset.rag_questions:
            for source in question.sources:
                if source.last_character_index <= source.first_character_index:
                    logger.warning("Invalid dataset characters index.")
                    return False
        return True
