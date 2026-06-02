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
    """Represents the results of the recall evaluation.

    Attributes:
        data_is_valid: A boolean indicating if the input data format is valid.
        number_of_questions: The total number of questions evaluated.
        number_of_questions_with_sources: Total questions that have ground
            truth sources.
        number_of_questions_with_student_sources: Total questions that have
            student sources.
        recall_1: The recall at k=1 score.
        recall_3: The recall at k=3 score.
        recall_5: The recall at k=5 score.
        recall_10: The recall at k=10 score.
    """

    data_is_valid: bool = False
    number_of_questions: int = 0
    number_of_questions_with_sources: int = 0
    number_of_questions_with_student_sources: int = 0
    recall_1: float = 0.0
    recall_3: float = 0.0
    recall_5: float = 0.0
    recall_10: float = 0.0


class EvaluationProcessor:
    """Processor for evaluating RAG retrieval performance using
    recall@k metrics.
    """

    def __init__(self, files_manager: FilesManager) -> None:
        """Initializes the EvaluationProcessor.

        Args:
            files_manager: The files manager utility to load dataset and
                answers.
        """
        self._files_manager = files_manager

    def evaluate(
        self, student_answer_path: str, dataset_path: str
    ) -> RecallEvaluation:
        """Evaluates student search results against the ground truth dataset.

        Args:
            student_answer_path: Path to the student search results JSON file.
            dataset_path: Path to the ground truth answered questions dataset.

        Returns:
            A RecallEvaluation object containing the computed recall metrics.
        """
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
        """Calculates recall@k score for the entire dataset.

        Args:
            student_answers: The retrieved search results from the student.
            dataset: The ground truth dataset.
            k: The cutoff parameter for top-k results.

        Returns:
            The average recall@k score across all questions.
        """
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
            metric = self._evaluate_sources(truth_sources, student_sources)
            metrics.append(metric)
            if k == 10 and metric == 0:
                logger.debug(
                    "Invalid sources for question %s", question.question_id
                )
        return sum(metrics) / len(metrics) if metrics else 0.0

    def _evaluate_sources(
        self,
        truth_sources: list[MinimalSource],
        student_sources: list[MinimalSource],
    ) -> float:
        """Evaluates retrieval recall score for a single question.

        A correct source is considered found if it has at least 5%
        character overlap with any retrieved source. The score is:
        number of ground truth sources found / total number of correct sources.

        Args:
            truth_sources: List of correct ground truth sources.
            student_sources: List of retrieved student sources.

        Returns:
            The recall score as a float between 0.0 and 1.0.
        """
        if not truth_sources:
            return 0.0

        found_indices = set()
        for student_source in student_sources:
            for idx, truth_source in enumerate(truth_sources):
                overlap = self._calulate_overlap(truth_source, student_source)
                if overlap >= 0.05:
                    found_indices.add(idx)

        return len(found_indices) / len(truth_sources)

    def _calulate_overlap(
        self, truth_source: MinimalSource, student_source: MinimalSource
    ) -> float:
        """Calculates character overlap ratio of truth with student source.

        Args:
            truth_source: The ground truth source.
            student_source: The retrieved student source.

        Returns:
            The ratio of character overlap relative to truth source's length.
        """
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
        """Initializes structural statistics for the RecallEvaluation report.

        Args:
            student_answers: The retrieved search results from the student.
            dataset: The ground truth dataset.

        Returns:
            An initial RecallEvaluation object populated with basic metadata.
        """
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
        """Validates that the student search results match the dataset
        structure.

        Args:
            student_answers: The retrieved student search results.
            dataset: The ground truth dataset.

        Returns:
            True if all checks pass, False otherwise.
        """
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
