from collections.abc import Generator
from contextlib import contextmanager
from textwrap import dedent
from typing import Any

from tqdm import tqdm

from rag.evaluation.evaluation_processor import RecallEvaluation


class TUI:
    """Terminal User Interface manager for printing outputs and showing progress bars."""

    def print_evaluation_results(
        self,
        evaluation: RecallEvaluation,
        dataset: str,
        student_search_results: str,
    ) -> None:
        """Prints the evaluation results in a formatted block.

        Args:
            evaluation: The RecallEvaluation containing evaluation metrics.
            dataset: Path or name of the dataset evaluated.
            student_search_results: Path or name of the search results evaluated.
        """
        self.print(
            dedent(f"""\
                Dataset: '{dataset}'
                Search results: '{student_search_results}'
                """)
        )
        self.print(self._format_evaluation_scores(evaluation))

    def _format_evaluation_scores(self, evaluation: RecallEvaluation) -> str:
        """Formats the evaluation scores into a readable string.

        Args:
            evaluation: The RecallEvaluation object containing the scores.

        Returns:
            A formatted multi-line string with the evaluation metrics.
        """
        return dedent(f"""\
            Student data is valid: {evaluation.data_is_valid}
            Total number of questions: {evaluation.number_of_questions}
            Total number of questions with sources: {
            evaluation.number_of_questions_with_sources
        }
            Total number of questions with student sources: {
            evaluation.number_of_questions_with_student_sources
        }

            Recall@1: {evaluation.recall_1:.3f}
            Recall@3: {evaluation.recall_3:.3f}
            Recall@5: {evaluation.recall_5:.3f}
            Recall@10: {evaluation.recall_10:.3f}""")

    def print(self, data: str) -> None:
        """Prints the provided data to standard output.

        Args:
            data: The string content to print.
        """
        print(data)

    @contextmanager
    def progress(self, desc: str, total: int, unit: str) -> Generator[Any]:
        """Provides a context manager for showing a tqdm progress bar.

        Args:
            desc: The description message shown next to the progress bar.
            total: The total number of items to iterate.
            unit: The name of the unit of progress (e.g., 'query').

        Yields:
            The tqdm progress bar instance.
        """
        pbar = tqdm(total=total, desc=desc, unit=unit)
        try:
            yield pbar
        finally:
            pbar.close()


_tui_instance = None


def get_tui() -> TUI:
    """Returns the singleton instance of the TUI.

    Returns:
        The TUI instance.
    """
    global _tui_instance

    if _tui_instance is None:
        _tui_instance = TUI()

    return _tui_instance
