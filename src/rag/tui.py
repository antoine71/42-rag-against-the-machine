from collections.abc import Generator
from contextlib import contextmanager
from textwrap import dedent

from tqdm import tqdm

from rag.evaluation.evaluation_processor import RecallEvaluation


class TUI:
    def print_evaluation_results(
        self,
        evaluation: RecallEvaluation,
        dataset: str,
        student_search_results: str,
    ) -> None:
        self.print(
            dedent(f"""\
                Dataset: '{dataset}'")
                Search results: '{student_search_results}'")
                """)
        )
        self.print(self._format_evaluation_scores(evaluation))

    def _format_evaluation_scores(self, evaluation: RecallEvaluation) -> str:
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
            Recall@10: {evaluation.recall_10:.3f}\
        """)

    def print(self, data: str) -> None:
        print(data)

    @contextmanager
    def progress(self, desc: str, total: int, unit: str) -> Generator[tqdm]:
        pbar = tqdm(total=total, desc=desc, unit=unit)
        try:
            yield pbar
        finally:
            pbar.close()


_tui_instance = None


def get_tui() -> TUI:
    global _tui_instance

    if _tui_instance is None:
        _tui_instance = TUI()

    return _tui_instance
