from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from tqdm import tqdm

from rag.evaluation.evaluation_processor import RecallEvaluation


class TUI:
    """Terminal User Interface manager for printing outputs and showing
    progress bars.
    """

    WIDTH = 100

    def __init__(self) -> None:
        """Initializes the terminal UI helper."""
        pass

    def print_evaluation_results(self, evaluation: RecallEvaluation) -> None:
        """Prints evaluation scores as a readable terminal report.

        Args:
            evaluation: Evaluation metrics and dataset validity counters.
        """
        padding = int(self.WIDTH // (3 / 2))
        data = [
            ("Student data is valid", evaluation.data_is_valid),
            ("Total number of questions", evaluation.number_of_questions),
            (
                "Total number of questions with source",
                evaluation.number_of_questions_with_sources,
            ),
            (
                "Total number of quesitons with student source",
                evaluation.number_of_questions_with_student_sources,
            ),
        ]
        for key, value in data:
            print(f"{key:<{padding}} {value}")
        print()
        metrics = [
            (1, evaluation.recall_1),
            (3, evaluation.recall_3),
            (5, evaluation.recall_5),
            (10, evaluation.recall_10),
        ]
        for recall, recall_value in metrics:
            print(
                f"{'Recall@' + str(recall):<{padding}} "
                f"{float(recall_value):.03f}"
            )

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
        pbar = tqdm(total=total, desc=desc, unit=unit, leave=False)
        try:
            yield pbar
        finally:
            pbar.close()

    def _format_subtitle(self, subtitle: tuple[str, str | int]) -> str:
        """Formats a title subtitle key-value pair.

        Args:
            subtitle: Pair containing the subtitle label and value.

        Returns:
            The formatted subtitle line.
        """
        title, value = subtitle
        return f"{title:<{self.WIDTH // 5}}{value:>{4 * self.WIDTH // 5}}"

    def print_title(
        self, rag_step: str, *subtitles: tuple[str, str | int]
    ) -> None:
        """Prints a section title for a RAG pipeline step.

        Args:
            rag_step: Name of the current RAG step.
            *subtitles: Optional subtitle key-value pairs to display.
        """
        bar = self.WIDTH * "="
        title = f"RAG {rag_step.upper()}"
        formatted_subtitles = [
            self._format_subtitle(subtitle) for subtitle in subtitles
        ]
        print(bar)
        print(title)
        for subtitle in formatted_subtitles:
            print(subtitle)
        print(bar)
        print()

    def print_phase_title(self, phase_title: str) -> None:
        """Prints the title of an execution phase.

        Args:
            phase_title: Human-readable phase name.
        """
        title = f"{phase_title}"
        bar = self.WIDTH * "-"
        print(title)
        print(bar)

    def print_task_report(
        self,
        title: str,
        delta_time_ms: int,
        unit: str = "",
        data: int = 0,
        new_line_before: bool = False,
        new_line_after: bool = False,
    ) -> None:
        """Prints a one-line task completion report.

        Args:
            title: Name of the completed task.
            delta_time_ms: Elapsed task time in milliseconds.
            unit: Unit name associated with the count.
            data: Count to display for the task.
            new_line_before: Whether to print a blank line before the report.
            new_line_after: Whether to print a blank line after the report.
        """
        task_display = f"✓ {title.capitalize()} completed"
        value = f"{data} {unit}"
        total_secondes = delta_time_ms / 1000
        minutes = int(total_secondes // 60)
        secondes = total_secondes % 60
        if minutes:
            timestamp = f"{minutes}m{secondes:.0f}s"
        else:
            timestamp = f"{secondes:.02f}s"

        if new_line_before:
            self.new_line()
        print(
            f"{task_display:<{self.WIDTH // 2}}{value:<{self.WIDTH // 4}}"
            f"{timestamp:<{self.WIDTH // 4}}"
        )
        if new_line_after:
            self.new_line()

    def new_line(self) -> None:
        """Prints a blank line."""
        print()

    def _format_summary_data(self, data: tuple[str, str]) -> str:
        """Formats a summary key-value pair.

        Args:
            data: Pair containing the summary label and value.

        Returns:
            The formatted summary line.
        """
        key, value = data
        return f"{key:<{self.WIDTH // 2}{value}}"

    def print_summary(self, duration: int, *data: tuple[str, str]) -> None:
        """Prints a final execution summary.

        Args:
            duration: Total execution duration in milliseconds.
            *data: Summary key-value pairs to display.
        """
        bar = self.WIDTH * "="
        title = "SUMMARY"
        formatted_data = [self._format_summary_data(d) for d in data]
        print(bar)
        print(title)
        for d in formatted_data:
            print(d)
        print(bar)


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
