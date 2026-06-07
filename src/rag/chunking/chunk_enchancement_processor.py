from pathlib import Path

from rag.models.chunk import Chunk
from rag.tui import TUI


class ChunkEnhancementProcessor:
    """Base processor for enriching chunks after splitting."""

    def __init__(self, tui: TUI) -> None:
        """Initializes the chunk enhancement processor.

        Args:
            tui: Terminal UI used for progress output.
        """
        self._tui = tui

    def process_list(self, chunks: list[Chunk]) -> list[Chunk]:
        """Processes a list of chunks.

        Args:
            chunks: Chunks to enhance.

        Returns:
            Enhanced chunks.
        """
        processed_chunks: list[Chunk] = []
        with self._tui.progress(
            self.__class__.__name__, len(chunks), "chunk"
        ) as progess:
            for chunk in chunks:
                processed_chunks.append(self.process(chunk))
                progess.update(1)
        return processed_chunks

    def process(self, chunk: Chunk) -> Chunk:
        """Processes a single chunk.

        Args:
            chunk: Chunk to enhance.

        Returns:
            Enhanced chunk.
        """
        return self._process_core(chunk)

    def _process_core(self, chunk: Chunk) -> Chunk:
        """Runs the concrete enhancement logic.

        Args:
            chunk: Chunk to enhance.

        Returns:
            Enhanced chunk.
        """
        return chunk


class FilePathExpansionProcessor(ChunkEnhancementProcessor):
    """Prepends path-derived breadcrumbs to chunk text."""

    def _generate_breadcrumbs(self, file_path: Path) -> str:
        """Builds search-friendly breadcrumbs from a file path.

        Args:
            file_path: Path of the chunk source file.

        Returns:
            Space-separated path components without the file suffix.
        """
        relative_path = file_path.relative_to("data/raw/vllm-0.10.1")
        path_components = list(relative_path.parts[:-1])
        path_components.append(relative_path.stem)
        return " ".join(c.replace("_", " ") for c in path_components)

    def _process_core(self, chunk: Chunk) -> Chunk:
        """Prepends generated breadcrumbs to a chunk.

        Args:
            chunk: Chunk to enrich.

        Returns:
            The enriched chunk.
        """
        breadcrumbs = self._generate_breadcrumbs(Path(chunk.file_path))
        chunk.text = breadcrumbs + "\n" + chunk.text
        return chunk
