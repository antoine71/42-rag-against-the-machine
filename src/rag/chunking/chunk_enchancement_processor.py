from pathlib import Path

from rag.models.chunk import Chunk
from rag.tui import TUI


class ChunkEnhancementProcessor:
    def __init__(self, tui: TUI) -> None:
        self._tui = tui

    def process_list(self, chunks: list[Chunk]) -> list[Chunk]:
        processed_chunks: list[Chunk] = []
        with self._tui.progress(
            self.__class__.__name__, len(chunks), "chunk"
        ) as progess:
            for chunk in chunks:
                processed_chunks.append(self.process(chunk))
                progess.update(1)
        return processed_chunks

    def process(self, chunk: Chunk) -> Chunk:
        return self._process_core(chunk)

    def _process_core(self, chunk: Chunk) -> Chunk:
        return chunk


class FilePathExpansionProcessor(ChunkEnhancementProcessor):
    def _generate_breadcrumbs(self, file_path: Path) -> str:
        relative_path = file_path.relative_to("data/raw/vllm-0.10.1")
        path_components = list(relative_path.parts[:-1])
        path_components.append(relative_path.stem)
        return " ".join(c.replace("_", " ") for c in path_components)

    def _process_core(self, chunk: Chunk) -> Chunk:
        breadcrumbs = self._generate_breadcrumbs(Path(chunk.file_path))
        chunk.text = breadcrumbs + "\n" + chunk.text
        return chunk
