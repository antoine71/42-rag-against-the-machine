import json
from pathlib import Path

from rag.models.search_result import MinimalSearchResults


class FilesManagerError(Exception):
    pass


class FilesManager:
    def save_results(
        self, results: list[MinimalSearchResults], file_path: str
    ) -> None:
        file_path_obj = Path(file_path)
        try:
            file_path_obj.parent.mkdir(parents=True, exist_ok=True)
            file_path_obj.write_text(
                json.dumps(
                    [result.model_dump() for result in results], indent=4
                )
            )
        except OSError as e:
            raise FilesManagerError(
                f"Failed to save results '{file_path}': {str(e)}"
            )
