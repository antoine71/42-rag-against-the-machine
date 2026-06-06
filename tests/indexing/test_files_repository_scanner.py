from pathlib import Path
from unittest.mock import Mock

import pytest

from rag.indexing.files_repository_scanner import FilesRepositoryScanner
from rag.models.file_category import FileCategory


@pytest.fixture
def files_repository_scanner() -> FilesRepositoryScanner:
    return FilesRepositoryScanner(Mock(spec=Path))


def file_mock(suffix: str, size: int = 1, raise_error: bool = False) -> Mock:
    mock = Mock()
    if not raise_error:
        mock.suffix = suffix
    else:
        mock.suffix.side_effect = OSError
    mock.stat.return_value.st_size = size
    return mock


class TestFilesRepositoryScanner:
    @pytest.mark.parametrize(
        "file_path,files_category,expected",
        (
            (file_mock(".py"), FileCategory.CODE, True),
            (file_mock(".md"), FileCategory.CODE, False),
            (file_mock(".txt"), FileCategory.CODE, False),
            (file_mock(".py"), FileCategory.DOCUMENTATION, False),
            (file_mock(".py"), FileCategory.ALL, True),
            (file_mock(".py", size=0), FileCategory.ALL, False),
            (file_mock(".py", raise_error=True), FileCategory.ALL, False),
        ),
    )
    def test_is_file_valid(
        self,
        files_repository_scanner: FilesRepositoryScanner,
        file_path: Path,
        files_category: FileCategory,
        expected: bool,
    ) -> None:
        assert (
            files_repository_scanner._is_file_valid(file_path, files_category)
            is expected
        )
