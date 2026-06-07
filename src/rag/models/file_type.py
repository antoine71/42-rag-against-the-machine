from enum import Enum


class FileType(str, Enum):
    """Supported concrete file types for indexing."""

    PYTHON = "python"
    MARKDOWN = "markdown"
    TEXT = "text"

    @property
    def suffix(self) -> str:
        """Returns the filesystem suffix for the file type.

        Returns:
            The suffix associated with the file type.
        """
        return {
            FileType.PYTHON: ".py",
            FileType.MARKDOWN: ".md",
            FileType.TEXT: ".txt",
        }[self]

    @classmethod
    def from_suffix(cls, suffix: str) -> "FileType":
        """Builds a file type from a filesystem suffix.

        Args:
            suffix: File suffix including the leading dot.

        Returns:
            The matching file type.

        Raises:
            KeyError: If the suffix is not supported.
        """
        return {
            ".py": cls.PYTHON,
            ".md": cls.MARKDOWN,
            ".txt": cls.TEXT,
        }[suffix]
