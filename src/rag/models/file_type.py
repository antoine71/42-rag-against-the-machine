from enum import Enum


class FileType(str, Enum):
    PYTHON = "python"
    MARKDOWN = "markdown"
    TEXT = "text"

    @property
    def suffix(self) -> str:
        return {
            FileType.PYTHON: ".py",
            FileType.MARKDOWN: ".md",
            FileType.TEXT: ".txt",
        }[self]

    @classmethod
    def from_suffix(cls, suffix: str) -> "FileType":
        return {
            ".py": cls.PYTHON,
            ".md": cls.MARKDOWN,
            ".txt": cls.TEXT,
        }[suffix]
