from pydantic import BaseModel


class MinimalSource(BaseModel, frozen=True):
    """Represents a text fragment source within a file using character
    offsets.
    """

    file_path: str
    first_character_index: int
    last_character_index: int
