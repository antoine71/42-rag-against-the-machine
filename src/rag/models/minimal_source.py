from pydantic import BaseModel


class MinimalSource(BaseModel, frozen=True):
    file_path: str
    first_character_index: int
    last_character_index: int
