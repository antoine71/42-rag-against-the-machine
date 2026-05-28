from pydantic_settings import BaseSettings


class RRFConfig(BaseSettings):
    k_factor: int = 4
