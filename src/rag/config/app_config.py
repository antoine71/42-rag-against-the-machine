from pydantic_settings import BaseSettings


class AppConfig(BaseSettings):
    """Application configuration settings for logging and environment."""

    log_level: str = "INFO"
