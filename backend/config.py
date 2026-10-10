from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # JWT Auth Configuration
    jwt_secret: str = "mach-curriculum-lens-secret-2026"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 480

    # Ollama Configuration
    ollama_model: str = "gemma4:12b-it-qat"
    ollama_base_url: str = "http://localhost:11434"
    ollama_embed_model: str = "nomic-embed-text"

    # Database configurations can also be moved here later

    class Config:
        env_file = ".env"
        extra = "ignore"  # Ignore extra variables in .env


settings = Settings()
