from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_ENV_FILE, extra="ignore")

    llama_embed_base_url: str = "http://127.0.0.1:8081"
    llama_token_embed_base_url: str = "http://192.168.31.92:8082"
    llama_chat_base_url: str = "http://127.0.0.1:8080"
    llama_embed_api_key: str = ""
    llama_token_embed_api_key: str = ""
    llama_chat_api_key: str = ""
    llama_embed_model: str = ""
    llama_chat_model: str = ""
    embed_document_prefix: str = ""
    embed_query_prefix: str = ""
    data_dir: Path = Path("./data")
    database_path: Path = Path("./data/rag-visualizer.db")
    max_upload_bytes: int = 20 * 1024 * 1024
    chat_context_tokens: int = 4096
    answer_max_tokens: int = 384
    token_embedding_dimensions: int = 384
    cors_origins: str = "http://localhost:5173"
    request_timeout_seconds: float = 120.0

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def allowed_origins(self) -> list[str]:
        return [value.strip() for value in self.cors_origins.split(",") if value.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
