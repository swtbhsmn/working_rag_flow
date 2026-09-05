from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class ModelInfo(BaseModel):
    role: Literal["chat", "embedding", "token_embedding"]
    ready: bool
    base_url: str
    model_id: str | None = None
    context_size: int | None = None
    embedding_dimensions: int | None = None
    error: str | None = None


class SystemModels(BaseModel):
    chat: ModelInfo
    embedding: ModelInfo
    token_embedding: ModelInfo


class ContextualTokenRequest(BaseModel):
    input: str = Field(min_length=1, max_length=10000)
    token_index: int = Field(ge=0)
    expected_token_ids: list[int] = Field(min_length=1, max_length=4096)


class ContextualTokenResponse(BaseModel):
    token_index: int
    token_id: int
    token_piece: str
    token_count: int
    dimensions: int
    vector: list[float]
    server_url: str
    pooling: Literal["none"] = "none"
    normalized: bool = False


class JobCreated(BaseModel):
    document_id: str
    job_id: str


class ReindexCreated(BaseModel):
    document_id: str
    job_id: str


class DocumentSummary(BaseModel):
    id: str
    filename: str
    media_type: str
    size_bytes: int
    status: str
    chunk_count: int = 0
    token_count: int = 0
    chunk_size: int
    chunk_overlap: int
    model_id: str | None = None
    embedding_dim: int | None = None
    reindex_required: bool = False
    created_at: str
    error: str | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    document_ids: list[str] = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    similarity_threshold: float = Field(default=0.2, ge=-1, le=1)
    generate_answer: bool = True

    @model_validator(mode="after")
    def unique_documents(self):
        self.document_ids = list(dict.fromkeys(self.document_ids))
        return self


class SearchCreated(BaseModel):
    search_id: str


class TransformerRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    max_tokens: int = Field(default=48, ge=1, le=256)
    temperature: float = Field(default=0.7, ge=0, le=2)


class TransformerCreated(BaseModel):
    run_id: str


class StageEvent(BaseModel):
    stage: str
    status: Literal["started", "progress", "completed", "failed"]
    sequence: int
    timestamp: str
    payload: dict[str, Any] = Field(default_factory=dict)
