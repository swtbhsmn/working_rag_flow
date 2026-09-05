import json
import math
from collections.abc import AsyncIterator
from typing import Any

import httpx

from .config import Settings
from .schemas import ModelInfo


class LlamaClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._contextual_cache: dict[str, tuple[list[dict[str, Any]], list[list[float]]]] = {}

    def _headers(self, role: str) -> dict[str, str]:
        if role == "embedding":
            key = self.settings.llama_embed_api_key
        elif role == "token_embedding":
            key = self.settings.llama_token_embed_api_key
        else:
            key = self.settings.llama_chat_api_key
        return {"Authorization": f"Bearer {key}"} if key else {}

    def _url(self, role: str) -> str:
        urls = {
            "embedding": self.settings.llama_embed_base_url,
            "token_embedding": self.settings.llama_token_embed_base_url,
            "chat": self.settings.llama_chat_base_url,
        }
        return urls[role].rstrip("/")

    @staticmethod
    def _token_rows(body: Any) -> list[Any]:
        if isinstance(body, dict):
            return body.get("tokens", [])
        if isinstance(body, list) and len(body) == 1 and isinstance(body[0], dict) and "tokens" in body[0]:
            return body[0]["tokens"]
        return body if isinstance(body, list) else []

    async def model_info(self, role: str) -> ModelInfo:
        base_url = self._url(role)
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                health = await client.get(f"{base_url}/health", headers=self._headers(role))
                health.raise_for_status()
                response = await client.get(f"{base_url}/v1/models", headers=self._headers(role))
                response.raise_for_status()
                data = response.json().get("data", [{}])[0]
                meta = data.get("meta") or {}
                return ModelInfo(
                    role=role,
                    ready=True,
                    base_url=base_url,
                    model_id=data.get("id"),
                    context_size=meta.get("n_ctx_train"),
                    embedding_dimensions=meta.get("n_embd") if role in {"embedding", "token_embedding"} else None,
                )
        except Exception as exc:
            return ModelInfo(role=role, ready=False, base_url=base_url, error=str(exc))

    async def tokenize(self, content: str, role: str = "embedding", add_special: bool = False) -> list[int]:
        async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
            response = await client.post(
                f"{self._url(role)}/tokenize",
                headers=self._headers(role),
                json={"content": content, "add_special": add_special},
            )
            response.raise_for_status()
            rows = self._token_rows(response.json())
            return [int(row["id"]) if isinstance(row, dict) else int(row) for row in rows]

    async def detokenize(self, tokens: list[int], role: str = "embedding") -> str:
        async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
            response = await client.post(
                f"{self._url(role)}/detokenize",
                headers=self._headers(role),
                json={"tokens": tokens},
            )
            response.raise_for_status()
            data = response.json()
            return data.get("content", "")

    async def embeddings(self, texts: list[str]) -> list[list[float]]:
        model = self.settings.llama_embed_model or "local-embedding-model"
        async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
            response = await client.post(
                f"{self._url('embedding')}/v1/embeddings",
                headers=self._headers("embedding"),
                json={"model": model, "input": texts, "encoding_format": "float"},
            )
            response.raise_for_status()
            rows = sorted(response.json()["data"], key=lambda row: row["index"])
            return [row["embedding"] for row in rows]

    async def native_pooled_embedding(self, content: str) -> list[float]:
        """Read the main server's real unnormalized pooled embedding."""
        async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
            response = await client.post(
                f"{self._url('embedding')}/embeddings",
                headers=self._headers("embedding"),
                json={"content": content, "embd_normalize": -1},
            )
            response.raise_for_status()
            body = response.json()
            rows = body.get("data", []) if isinstance(body, dict) else body
            embedding = rows[0]["embedding"] if rows else []
            if embedding and isinstance(embedding[0], list):
                embedding = embedding[0]
            if not embedding or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in embedding):
                raise ValueError("Main embedding server did not return one pooled vector")
            return [float(value) for value in embedding]

    async def contextual_embeddings(self, content: str) -> tuple[list[dict[str, Any]], list[list[float]]]:
        """Read and briefly cache real, unnormalized per-token vectors from pooling-none llama.cpp."""
        if content in self._contextual_cache:
            return self._contextual_cache[content]
        async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
            token_response = await client.post(
                f"{self._url('token_embedding')}/tokenize",
                headers=self._headers("token_embedding"),
                json={"content": content, "add_special": True, "with_pieces": True},
            )
            token_response.raise_for_status()
            raw_tokens = self._token_rows(token_response.json())
            tokens = [item if isinstance(item, dict) else {"id": item, "piece": ""} for item in raw_tokens]
            vector_response = await client.post(
                f"{self._url('token_embedding')}/embeddings",
                headers=self._headers("token_embedding"),
                json={"content": content, "embd_normalize": -1},
            )
            vector_response.raise_for_status()
            body = vector_response.json()
            rows = body.get("data", []) if isinstance(body, dict) else body
            vectors = rows[0].get("embedding", []) if rows else []
        if not tokens or not vectors or not isinstance(vectors[0], list):
            raise ValueError("Pooling-none server did not return token-level vectors")
        if len(tokens) != len(vectors):
            token_shape = [sorted(token.keys()) if isinstance(token, dict) else type(token).__name__ for token in tokens]
            raise ValueError(f"Pooling-none response has {len(vectors)} vectors for {len(tokens)} token IDs; token response shape: {token_shape}")
        parsed = [[float(value) for value in vector] for vector in vectors]
        if any(not math.isfinite(value) for vector in parsed for value in vector):
            raise ValueError("Pooling-none server returned a non-finite contextual value")
        dimensions = {len(vector) for vector in parsed}
        if dimensions != {self.settings.token_embedding_dimensions}:
            actual = next(iter(dimensions), 0)
            raise ValueError(f"Expected {self.settings.token_embedding_dimensions} dimensions from pooling-none server, received {actual}")
        if len(self._contextual_cache) >= 8:
            self._contextual_cache.pop(next(iter(self._contextual_cache)))
        self._contextual_cache[content] = (tokens, parsed)
        return tokens, parsed

    async def apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
            response = await client.post(
                f"{self._url('chat')}/apply-template",
                headers=self._headers("chat"),
                json={"messages": messages},
            )
            response.raise_for_status()
            return response.json()["prompt"]

    async def chat(self, messages: list[dict[str, str]], max_tokens: int = 384, temperature: float = 0.2) -> AsyncIterator[str]:
        model = self.settings.llama_chat_model or "local-chat-model"
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                f"{self._url('chat')}/v1/chat/completions",
                headers=self._headers("chat"),
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    data = json.loads(line[6:])
                    content = data.get("choices", [{}])[0].get("delta", {}).get("content")
                    if content:
                        yield content
