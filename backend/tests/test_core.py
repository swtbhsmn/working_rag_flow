from pathlib import Path
from uuid import uuid4

import numpy as np

from app.config import Settings
from app.database import Database, utc_now
from app.documents import DocumentError, clean_text, extract_document, structured_units, text_statistics
from app.llama_client import LlamaClient
from app.pipelines import Pipeline
from app.schemas import ModelInfo, SearchRequest


def test_clean_text_and_statistics():
    cleaned = clean_text(" First   sentence.\r\n\r\n\r\nSecond paragraph! \x00")
    assert cleaned == "First sentence.\n\nSecond paragraph!"
    assert text_statistics(cleaned) == {"characters": 34, "paragraphs": 2, "sentences": 2}
    units = structured_units("# Heading\n\nFirst sentence. Second sentence!")
    assert [unit["kind"] for unit in units] == ["heading", "sentence", "sentence"]


def test_text_extraction_rejects_empty(tmp_path: Path):
    path = tmp_path / "empty.txt"
    path.write_text("  \n", encoding="utf-8")
    try:
        extract_document(path, ".txt")
    except DocumentError as exc:
        assert "No extractable text" in str(exc)
    else:
        raise AssertionError("empty input should fail")


def test_database_vector_round_trip(tmp_path: Path):
    db = Database(tmp_path / "test.db")
    db.initialize()
    document_id = str(uuid4())
    db.insert_document({
        "id": document_id, "filename": "notes.md", "stored_path": "/tmp/notes.md",
        "media_type": "text/markdown", "size_bytes": 10, "status": "ready",
        "chunk_size": 256, "chunk_overlap": 32, "created_at": utc_now(),
    })
    vector = np.array([0.6, 0.8], dtype=np.float32)
    db.replace_chunks(document_id, [{
        "id": "chunk-1", "document_id": document_id, "chunk_index": 0, "page": 1,
        "text": "hello", "token_count": 1, "vector": vector.tobytes(),
    }])
    loaded = db.chunks_for([document_id])
    np.testing.assert_allclose(loaded[0]["vector"], vector)


def test_projection_and_prompt_are_deterministic():
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    vectors = np.array([[0.9, 0.1, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    points = Pipeline._project(query, vectors)
    assert [point["id"] for point in points] == ["query", "candidate-1", "candidate-2"]
    chunks = [{"marker": "[D101:P1:C1]", "filename": "a.txt", "text": "Grounded fact."}]
    prompt = Pipeline._build_prompt("What is true?", chunks)
    assert "[D101:P1:C1]" in prompt
    assert "using only the supplied sources" in prompt
    assert prompt.endswith("What is true?\n\nANSWER")


class FakeLlama:
    async def model_info(self, role: str):
        return ModelInfo(role=role, ready=True, base_url="http://local", model_id=f"fake-{role}", context_size=512, embedding_dimensions=3 if role == "embedding" else None)

    async def tokenize(self, content: str, role: str = "embedding", add_special: bool = False):
        return list(content.encode("utf-8"))

    async def detokenize(self, tokens: list[int], role: str = "embedding"):
        return bytes(tokens).decode("utf-8", errors="ignore")

    async def embeddings(self, texts: list[str]):
        vectors = []
        for text in texts:
            vector = np.array([len(text), text.lower().count("local") + 1, text.count(" ") + 1], dtype=float)
            vectors.append((vector / np.linalg.norm(vector)).tolist())
        return vectors

    async def chat(self, messages, max_tokens=384, temperature=0.2):
        yield "A grounded answer [D1:P1:C1]."

    async def apply_chat_template(self, messages):
        return "<chat>" + messages[0]["content"] + "</chat>"


async def test_real_pipeline_events_with_mocked_llama(tmp_path: Path):
    settings = Settings(
        _env_file=None, data_dir=tmp_path, database_path=tmp_path / "pipeline.db",
        embed_document_prefix="passage: ", chat_context_tokens=400, answer_max_tokens=48,
    )
    db = Database(settings.database_path)
    db.initialize()
    source = tmp_path / "source.txt"
    source.write_text("Local models keep documents private. " * 8, encoding="utf-8")
    document_id = str(uuid4())
    db.insert_document({
        "id": document_id, "filename": "source.txt", "stored_path": str(source), "media_type": "text/plain",
        "size_bytes": source.stat().st_size, "status": "processing", "chunk_size": 64, "chunk_overlap": 8,
        "created_at": utc_now(),
    })
    pipeline = Pipeline(db, FakeLlama(), settings)  # type: ignore[arg-type]
    ingest_job = str(uuid4())
    db.create_job(ingest_job, "ingestion", document_id)
    await pipeline.process_document(document_id, ingest_job)
    assert db.document(document_id)["status"] == "ready"
    assert db.document(document_id)["chunk_count"] > 1
    stages = [event["stage"] for event in db.events_after(ingest_job, 0)]
    assert stages[0] == "validation"
    assert stages[-1] == "complete"
    assert {"extraction", "tokenization", "embedding", "persistence"}.issubset(stages)
    token_event = next(event for event in db.events_after(ingest_job, 0) if event["stage"] == "tokenization" and event["status"] == "completed")
    assert bytes(token_event["payload"]["token_ids"]).startswith(b"passage: ")
    assert max(chunk["token_count"] for chunk in db.chunks_for([document_id])) <= 64

    search_id, search_job = str(uuid4()), str(uuid4())
    db.create_job(search_job, "search", search_id)
    await pipeline.process_search(search_id, search_job, SearchRequest(query="Are local documents private?", document_ids=[document_id], top_k=5, generate_answer=True))
    search_events = db.events_after(search_job, 0)
    selected = next(event for event in search_events if event["stage"] == "selection" and event["status"] == "completed")
    assert selected["payload"]["selected"]
    assert any(event["stage"] == "selection" and event["status"] == "progress" for event in search_events)
    assert any(event["stage"] == "prompt_budget" and event["status"] == "progress" and event["payload"]["phase"] == "tokenization" for event in search_events)
    budget = next(event for event in search_events if event["stage"] == "prompt_budget" and event["status"] == "completed")
    assert budget["payload"]["prompt_tokens"] + budget["payload"]["answer_reserve"] <= budget["payload"]["context_limit"]
    generated = next(event for event in search_events if event["stage"] == "generation" and event["status"] == "completed")
    assert generated["payload"]["citations"] == ["[D1:P1:C1]"]
    assert search_events[-1]["stage"] == "complete"


def test_citation_markers_are_unique_per_selected_document():
    base = {"id": "c", "filename": "a.txt", "page": 1, "chunk_index": 0, "document_id": "doc-a", "text": "x", "token_count": 1}
    first = Pipeline._ranked_chunk(base, 0.9, 1, 1)
    second = Pipeline._ranked_chunk({**base, "id": "c2", "document_id": "doc-b"}, 0.8, 2, 2)
    assert first["marker"] == "[D1:P1:C1]"
    assert second["marker"] == "[D2:P1:C1]"


async def test_pooling_none_response_returns_real_token_rows(monkeypatch):
    vector = [float(index) / 384 for index in range(384)]

    class FakeResponse:
        def __init__(self, body):
            self.body = body

        def raise_for_status(self):
            return None

        def json(self):
            return self.body

    class FakeHttpClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, **kwargs):
            if url.endswith("/tokenize"):
                return FakeResponse({"tokens": [{"id": 10, "piece": "real"}, {"id": 11, "piece": " vector"}]})
            return FakeResponse([{"index": 0, "embedding": [vector, vector]}])

    monkeypatch.setattr("app.llama_client.httpx.AsyncClient", FakeHttpClient)
    client = LlamaClient(Settings(_env_file=None, token_embedding_dimensions=384))
    tokens, vectors = await client.contextual_embeddings("real vector")
    assert [token["id"] for token in tokens] == [10, 11]
    assert len(vectors) == 2
    assert len(vectors[0]) == 384
    assert vectors[0][127] == vector[127]
