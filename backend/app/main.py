import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .config import get_settings
from .database import Database, utc_now
from .llama_client import LlamaClient
from .pipelines import Pipeline
from .schemas import (
    DocumentSummary,
    ContextualTokenRequest,
    ContextualTokenResponse,
    JobCreated,
    ReindexCreated,
    SearchCreated,
    SearchRequest,
    SystemModels,
    TransformerCreated,
    TransformerRequest,
)

settings = get_settings()
db = Database(settings.database_path)
llama = LlamaClient(settings)
pipeline = Pipeline(db, llama, settings)


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    db.initialize()
    yield


app = FastAPI(title="Local RAG Process Visualizer", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_SUFFIXES = {".pdf": "application/pdf", ".txt": "text/plain", ".md": "text/markdown", ".markdown": "text/markdown"}


@app.get("/api/system/models", response_model=SystemModels)
async def system_models() -> SystemModels:
    chat, embedding, token_embedding = await asyncio.gather(
        llama.model_info("chat"), llama.model_info("embedding"), llama.model_info("token_embedding")
    )
    return SystemModels(chat=chat, embedding=embedding, token_embedding=token_embedding)


@app.post("/api/embeddings/contextual-token", response_model=ContextualTokenResponse)
async def contextual_token(request: ContextualTokenRequest) -> ContextualTokenResponse:
    try:
        tokens, vectors = await llama.contextual_embeddings(request.input)
    except Exception as exc:
        raise HTTPException(503, f"Token-level embedding server unavailable or incompatible: {exc}") from exc
    actual_ids = [int(token.get("id", -1)) for token in tokens]
    if actual_ids != request.expected_token_ids:
        raise HTTPException(409, "Token IDs from the pooling-none server do not match the main embedding server; verify both servers use the same model and tokenizer")
    if request.token_index >= len(vectors):
        raise HTTPException(422, "token_index is outside the contextual embedding sequence")
    piece = tokens[request.token_index].get("piece", "")
    if isinstance(piece, list):
        piece = bytes(piece).decode("utf-8", errors="replace")
    vector = vectors[request.token_index]
    return ContextualTokenResponse(
        token_index=request.token_index, token_id=actual_ids[request.token_index], token_piece=str(piece),
        token_count=len(tokens), dimensions=len(vector), vector=vector,
        server_url=settings.llama_token_embed_base_url,
    )


@app.post("/api/documents", response_model=JobCreated, status_code=202)
async def upload_document(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    chunk_size: int = Form(256),
    chunk_overlap: int = Form(32),
) -> JobCreated:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(415, "Only PDF, TXT, and Markdown files are supported")
    if chunk_size < 32 or chunk_size > 2048:
        raise HTTPException(422, "chunk_size must be between 32 and 2048")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise HTTPException(422, "chunk_overlap must be non-negative and smaller than chunk_size")
    embedding_info = await llama.model_info("embedding")
    if embedding_info.context_size and chunk_size > embedding_info.context_size:
        raise HTTPException(422, f"chunk_size exceeds the embedding model context limit ({embedding_info.context_size})")

    content = await file.read(settings.max_upload_bytes + 1)
    if not content:
        raise HTTPException(422, "The uploaded file is empty")
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(413, "The maximum upload size is 20 MB")
    if suffix == ".pdf" and not content.startswith(b"%PDF-"):
        raise HTTPException(415, "The file extension is PDF but its signature is invalid")

    document_id, job_id = str(uuid4()), str(uuid4())
    stored_path = settings.uploads_dir / f"{document_id}{suffix}"
    stored_path.write_bytes(content)
    db.insert_document(
        {
            "id": document_id, "filename": Path(file.filename or f"document{suffix}").name,
            "stored_path": str(stored_path), "media_type": ALLOWED_SUFFIXES[suffix], "size_bytes": len(content),
            "status": "processing", "chunk_size": chunk_size, "chunk_overlap": chunk_overlap, "created_at": utc_now(),
        }
    )
    db.create_job(job_id, "ingestion", document_id)
    background.add_task(pipeline.process_document, document_id, job_id)
    return JobCreated(document_id=document_id, job_id=job_id)


def document_summary(row: dict, current_fingerprint: str | None = None) -> DocumentSummary:
    return DocumentSummary(
        id=row["id"], filename=row["filename"], media_type=row["media_type"], size_bytes=row["size_bytes"],
        status=row["status"], chunk_count=row["chunk_count"], token_count=row["token_count"],
        chunk_size=row["chunk_size"], chunk_overlap=row["chunk_overlap"], model_id=row["model_id"],
        embedding_dim=row["embedding_dim"], reindex_required=bool(current_fingerprint and row["fingerprint"] and row["fingerprint"] != current_fingerprint),
        created_at=row["created_at"], error=row["error"],
    )


@app.get("/api/documents", response_model=list[DocumentSummary])
async def list_documents() -> list[DocumentSummary]:
    try:
        _, _, fingerprint = await pipeline.embedding_identity()
    except Exception:
        fingerprint = None
    return [document_summary(row, fingerprint) for row in db.documents()]


@app.get("/api/documents/{document_id}/pipeline")
async def document_events(document_id: str) -> dict:
    if not db.document(document_id):
        raise HTTPException(404, "Document not found")
    job = db.latest_job_for_entity(document_id, "ingestion")
    if not job:
        return {"events": []}
    return {"job_id": job["id"], "status": job["status"], "events": db.events_after(job["id"], 0)}


@app.post("/api/documents/{document_id}/reindex", response_model=ReindexCreated, status_code=202)
async def reindex_document(
    document_id: str,
    background: BackgroundTasks,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> ReindexCreated:
    document = db.document(document_id)
    if not document:
        raise HTTPException(404, "Document not found")
    if not Path(document["stored_path"]).exists():
        raise HTTPException(409, "Original upload is missing")
    next_size = chunk_size if chunk_size is not None else int(document["chunk_size"])
    next_overlap = chunk_overlap if chunk_overlap is not None else int(document["chunk_overlap"])
    if next_size < 32 or next_size > 2048 or next_overlap < 0 or next_overlap >= next_size:
        raise HTTPException(422, "Invalid chunk size or overlap")
    job_id = str(uuid4())
    db.update_document(document_id, status="processing", error=None, chunk_size=next_size, chunk_overlap=next_overlap)
    db.create_job(job_id, "ingestion", document_id)
    background.add_task(pipeline.process_document, document_id, job_id)
    return ReindexCreated(document_id=document_id, job_id=job_id)


@app.delete("/api/documents/{document_id}", status_code=204)
async def delete_document(document_id: str) -> None:
    document = db.document(document_id)
    if not document:
        raise HTTPException(404, "Document not found")
    path = Path(document["stored_path"])
    if path.exists():
        path.unlink()
    db.delete_document(document_id)


@app.post("/api/searches", response_model=SearchCreated, status_code=202)
async def create_search(request: SearchRequest, background: BackgroundTasks) -> SearchCreated:
    missing = [document_id for document_id in request.document_ids if not db.document(document_id)]
    if missing:
        raise HTTPException(404, f"Unknown document IDs: {', '.join(missing)}")
    search_id, job_id = str(uuid4()), str(uuid4())
    db.create_job(job_id, "search", search_id)
    background.add_task(pipeline.process_search, search_id, job_id, request)
    return SearchCreated(search_id=search_id)


@app.post("/api/transformer/runs", response_model=TransformerCreated, status_code=202)
async def create_transformer_run(request: TransformerRequest, background: BackgroundTasks) -> TransformerCreated:
    run_id, job_id = str(uuid4()), str(uuid4())
    db.create_job(job_id, "transformer", run_id)
    background.add_task(pipeline.process_transformer, run_id, job_id, request)
    return TransformerCreated(run_id=run_id)


async def event_stream(entity_id: str, request: Request, kind: str):
    job = db.latest_job_for_entity(entity_id, kind)
    if not job:
        yield f"event: error\ndata: {json.dumps({'detail': 'Job not found'})}\n\n"
        return
    last_sequence = 0
    while True:
        if await request.is_disconnected():
            return
        events = db.events_after(job["id"], last_sequence)
        for event in events:
            last_sequence = event["sequence"]
            yield f"id: {last_sequence}\nevent: stage\ndata: {json.dumps(event)}\n\n"
        current = db.job(job["id"])
        if current and current["status"] in {"completed", "failed"} and not db.events_after(job["id"], last_sequence):
            yield f"event: end\ndata: {json.dumps({'status': current['status']})}\n\n"
            return
        yield ": keep-alive\n\n"
        await asyncio.sleep(0.35)


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request):
    if not db.job(job_id):
        raise HTTPException(404, "Job not found")

    async def direct_stream():
        last_sequence = 0
        while True:
            if await request.is_disconnected():
                return
            events = db.events_after(job_id, last_sequence)
            for event in events:
                last_sequence = event["sequence"]
                yield f"id: {last_sequence}\nevent: stage\ndata: {json.dumps(event)}\n\n"
            current = db.job(job_id)
            if current and current["status"] in {"completed", "failed"} and not db.events_after(job_id, last_sequence):
                yield f"event: end\ndata: {json.dumps({'status': current['status']})}\n\n"
                return
            yield ": keep-alive\n\n"
            await asyncio.sleep(0.35)

    return StreamingResponse(direct_stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.get("/api/searches/{search_id}/events")
async def search_events(search_id: str, request: Request):
    return StreamingResponse(event_stream(search_id, request, "search"), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.get("/api/transformer/runs/{run_id}/events")
async def transformer_events(run_id: str, request: Request):
    return StreamingResponse(event_stream(run_id, request, "transformer"), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
