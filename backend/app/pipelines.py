import asyncio
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from .config import Settings
from .database import Database
from .documents import clean_text, extract_document, structured_units, text_statistics
from .llama_client import LlamaClient
from .schemas import SearchRequest, TransformerRequest


class Pipeline:
    PIPELINE_VERSION = 3

    def __init__(self, db: Database, llama: LlamaClient, settings: Settings):
        self.db = db
        self.llama = llama
        self.settings = settings

    def emit(self, job_id: str, stage: str, status: str, **payload: Any) -> None:
        self.db.add_event(job_id, stage, status, payload)

    async def embedding_identity(self) -> tuple[str, int | None, str]:
        info = await self.llama.model_info("embedding")
        model_id = info.model_id or self.settings.llama_embed_model or "unknown"
        fingerprint_input = {
            "pipeline_version": self.PIPELINE_VERSION,
            "model": model_id,
            "dimension": info.embedding_dimensions,
            "document_prefix": self.settings.embed_document_prefix,
            "query_prefix": self.settings.embed_query_prefix,
        }
        fingerprint = hashlib.sha256(json.dumps(fingerprint_input, sort_keys=True).encode()).hexdigest()
        return model_id, info.embedding_dimensions, fingerprint

    async def _chunk_page(self, text: str, page: int, chunk_size: int, overlap: int) -> list[dict[str, Any]]:
        prefix = self.settings.embed_document_prefix
        prefix_tokens = await self.llama.tokenize(prefix, "embedding", add_special=True)
        content_budget = chunk_size - len(prefix_tokens)
        if content_budget < 16:
            raise ValueError(f"The document prefix leaves only {content_budget} tokens for chunk content")
        units = structured_units(text) or [{"kind": "paragraph", "text": text}]
        chunks: list[dict[str, Any]] = []

        async def append_chunk(content: str, boundary: str, overlap_ids: list[int]) -> None:
            input_text = prefix + content
            token_ids = await self.llama.tokenize(input_text, "embedding", add_special=True)
            if len(token_ids) > chunk_size:
                raise ValueError(f"Exact embedding input contains {len(token_ids)} tokens, exceeding chunk size {chunk_size}")
            chunks.append({
                "page": page, "text": content, "token_count": len(token_ids), "token_ids": token_ids,
                "content_token_count": len(await self.llama.tokenize(content, "embedding")),
                "boundary": boundary, "overlap_token_ids": overlap_ids,
            })

        current = ""
        current_boundary = "paragraph"
        current_overlap_ids: list[int] = []
        for unit in units:
            separator = "\n\n" if current else ""
            candidate = current + separator + unit["text"]
            if len(await self.llama.tokenize(prefix + candidate, "embedding", add_special=True)) <= chunk_size:
                current, current_boundary = candidate, unit["kind"]
                continue
            overlap_ids: list[int] = []
            if current:
                await append_chunk(current, current_boundary, current_overlap_ids)
                current_ids = await self.llama.tokenize(current, "embedding")
                overlap_ids = current_ids[-overlap:] if overlap else []
            overlap_text = await self.llama.detokenize(overlap_ids, "embedding") if overlap_ids else ""
            next_text = overlap_text + ("\n\n" if overlap_text else "") + unit["text"]
            if len(await self.llama.tokenize(prefix + next_text, "embedding", add_special=True)) <= chunk_size:
                current, current_boundary = next_text, unit["kind"]
                current_overlap_ids = overlap_ids
                if overlap_ids:
                    # Stored when this chunk is finalized.
                    current_boundary = f"{unit['kind']} + overlap"
                continue

            unit_tokens = overlap_ids + await self.llama.tokenize(unit["text"], "embedding")
            available = max(1, content_budget)
            step = max(1, available - overlap)
            for offset in range(0, len(unit_tokens), step):
                window = unit_tokens[offset : offset + available]
                if not window:
                    break
                decoded = await self.llama.detokenize(window, "embedding")
                actual = await self.llama.tokenize(prefix + decoded, "embedding", add_special=True)
                while len(actual) > chunk_size and window:
                    window = window[:-1]
                    decoded = await self.llama.detokenize(window, "embedding")
                    actual = await self.llama.tokenize(prefix + decoded, "embedding", add_special=True)
                window_overlap = overlap_ids if offset == 0 else (window[:overlap] if overlap else [])
                chunks.append({
                    "page": page, "text": decoded, "token_count": len(actual), "token_ids": actual,
                    "content_token_count": len(window), "boundary": "token split", "overlap_token_ids": window_overlap,
                })
                if offset + available >= len(unit_tokens):
                    break
            current = ""
            current_overlap_ids = []
        if current:
            await append_chunk(current, current_boundary, current_overlap_ids)
        return chunks

    async def process_document(self, document_id: str, job_id: str) -> None:
        started = time.perf_counter()
        document = self.db.document(document_id)
        if not document:
            return
        try:
            path = Path(document["stored_path"])
            self.emit(job_id, "validation", "started", filename=document["filename"])
            if not path.exists() or path.stat().st_size == 0:
                raise ValueError("Uploaded file is empty or missing")
            self.emit(
                job_id, "validation", "completed",
                input={"filename": document["filename"], "media_type": document["media_type"], "size_bytes": path.stat().st_size},
                process="Verify that the staged file exists, is non-empty, and matches the accepted upload constraints.",
                output={"valid": True, "safe_filename": Path(document["filename"]).name},
                size_bytes=path.stat().st_size, media_type=document["media_type"],
            )

            self.emit(job_id, "extraction", "started")
            pages = await asyncio.to_thread(extract_document, path, path.suffix.lower())
            raw_text = "\n\n".join(str(page["text"]) for page in pages)
            self.emit(
                job_id, "extraction", "completed",
                input={"media_type": document["media_type"], "bytes": path.stat().st_size},
                process="Parse the file by type and preserve page boundaries while extracting text.",
                output={"pages": len(pages), "characters": len(raw_text), "preview": raw_text[:600]},
                pages=len(pages), preview=raw_text[:600], characters=len(raw_text),
            )

            self.emit(job_id, "cleaning", "started", before=raw_text[:300])
            cleaned_pages = [{"page": page["page"], "text": clean_text(str(page["text"]))} for page in pages]
            cleaned_text = "\n\n".join(str(page["text"]) for page in cleaned_pages if page["text"])
            stats = text_statistics(cleaned_text)
            unit_count = sum(len(structured_units(str(page["text"]))) for page in cleaned_pages)
            self.emit(
                job_id, "cleaning", "completed",
                input={"characters": len(raw_text), "preview": raw_text[:300]},
                process="Normalize Unicode whitespace, line endings, blank lines, and non-content control characters.",
                output={"characters": len(cleaned_text), "removed_characters": len(raw_text) - len(cleaned_text), "preview": cleaned_text[:300]},
                after=cleaned_text[:300], removed_characters=len(raw_text) - len(cleaned_text),
            )
            self.emit(
                job_id, "structure", "completed",
                input={"pages": len(cleaned_pages), "characters": len(cleaned_text)},
                process="Detect headings, paragraphs, and sentence boundaries before falling back to token windows.",
                output={**stats, "structural_units": unit_count},
                **stats, structural_units=unit_count, strategy="headings → paragraphs → sentences → token fallback",
            )

            chunk_size = int(document["chunk_size"])
            overlap = int(document["chunk_overlap"])
            self.emit(
                job_id, "chunking", "started",
                input={"characters": len(cleaned_text), "structural_units": unit_count},
                process="Pack structural units into bounded token windows and carry overlap into the next chunk.",
                chunk_size=chunk_size, overlap=overlap,
            )
            chunk_specs: list[dict[str, Any]] = []
            total_tokens = 0
            for page in cleaned_pages:
                if not page["text"]:
                    continue
                token_ids = await self.llama.tokenize(str(page["text"]), "embedding", add_special=True)
                total_tokens += len(token_ids)
                chunk_specs.extend(await self._chunk_page(str(page["text"]), int(page["page"]), chunk_size, overlap))
            if not chunk_specs:
                raise ValueError("Tokenization produced no chunks")
            overlap_example = {
                "token_ids": chunk_specs[1]["overlap_token_ids"],
                "text": await self.llama.detokenize(chunk_specs[1]["overlap_token_ids"], "embedding") if chunk_specs[1]["overlap_token_ids"] else "",
                "from_chunk": 1, "into_chunk": 2,
            } if len(chunk_specs) > 1 else None
            chunk_preview = [
                {
                    "chunk_index": index,
                    "page": spec["page"],
                    "boundary": spec["boundary"],
                    "content_tokens": spec["content_token_count"],
                    "overlap_tokens": len(spec["overlap_token_ids"]),
                    "text": spec["text"][:320],
                }
                for index, spec in enumerate(chunk_specs[:8])
            ]
            self.emit(
                job_id, "chunking", "completed",
                input={"chunk_size": chunk_size, "overlap_tokens": overlap},
                process="Prefer semantic structure boundaries; split oversized units by exact tokenizer windows and copy the configured overlap.",
                output={"chunk_count": len(chunk_specs), "chunks": chunk_preview, "overlap_example": overlap_example},
                chunk_count=len(chunk_specs), chunks=chunk_preview, overlap_example=overlap_example,
            )
            self.emit(
                job_id, "tokenization", "completed", token_count=total_tokens, chunk_count=len(chunk_specs),
                input={"chunk_count": len(chunk_specs), "document_prefix": self.settings.embed_document_prefix},
                process="Run the embedding model tokenizer on every exact prefixed chunk input.",
                output={"document_tokens": total_tokens, "first_chunk_token_ids": chunk_specs[0]["token_ids"]},
                token_preview=chunk_specs[0]["text"][:300], embedding_input=self.settings.embed_document_prefix + chunk_specs[0]["text"], token_ids=chunk_specs[0]["token_ids"],
                token_scope="first exact embedding input", token_ids_truncated=len(chunk_specs) > 1,
                document_prefix=self.settings.embed_document_prefix, boundaries=[spec["boundary"] for spec in chunk_specs[:20]],
            )
            metadata_preview = [
                {
                    "document_id": document_id,
                    "filename": document["filename"],
                    "media_type": document["media_type"],
                    "chunk_index": index,
                    "page": spec["page"],
                    "token_count": spec["token_count"],
                    "boundary": spec["boundary"],
                }
                for index, spec in enumerate(chunk_specs[:8])
            ]
            self.emit(
                job_id, "metadata",
                "completed",
                input={"chunks": len(chunk_specs), "document_id": document_id},
                process="Attach source identity, page, chunk order, token count, media type, and boundary strategy to every chunk.",
                output={"records": metadata_preview, "truncated": len(chunk_specs) > len(metadata_preview)},
                records=metadata_preview,
            )

            self.emit(job_id, "embedding", "started", batches=(len(chunk_specs) + 15) // 16)
            vectors: list[list[float]] = []
            for start in range(0, len(chunk_specs), 16):
                texts = [self.settings.embed_document_prefix + spec["text"] for spec in chunk_specs[start : start + 16]]
                vectors.extend(await self.llama.embeddings(texts))
                self.emit(job_id, "embedding", "progress", embedded=min(start + 16, len(chunk_specs)), total=len(chunk_specs))
            array = np.asarray(vectors, dtype=np.float32)
            if array.ndim != 2 or len(array) != len(chunk_specs):
                raise ValueError("Embedding server returned an unexpected tensor shape")
            norms = np.linalg.norm(array, axis=1)
            if np.any(norms == 0) or not np.all(np.isfinite(array)):
                raise ValueError("Embedding server returned invalid vectors")
            array = array / norms[:, None]
            model_id, _, fingerprint = await self.embedding_identity()
            vector_points = self._project_embeddings(array)
            self.emit(
                job_id, "embedding", "completed",
                input={"chunks": len(chunk_specs), "model_input_example": self.settings.embed_document_prefix + chunk_specs[0]["text"][:300]},
                process="Embed chunks in batches, reject invalid tensors, then L2-normalize each vector for cosine search.",
                output={"vectors": len(array), "dimensions": int(array.shape[1]), "first_vector_preview": array[0, :12].round(6).tolist()},
                dimensions=int(array.shape[1]), norm_min=float(norms.min()), norm_max=float(norms.max()),
                vector_preview=array[0, :12].round(6).tolist(), points=vector_points,
            )

            self.emit(job_id, "storage", "started")
            stored_chunks = []
            for index, (spec, vector) in enumerate(zip(chunk_specs, array, strict=True)):
                stored_chunks.append(
                    {
                        "id": str(uuid4()), "document_id": document_id, "chunk_index": index,
                        "page": spec["page"], "text": spec["text"], "token_count": spec["token_count"],
                        "vector": vector.astype(np.float32).tobytes(),
                    }
                )
            self.db.replace_chunks(document_id, stored_chunks)
            self.db.update_document(
                document_id,
                status="ready", extracted_text=cleaned_text, chunk_count=len(stored_chunks), token_count=total_tokens,
                model_id=model_id, embedding_dim=int(array.shape[1]), fingerprint=fingerprint, error=None,
            )
            stored_records = [
                {
                    "id": chunk["id"],
                    "document_id": document_id,
                    "chunk_index": chunk["chunk_index"],
                    "page": chunk["page"],
                    "text": chunk["text"],
                    "token_count": chunk["token_count"],
                    "embedding_dimensions": int(array.shape[1]),
                    "vector": array[index].round(8).tolist(),
                }
                for index, chunk in enumerate(stored_chunks)
            ]
            stored_record = stored_records[0]
            self.emit(
                job_id, "storage", "completed",
                input={"records": len(stored_chunks), "vector_dimensions": int(array.shape[1])},
                process="Atomically replace prior chunk rows and persist normalized float32 vectors with their source records.",
                output={"rows_written": len(stored_chunks), "database": "SQLite local vector store", "dimensions_per_vector": int(array.shape[1])},
                rows=len(stored_chunks), database="SQLite local vector store", stored_record=stored_record,
                stored_records=stored_records,
            )
            self.emit(
                job_id, "index", "completed",
                input={"document_id": document_id, "stored_vectors": len(stored_chunks)},
                process="Commit the document lookup index and expose the normalized vector matrix to exact cosine similarity search.",
                output={"searchable": True, "lookup_index": "chunks_document_idx", "vector_index": "exact cosine scan"},
                searchable=True, lookup_index="chunks_document_idx", vector_index="exact cosine scan",
            )
            self.emit(job_id, "complete", "completed", duration_ms=round((time.perf_counter() - started) * 1000), chunks=len(stored_chunks), tokens=total_tokens, model=model_id)
        except Exception as exc:
            self.db.update_document(document_id, status="failed", error=str(exc))
            self.emit(job_id, "pipeline", "failed", message=str(exc), type=type(exc).__name__)

    async def process_search(self, search_id: str, job_id: str, request: SearchRequest) -> None:
        started = time.perf_counter()
        try:
            documents = [self.db.document(document_id) for document_id in request.document_ids]
            if any(doc is None or doc["status"] != "ready" for doc in documents):
                raise ValueError("Every selected document must exist and be fully indexed")
            dimensions = {doc["embedding_dim"] for doc in documents if doc}
            fingerprints = {doc["fingerprint"] for doc in documents if doc}
            _, _, current_fingerprint = await self.embedding_identity()
            if len(dimensions) != 1 or len(fingerprints) != 1 or current_fingerprint not in fingerprints:
                raise ValueError("Selected documents use incompatible embeddings and must be reindexed")

            normalized_query = " ".join(request.query.split())
            self.emit(job_id, "query_normalization", "completed", original=request.query, normalized=normalized_query)
            query_input = self.settings.embed_query_prefix + normalized_query
            tokens = await self.llama.tokenize(query_input, "embedding", add_special=True)
            self.emit(job_id, "query_tokenization", "completed", token_count=len(tokens), token_ids=tokens, embedding_input=query_input, prefix=self.settings.embed_query_prefix, token_ids_truncated=False)
            self.emit(job_id, "query_embedding", "started")
            query_vector = np.asarray((await self.llama.embeddings([query_input]))[0], dtype=np.float32)
            returned_norm = float(np.linalg.norm(query_vector))
            if not np.isfinite(query_vector).all() or returned_norm == 0:
                raise ValueError("Embedding server returned an invalid query vector")
            query_vector /= returned_norm
            raw_cls: list[float] | None = None
            raw_cls_error: str | None = None
            try:
                raw_cls = await self.llama.native_pooled_embedding(query_input)
            except Exception as exc:
                raw_cls_error = str(exc)
            self.emit(
                job_id, "query_embedding", "completed", dimensions=len(query_vector), vector=query_vector.round(6).tolist(),
                returned_vector_norm=returned_norm, final_vector_norm=float(np.linalg.norm(query_vector)), pooling="cls",
                raw_cls_vector=raw_cls, raw_cls_norm=float(np.linalg.norm(raw_cls)) if raw_cls else None,
                raw_cls_error=raw_cls_error, search_vector_source="/v1/embeddings then verified L2 normalization",
            )

            chunks = self.db.chunks_for(request.document_ids)
            if not chunks:
                raise ValueError("No indexed chunks were found")
            matrix = np.stack([chunk["vector"] for chunk in chunks])
            scores = matrix @ query_vector
            order = np.argsort(-scores)
            document_numbers = {document_id: index + 1 for index, document_id in enumerate(request.document_ids)}
            candidates = [self._ranked_chunk(chunks[index], float(scores[index]), rank + 1, document_numbers[chunks[index]["document_id"]]) for rank, index in enumerate(order[: min(30, len(order))])]
            self.emit(job_id, "comparison", "completed", formula="cosine(q, d) = dot(q, d) / (||q|| × ||d||)", comparisons=len(chunks), candidates=candidates)

            self.emit(job_id, "selection", "started", message="Applying score threshold and source diversity", top_k=request.top_k)
            chosen: list[dict[str, Any]] = []
            per_document: dict[str, int] = {}
            diversity_deferred = 0
            for index in order:
                chunk = chunks[int(index)]
                score = float(scores[int(index)])
                if score < request.similarity_threshold:
                    continue
                if len(request.document_ids) > 1 and per_document.get(chunk["document_id"], 0) >= 2:
                    diversity_deferred += 1
                    continue
                chosen.append(self._ranked_chunk(chunk, score, len(chosen) + 1, document_numbers[chunk["document_id"]]))
                per_document[chunk["document_id"]] = per_document.get(chunk["document_id"], 0) + 1
                if len(chosen) == request.top_k:
                    break
            if len(chosen) < request.top_k:
                selected_ids = {item["chunk_id"] for item in chosen}
                for index in order:
                    chunk = chunks[int(index)]
                    score = float(scores[int(index)])
                    if score >= request.similarity_threshold and chunk["id"] not in selected_ids:
                        chosen.append(self._ranked_chunk(chunk, score, len(chosen) + 1, document_numbers[chunk["document_id"]]))
                    if len(chosen) == request.top_k:
                        break
            self.emit(
                job_id, "selection", "progress", message="Initial context selected; checking the chat context window",
                candidate_count=len(candidates), selected_count=len(chosen), deferred_for_source_diversity=diversity_deferred,
            )
            self.emit(job_id, "prompt_budget", "started", phase="model_metadata", message="Reading chat-model context metadata")
            chat_info = await self.llama.model_info("chat")
            detected_context = chat_info.context_size if chat_info.ready else None
            context_limit = min(self.settings.chat_context_tokens, int(detected_context)) if detected_context else self.settings.chat_context_tokens
            reserve = self.settings.answer_max_tokens
            removed_for_budget: list[dict[str, Any]] = []
            template_applied = True
            while True:
                message_content = self._build_prompt(normalized_query, chosen)
                messages = [{"role": "user", "content": message_content}]
                self.emit(
                    job_id, "prompt_budget", "progress", phase="chat_template",
                    message="Formatting the selected context with the model chat template", selected_count=len(chosen),
                )
                try:
                    formatted_prompt = await self.llama.apply_chat_template(messages)
                except Exception:
                    formatted_prompt = message_content
                    template_applied = False
                self.emit(
                    job_id, "prompt_budget", "progress", phase="tokenization",
                    message="Counting the exact formatted prompt tokens", selected_count=len(chosen),
                )
                prompt_tokens = await self.llama.tokenize(formatted_prompt, "chat")
                if len(prompt_tokens) + reserve <= context_limit:
                    break
                if not chosen:
                    raise ValueError(f"The question and chat template require {len(prompt_tokens)} tokens, leaving insufficient room for {reserve} answer tokens in a {context_limit}-token context")
                removed = chosen.pop()
                removed_for_budget.append(removed)
                self.emit(
                    job_id, "prompt_budget", "progress", phase="trim_context",
                    message=f"Removing {removed['marker']} to fit the context window",
                    prompt_tokens=len(prompt_tokens), context_limit=context_limit, selected_count=len(chosen),
                )

            self.emit(
                job_id, "selection", "completed", top_k=request.top_k, threshold=request.similarity_threshold,
                selected=chosen, deferred_for_source_diversity=diversity_deferred,
                removed_for_context=[item["marker"] for item in removed_for_budget],
            )
            selected_by_id = {chunk["id"]: chunk for chunk in chunks}
            selected_matrix = np.stack([selected_by_id[item["chunk_id"]]["vector"] for item in chosen]) if chosen else np.empty((0, len(query_vector)), dtype=np.float32)
            coords = self._project(query_vector, selected_matrix, chosen)
            self.emit(job_id, "projection", "completed", points=coords, method="centered SVD", scope="selected context only")
            self.emit(
                job_id, "prompt_budget", "completed", context_limit=context_limit, prompt_tokens=len(prompt_tokens),
                answer_reserve=reserve, available_tokens=context_limit - len(prompt_tokens) - reserve,
                token_ids=prompt_tokens, template_applied=template_applied,
                context_source="minimum of llama.cpp metadata and CHAT_CONTEXT_TOKENS" if detected_context else "CHAT_CONTEXT_TOKENS fallback",
                removed_for_context=[item["marker"] for item in removed_for_budget],
            )
            self.emit(job_id, "prompt", "completed", prompt=formatted_prompt, message_content=message_content, context_chunks=len(chosen))
            answer = ""
            if request.generate_answer:
                self.emit(job_id, "generation", "started", endpoint="/v1/chat/completions")
                async for fragment in self.llama.chat(messages, max_tokens=reserve):
                    answer += fragment
                    self.emit(job_id, "generation", "progress", token=fragment)
                valid_markers = sorted(set(re.findall(r"\[D\d+:P\d+:C\d+\]", answer)))
                allowed = {item["marker"] for item in chosen}
                citations = [marker for marker in valid_markers if marker in allowed]
                self.emit(job_id, "generation", "completed", answer=answer, citations=citations)
            self.emit(job_id, "complete", "completed", duration_ms=round((time.perf_counter() - started) * 1000), answer=answer, result_count=len(chosen))
        except Exception as exc:
            self.emit(job_id, "pipeline", "failed", message=str(exc), type=type(exc).__name__)

    async def process_transformer(self, run_id: str, job_id: str, request: TransformerRequest) -> None:
        started = time.perf_counter()
        try:
            info = await self.llama.model_info("chat")
            self.emit(job_id, "model", "completed", model=info.model_id, context_size=info.context_size, observable=True)
            tokens = await self.llama.tokenize(request.prompt, "chat")
            self.emit(job_id, "tokenization", "completed", prompt=request.prompt, token_ids=tokens, token_count=len(tokens), observable=True)
            self.emit(job_id, "generation", "started", max_tokens=request.max_tokens, temperature=request.temperature, observable=True)
            output = ""
            async for fragment in self.llama.chat(
                [{"role": "user", "content": request.prompt}], request.max_tokens, request.temperature
            ):
                output += fragment
                self.emit(job_id, "generation", "progress", token=fragment, observable=True)
            output_tokens = await self.llama.tokenize(output, "chat") if output else []
            self.emit(job_id, "generation", "completed", output=output, output_token_ids=output_tokens, observable=True)
            self.emit(job_id, "complete", "completed", duration_ms=round((time.perf_counter() - started) * 1000), output=output)
        except Exception as exc:
            self.emit(job_id, "pipeline", "failed", message=str(exc), type=type(exc).__name__)

    @staticmethod
    def _ranked_chunk(chunk: dict[str, Any], score: float, rank: int, document_number: int) -> dict[str, Any]:
        marker = f"[D{document_number}:P{chunk['page'] or 1}:C{chunk['chunk_index'] + 1}]"
        return {
            "rank": rank, "chunk_id": chunk["id"], "document_id": chunk["document_id"],
            "filename": chunk["filename"], "page": chunk["page"], "chunk_index": chunk["chunk_index"],
            "text": chunk["text"], "token_count": chunk["token_count"], "score": round(score, 7), "marker": marker,
        }

    @staticmethod
    def _project(query: np.ndarray, vectors: np.ndarray, metadata: list[dict[str, Any]] | None = None) -> list[dict[str, float | str]]:
        all_vectors = np.vstack([query, vectors])
        centered = all_vectors - all_vectors.mean(axis=0)
        if len(all_vectors) > 1:
            u, singular, _ = np.linalg.svd(centered, full_matrices=False)
            projection = u[:, :2] * singular[:2]
            if projection.shape[1] == 1:
                projection = np.column_stack([projection[:, 0], np.zeros(len(projection))])
        else:
            projection = np.zeros((1, 2))
        points = [{"id": "query", "x": float(projection[0, 0]), "y": float(projection[0, 1])}]
        for i, row in enumerate(projection[1:]):
            item = metadata[i] if metadata and i < len(metadata) else None
            points.append({
                "id": item["marker"] if item else f"candidate-{i + 1}",
                "x": float(row[0]), "y": float(row[1]),
                **({"chunk_id": item["chunk_id"], "filename": item["filename"], "score": item["score"]} if item else {}),
            })
        return points

    @staticmethod
    def _project_embeddings(vectors: np.ndarray) -> list[dict[str, float | str]]:
        centered = vectors - vectors.mean(axis=0)
        if len(vectors) > 1:
            u, singular, _ = np.linalg.svd(centered, full_matrices=False)
            projection = u[:, :2] * singular[:2]
            if projection.shape[1] == 1:
                projection = np.column_stack([projection[:, 0], np.zeros(len(projection))])
        else:
            projection = np.zeros((1, 2))
        return [
            {"id": f"chunk-{index + 1}", "x": float(row[0]), "y": float(row[1])}
            for index, row in enumerate(projection)
        ]

    @staticmethod
    def _build_prompt(query: str, chunks: list[dict[str, Any]]) -> str:
        context = "\n\n".join(f"{item['marker']} {item['filename']}\n{item['text']}" for item in chunks)
        return (
            "Answer the question using only the supplied sources. Cite each factual claim with its exact source marker. "
            "If the sources do not contain enough information, say so clearly.\n\n"
            f"SOURCES\n{context or '(No source passed the similarity threshold.)'}\n\nQUESTION\n{query}\n\nANSWER"
        )
