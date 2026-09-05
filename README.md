# Under the Token

A local-first teaching interface that makes document ingestion, embeddings, vector search, RAG prompt construction, generation, and transformer concepts visible. React renders the learning workspaces, FastAPI records real pipeline events, SQLite stores local documents and vectors, and two existing `llama-server` processes provide embeddings and chat generation.

## Knowledge ingestion visualizer

The **Add knowledge** workspace follows an uploaded file through the complete searchable-knowledge lifecycle:

`Upload → Validate → Extract → Clean → Detect structure → Chunk with overlap → Tokenize → Enrich metadata → Embed → Store → Index`

Step mode pauses after every completed phase. Use **Next phase** to reveal the next buffered phase, or **Live** to resume the real-time event stream. Selecting any stage exposes its input, process, and output. The embedding phase includes every chunk in a two-dimensional vector projection. The storage phase exposes every stored chunk in a vertically virtualized list, and every complete embedding from `d0` through its final dimension in a horizontally virtualized strip.

The interface reports the actual implementation: vectors and metadata are stored in SQLite and searched with an exact cosine scan. The index event therefore describes lookup indexes and search readiness without claiming an approximate-nearest-neighbor index that is not present.

## Prerequisites

- Python 3.11+
- Node.js 20+
- A chat `llama-server`, normally at `http://127.0.0.1:8080`
- An embedding `llama-server`, normally at `http://127.0.0.1:8081`
- A second embedding `llama-server` using the same model/tokenizer with `--pooling none`, normally at port `8082`

For example, start the embedding server with an embedding-capable GGUF model and an appropriate pooling strategy:

```sh
llama-server -m /path/to/embedding-model.gguf --embedding --pooling cls --port 8081
```

For clickable real token-level contextual vectors, start the same embedding model separately with pooling disabled:

```sh
llama-server -m /path/to/embedding-model.gguf --embedding --pooling none --port 8082
```

Start the chat model separately:

```sh
llama-server -m /path/to/chat-model.gguf --port 8080
```

Pooling and optional query/document prefixes depend on the embedding model. Set them according to that model's documentation; both ingestion and search must use the same model configuration.

## Local development

Copy the root `.env.example` to a root `.env` and adjust the llama.cpp URLs and model aliases. The same root file is loaded by Make, FastAPI, and Docker Compose. The shortest setup is:

```sh
make setup
make dev
```

The important model settings are `LLAMA_CHAT_BASE_URL`, `LLAMA_EMBED_BASE_URL`, `LLAMA_TOKEN_EMBED_BASE_URL`, and any model-specific `EMBED_QUERY_PREFIX` / `EMBED_DOCUMENT_PREFIX`. The token server must use the same embedding model and tokenizer as the main server; the API rejects mismatched token IDs rather than displaying incorrectly aligned vectors. Set `LLAMA_TOKEN_EMBED_API_KEY` only when the second server requires authentication. Set `CHAT_CONTEXT_TOKENS` to the chat server's effective context window and `ANSWER_MAX_TOKENS` to the output space reserved during RAG prompt construction. Changing an embedding model, prefix, dimension, or ingestion pipeline version marks existing documents as requiring reindexing.

If port 8000 is already occupied, choose another API port; the Vite proxy is configured automatically:

```sh
make dev API_PORT=8001
```

The frontend port can likewise be changed with `WEB_PORT=5174`.

Other useful commands:

```sh
make help            # list every command
make test            # run backend and frontend tests
make build           # production build and Python compile check
make health          # check FastAPI and both llama.cpp servers
make docker-up       # run with Docker Compose
```

The equivalent manual setup is:

```sh
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
uvicorn app.main:app --reload
```

In a second terminal:

```sh
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. Uploaded content and SQLite data are written beneath `./data`, which is ignored by Git.

## Docker

With both llama.cpp servers already running on the host:

```sh
docker compose up --build
```

Open `http://localhost:8088`. On Linux, the included `host-gateway` mapping lets the API reach both host services.

## Verification

```sh
cd backend && pytest
cd frontend && npm test
cd frontend && npm run build
```

The transformer lesson intentionally labels its small matrices as an educational simulation. Token IDs, model metadata, streamed text, ingestion values, embeddings, cosine scores, selected context, and prompts come from the real local pipeline; llama.cpp does not expose stable internal attention activations through `llama-server`.
