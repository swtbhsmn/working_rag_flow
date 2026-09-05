SHELL := /bin/bash

PYTHON ?= python3
BACKEND_DIR := backend
FRONTEND_DIR := frontend
API_PORT ?= 8000
WEB_PORT ?= 5173
VENV := $(BACKEND_DIR)/.venv
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest
UVICORN := $(VENV)/bin/uvicorn

.DEFAULT_GOAL := help
.PHONY: help setup install backend-install frontend-install dev api web test test-backend test-frontend build docker-up docker-down docker-logs health

help: ## Show available commands
	@awk 'BEGIN {FS = ":.*## "; printf "Under the Token commands:\n\n"} /^[a-zA-Z_-]+:.*## / {printf "  make %-18s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: backend-install frontend-install ## Install backend and frontend dependencies

install: setup ## Alias for setup

backend-install: ## Create the Python environment and install API dependencies
	@test -d $(VENV) || $(PYTHON) -m venv $(VENV)
	@cd $(BACKEND_DIR) && .venv/bin/pip install -e '.[test]'

frontend-install: ## Install React dependencies
	@cd $(FRONTEND_DIR) && npm install

dev: ## Run FastAPI and React together; Ctrl-C stops both
	@test -x $(UVICORN) || { echo "Run 'make setup' first."; exit 1; }
	@test -d $(FRONTEND_DIR)/node_modules || { echo "Run 'make setup' first."; exit 1; }
	@if lsof -nP -iTCP:$(API_PORT) -sTCP:LISTEN >/dev/null 2>&1; then \
	  echo "Port $(API_PORT) is already in use. Run: make dev API_PORT=8001"; exit 1; \
	fi
	@if lsof -nP -iTCP:$(WEB_PORT) -sTCP:LISTEN >/dev/null 2>&1; then \
	  echo "Port $(WEB_PORT) is already in use. Override it with WEB_PORT=5174."; exit 1; \
	fi
	@(cd $(BACKEND_DIR) && .venv/bin/uvicorn app.main:app --reload --host 127.0.0.1 --port $(API_PORT)) & api_pid=$$!; \
	  (cd $(FRONTEND_DIR) && VITE_API_TARGET=http://127.0.0.1:$(API_PORT) npm run dev -- --host 127.0.0.1 --port $(WEB_PORT)) & web_pid=$$!; \
	  trap 'kill $$api_pid $$web_pid 2>/dev/null || true' INT TERM EXIT; \
	  wait

api: ## Run only the FastAPI development server
	@cd $(BACKEND_DIR) && .venv/bin/uvicorn app.main:app --reload --host 127.0.0.1 --port $(API_PORT)

web: ## Run only the React development server
	@cd $(FRONTEND_DIR) && VITE_API_TARGET=http://127.0.0.1:$(API_PORT) npm run dev -- --host 127.0.0.1 --port $(WEB_PORT)

test: test-backend test-frontend ## Run all automated tests

test-backend: ## Run backend tests
	@cd $(BACKEND_DIR) && .venv/bin/pytest -q

test-frontend: ## Run frontend tests
	@cd $(FRONTEND_DIR) && npm test

build: ## Build the production frontend and compile-check Python
	@$(VENV)/bin/python -m compileall -q $(BACKEND_DIR)/app
	@cd $(FRONTEND_DIR) && npm run build

docker-up: ## Build and run the application with Docker Compose
	@docker compose up --build

docker-down: ## Stop Docker Compose services
	@docker compose down

docker-logs: ## Follow Docker Compose logs
	@docker compose logs -f

health: ## Check FastAPI and both configured llama.cpp servers
	@echo "FastAPI:"; curl -fsS http://127.0.0.1:$(API_PORT)/health || true; echo
	@echo "Chat llama-server ($${LLAMA_CHAT_BASE_URL:-http://127.0.0.1:8080}):"; curl -fsS "$${LLAMA_CHAT_BASE_URL:-http://127.0.0.1:8080}/health" || true; echo
	@echo "Embedding llama-server ($${LLAMA_EMBED_BASE_URL:-http://127.0.0.1:8081}):"; curl -fsS "$${LLAMA_EMBED_BASE_URL:-http://127.0.0.1:8081}/health" || true; echo
	@echo "Token embedding llama-server ($${LLAMA_TOKEN_EMBED_BASE_URL:-http://192.168.31.92:8082}):"; curl -fsS "$${LLAMA_TOKEN_EMBED_BASE_URL:-http://192.168.31.92:8082}/health" || true; echo
