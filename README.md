# Enterprise AI Copilot

AI-powered enterprise copilot leveraging Retrieval-Augmented Generation (RAG),
local LLM (Ollama), FAISS vector search, and FastAPI backend.

## Features
- Local LLM via Ollama (llama3)
- Local embeddings (nomic-embed-text)
- FAISS vector index
- Conversational memory
- Tool routing (Calculator)
- REST API via FastAPI

## Run Locally

1. Install dependencies
2. Start Ollama
3. Pull models:
   ollama pull llama3
   ollama pull nomic-embed-text
4. Run:
   uvicorn main:app --reload