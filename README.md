# Taylor Swift RAG — Lyrics & Theme Q&A System

A Retrieval-Augmented Generation (RAG) system for querying Taylor Swift's discography. Ask questions about lyrics, themes, emotions, or get song recommendations — the system retrieves relevant context and generates answers powered by an LLM.

## Architecture

```
User Question
    │
    ▼
┌─────────────────┐
│ 1. Intent Router│ ← lyrics vs. theme classification
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 2. Hybrid Search│ ← BM25 + Semantic + RRF Fusion
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 3. LLM Filter   │ ← remove irrelevant documents
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 4. Reranker     │ ← Cross-Encoder / LLM re-scoring
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 5. Context      │ ← compress redundant information
│    Compression   │
└────────┬────────┘
         │
         ▼
    Generate Answer
```

## Features

- **Intent-aware routing** — automatically detects whether the query is about lyrics or themes/emotions
- **Hybrid retrieval** — BM25 keyword search + dense semantic search fused via Reciprocal Rank Fusion (RRF)
- **Dual FAISS indexes** — separate vector stores for lyrics chunks and theme/emotion metadata
- **Query rewriting** — synonym expansion and multi-perspective query generation (optional)
- **Reranking** — LLM-based or Cross-Encoder secondary scoring
- **Context compression** — LLM-powered reduction of redundant retrieved content
- **FastAPI serving** — REST API with SSE streaming and P50/P90/P99 latency monitoring
- **Evaluation suite** — MRR, Hit Rate@K, NDCG@K, Recall@K, Precision@K with per-category breakdown

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Embeddings | `BAAI/bge-small-en-v1.5` (HuggingFace) |
| LLM | DeepSeek (via OpenAI-compatible API) |
| Vector Store | FAISS |
| Framework | LangChain, LlamaIndex |
| API | FastAPI + Uvicorn |
| Evaluation | Custom metrics suite |

## Evaluation Results

Tested on 20 queries (10 lyrics retrieval, 10 emotion analysis):

| Metric | Score |
|--------|-------|
| Song Hit Rate | **100%** (20/20) |
| Hit Rate@5 | **100%** |
| Hit Rate@3 | 95% |
| Hit Rate@1 | 60% |
| MRR | 0.777 |
| NDCG@5 | 0.819 |
| Avg Latency | 1.48s (retrieval: 0.02s, generation: 1.46s) |

## Project Structure

```
├── code/
│   ├── rag/
│   │   ├── components/
│   │   │   ├── data_load.py          # FAISS index loader (singleton)
│   │   │   └── generate_answer.py    # LLM answer generation
│   │   ├── retrieval/
│   │   │   ├── retrieval_search.py   # BM25 + semantic hybrid search + RRF
│   │   │   ├── pipeline.py           # Full retrieval pipeline orchestration
│   │   │   ├── query_rewrite.py      # Query expansion & synonym rewriting
│   │   │   ├── reranker.py           # LLM / Cross-Encoder reranking
│   │   │   └── context_compressor.py # Context compression & LLM filtering
│   │   ├── config.py                 # Unified configuration
│   │   └── logger.py                 # Logging setup
│   ├── tests/                        # Unit tests
│   ├── api.py                        # FastAPI server
│   ├── main.py                       # CLI interactive mode
│   ├── evaluate.py                   # Batch evaluation runner
│   └── test_latency.py               # Performance benchmarking
├── data/
│   ├── Taylor_Swift_Genius/          # Lyrics corpus (Genius)
│   └── test_dataset.json             # Evaluation test set
├── index/                            # FAISS indexes (generated, in .gitignore)
└── result/                           # Evaluation outputs
```

## Setup

```bash
# 1. Clone and create virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set API key
cp .env.example .env
# Edit .env and add your DEEPSEEK_API_KEY

# 4. Build FAISS indexes (from lyrics corpus)
python code/scripts/generate_track_index.py
python code/scripts/generate_lyrics_index.py
```

## Usage

### CLI (interactive)

```bash
python code/main.py
```

### API Server

```bash
python code/api.py
# → http://localhost:8000
# → Swagger docs at http://localhost:8000/docs
```

Endpoints:
- `POST /query` — standard RAG query
- `POST /query/stream` — SSE streaming response
- `GET /health` — health check
- `GET /stats` — P50/P90/P99 latency stats

### Evaluation

```bash
python code/evaluate.py --test-data data/test_dataset.json --output data/test_results.json
```

### Run Tests

```bash
python -m pytest code/tests/ -v
```
