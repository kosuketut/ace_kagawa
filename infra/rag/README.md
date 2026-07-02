# RAG for Tokkio

This directory contains the local SQLite RAG tools used by default and optional wrappers for running NVIDIA RAG Blueprint beside the `ace_kagawa` Tokkio stack.

## Storage Layout

Heavy runtime state stays outside the repo:

```bash
/data/ACE/rag/blueprint     # NVIDIA-AI-Blueprints/rag checkout
/data/ACE/data/rag/corpus   # local source documents via ./data/rag/corpus
/data/ACE/data/rag/local    # local SQLite RAG index via ./data/rag/local
```

The repo-local `data` entry is a symlink to `/data/ACE/data`, so use this path from the checkout:

```bash
mkdir -p data/rag/corpus
```

## Local SQLite RAG

For the current Markdown/text corpus size, local RAG is the default fast path. It builds a SQLite FTS5/BM25 index under `data/rag/local` and Tokkio injects only the top local chunks into the hosted Stockmark NIM prompt.

Build the index:

```bash
python3 infra/rag/build_local_index.py \
  --corpus data/rag/corpus \
  --db data/rag/local/local_rag.sqlite
```

Query the index directly:

```bash
python3 infra/rag/query_local_index.py \
  --db data/rag/local/local_rag.sqlite \
  "香川先生の専門分野は何ですか"
```

The local index supports Markdown and text files directly. PDF files are indexed when `pdftotext` is installed; otherwise they are skipped, so convert important PDFs to Markdown/text if exact local retrieval is required.

Tokkio uses local RAG when:

```dotenv
TOKKIO_RAG_ENABLED=true
TOKKIO_RAG_MODE=auto
TOKKIO_RAG_PROVIDER=local
TOKKIO_LOCAL_RAG_DB=data/rag/local/local_rag.sqlite
TOKKIO_LOCAL_RAG_RUNTIME_DB_PATH=/code/configs/local_rag.sqlite
TOKKIO_LOCAL_RAG_TOP_K=3
TOKKIO_LOCAL_RAG_MAX_CONTEXT_CHARS=1800
```

After rebuilding the index, regenerate and sync the controller so the DB is copied into the controller config bundle:

```bash
python3 infra/tokkio/prepare_tokkio_workspace.py --env-file infra/tokkio/.env
infra/tokkio/manage_tokkio.sh sync-controller --env-file infra/tokkio/.env
```

## NVIDIA RAG Blueprint

Use NVIDIA RAG Blueprint when you need its ingestion pipeline, citations, or multimodal/table handling. Set `TOKKIO_RAG_PROVIDER=nvidia` before regenerating Tokkio.

Set `NGC_API_KEY` or `NVIDIA_API_KEY` in the shell, then run:

```bash
infra/rag/manage_rag.sh init
infra/rag/manage_rag.sh start
infra/rag/manage_rag.sh health
```

The wrapper uses NVIDIA-hosted NIMs, Elasticsearch, and the Blueprint compose files under `/data/ACE/rag/blueprint/deploy/compose`.

## Ingest Documents Into NVIDIA RAG

Put PDF, Markdown, or text files under `data/rag/corpus`, then run:

```bash
infra/rag/ingest_local_corpus.sh --collection ace_kagawa
```

The script calls the Blueprint ingestor API on `http://127.0.0.1:8082/v1/documents` and polls `/v1/status`.

If you deleted files from `data/rag/corpus`, reset the collection before re-ingesting so stale vectors are removed:

```bash
infra/rag/ingest_local_corpus.sh --collection ace_kagawa --reset-collection
```

This deletes the `ace_kagawa` collection from the ingestor server, recreates it, and uploads the current files under `data/rag/corpus`.

## Tokkio Integration

`infra/tokkio/.env` enables local RAG for this variant by default. Use `TOKKIO_RAG_PROVIDER=nvidia` to switch the routed turns to the NVIDIA RAG server:

```dotenv
TOKKIO_RAG_ENABLED=true
TOKKIO_RAG_MODE=auto
TOKKIO_RAG_PROVIDER=local
TOKKIO_RAG_SERVER_URL=http://10.209.1.12:8081/v1
TOKKIO_RAG_COLLECTION_NAME=ace_kagawa
TOKKIO_RAG_MAX_TOKENS=128
TOKKIO_RAG_VDB_TOP_K=12
TOKKIO_RAG_RERANKER_TOP_K=5
TOKKIO_RAG_MULTIMODAL_RERANKER_TOP_K=10
TOKKIO_RAG_ENABLE_RERANKER=true
TOKKIO_LOCAL_RAG_DB=data/rag/local/local_rag.sqlite
TOKKIO_LOCAL_RAG_RUNTIME_DB_PATH=/code/configs/local_rag.sqlite
```

Regenerate and sync the controller after local index rebuilds or NVIDIA RAG config changes:

```bash
python3 infra/tokkio/prepare_tokkio_workspace.py --env-file infra/tokkio/.env
infra/tokkio/manage_tokkio.sh sync-controller --env-file infra/tokkio/.env
```
