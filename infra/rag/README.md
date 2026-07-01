# Host-Side NVIDIA RAG for Tokkio

This directory contains lightweight wrappers for running NVIDIA RAG Blueprint beside the `ace_kagawa` Tokkio stack.

## Storage Layout

Heavy runtime state stays outside the repo:

```bash
/data/ACE/rag/blueprint     # NVIDIA-AI-Blueprints/rag checkout
/data/ACE/data/rag/corpus   # local source documents via ./data/rag/corpus
```

The repo-local `data` entry is a symlink to `/data/ACE/data`, so use this path from the checkout:

```bash
mkdir -p data/rag/corpus
```

## Start RAG

Set `NGC_API_KEY` or `NVIDIA_API_KEY` in the shell, then run:

```bash
infra/rag/manage_rag.sh init
infra/rag/manage_rag.sh start
infra/rag/manage_rag.sh health
```

The wrapper uses NVIDIA-hosted NIMs, Elasticsearch, and the Blueprint compose files under `/data/ACE/rag/blueprint/deploy/compose`.

## Ingest Local Documents

Put PDF, Markdown, or text files under `data/rag/corpus`, then run:

```bash
infra/rag/ingest_local_corpus.sh --collection ace_kagawa
```

The script calls the Blueprint ingestor API on `http://127.0.0.1:8082/v1/documents` and polls `/v1/status`.

## Tokkio Integration

`infra/tokkio/.env` now enables RAG for this variant:

```dotenv
TOKKIO_RAG_ENABLED=true
TOKKIO_RAG_SERVER_URL=http://10.209.1.12:8081/v1
TOKKIO_RAG_COLLECTION_NAME=ace_kagawa
```

Regenerate and sync the controller after RAG is healthy:

```bash
python3 infra/tokkio/prepare_tokkio_workspace.py --env-file infra/tokkio/.env
infra/tokkio/manage_tokkio.sh sync-controller --env-file infra/tokkio/.env
```
