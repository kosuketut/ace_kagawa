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

### Import The TEU Faculty And Admissions Dataset

Keep `data/teu_rag_dataset.zip` under the existing `data` symlink and import it
with the repository adapter:

```bash
python3 infra/rag/import_teu_dataset.py
```

The importer verifies the ZIP CRC and every entry in `checksums.sha256`, then
uses only the recommended `chunks_plaintext.jsonl`. It creates
`data/rag/corpus/02_teu_faculty_admissions_dataset.md` with the official source
URL, source format, PDF page, effective year, temporal status, and chunk ID
preserved per section. Previous faculty/admissions corpus files are moved to a
timestamped directory under `data/rag/backups/teu_import`; the original ZIP is
not modified. A verified input copy and import manifest are stored under
`data/rag/sources/teu/current`.

Do not unpack the whole ZIP into `data/rag/corpus`: doing so indexes both the
structured and plaintext representations, duplicates evidence, and loses the
metadata-aware year selection. The importer also excludes the known
`/entrance/006272.html` record whose dataset metadata labels it as 2027 even
though the page describes 2026 entrants. The official 2027 admissions PDF,
page 18, remains the 2027 tuition source.

### Import The SEIRAN Supercomputer Dataset

Keep `data/tut_seiran_rag_dataset.zip` under the existing `data` symlink and
run the repository adapter:

```bash
python3 infra/rag/import_seiran_dataset.py
```

The importer validates the ZIP layout and CRC, JSONL schemas, unique IDs,
official-source allowlist, manifest-to-row source integrity, fact statuses,
and source URL mappings. It indexes only the 40 curated knowledge chunks as
`data/rag/corpus/04_tut_seiran.md`; the 18 FAQ rows are retained under
`data/rag/sources/seiran/current` but are not indexed, preventing duplicate
evidence. Because the supplied ZIP has no publisher-provided per-file SHA-256
manifest, the import manifest pins the archive SHA-256 and every extracted
member hash. Ranking answers retain their edition year, derived values remain
marked as derived, and unpublished application, fee, scheduler, and queue
details remain explicitly unconfirmed.

After import, use the same atomic rebuild and controller sync path:

```bash
infra/rag/rebuild_and_sync_local_rag.sh --env-file infra/tokkio/.env
```

After import, rebuild, verify, regenerate, and sync the controller in one step:

```bash
infra/rag/rebuild_and_sync_local_rag.sh --env-file infra/tokkio/.env
```

Use `--no-sync` when the cluster is unavailable.

Build the index:

```bash
python3 infra/rag/build_local_index.py \
  --corpus data/rag/corpus \
  --db data/rag/local/local_rag.sqlite
```

The build is atomic: it writes and validates a temporary SQLite database before
replacing the live DB. It also writes `local_rag.manifest.json` with the corpus
fingerprint, build time, chunk settings, per-file path/size/mtime/SHA-256, and
the final index SHA-256. Verify freshness explicitly with:

```bash
python3 infra/rag/verify_local_index.py \
  --corpus data/rag/corpus \
  --db data/rag/local/local_rag.sqlite
```

Query the index directly:

```bash
python3 infra/rag/query_local_index.py \
  --db data/rag/local/local_rag.sqlite \
  "香川先生の専門分野は何ですか"
```

The local index supports Markdown and text files directly. PDF files are indexed when `pdftotext` is installed; otherwise they are skipped, so convert important PDFs to Markdown/text if exact local retrieval is required. TEU dataset citations retain `source_format`, `page_number`, `effective_year`, and `temporal_status`; explicit year queries prefer the matching year, while current records are preferred for queries that do not specify a year.

Tokkio uses local RAG when:

```dotenv
TOKKIO_RAG_ENABLED=true
TOKKIO_RAG_MODE=auto
TOKKIO_RAG_PROVIDER=local
TOKKIO_LOCAL_RAG_DB=data/rag/local/local_rag.sqlite
TOKKIO_LOCAL_RAG_RUNTIME_DB_PATH=/code/configs/local_rag.sqlite
TOKKIO_LOCAL_RAG_TOP_K=3
TOKKIO_LOCAL_RAG_MAX_CONTEXT_CHARS=2800
```

After rebuilding the index, regenerate and sync the controller so the DB is copied into the controller config bundle:

```bash
python3 infra/tokkio/prepare_tokkio_workspace.py --env-file infra/tokkio/.env
infra/tokkio/manage_tokkio.sh sync-controller --env-file infra/tokkio/.env
```

For corpus updates, use the combined safe path. It stops before controller sync
if the build, fingerprint check, generated-copy hash, or controller-copy hash
does not match:

```bash
infra/rag/rebuild_and_sync_local_rag.sh --env-file infra/tokkio/.env
```

Use `--no-sync` when preparing artifacts without a reachable cluster.

## Routing, Confidence, And Citations

Auto mode uses a dependency-free in-process classifier plus the configured
`TOKKIO_RAG_ROUTE_KEYWORDS`. It first excludes greetings, assistant-name checks,
weather, and general chat; then classifies supported corpus domains such as
Kagawa profile/research, university overview and student support, faculties,
access, admissions including tuition/scholarships, open campus, pamphlets, and
the SEIRAN supercomputer.
Short two-character Japanese terms such as `学費` use an exact lexical
supplement because SQLite's trigram tokenizer cannot retrieve them reliably.
Short contextual follow-ups such as `もう少し詳しく` inherit the most recent
supported user topic.

The domain decides which records are eligible; the query's content terms decide
which eligible record answers. Content terms are what remains after removing
scope words (`香川先生`, `東京工科大学`), routing vocabulary (`研究`, `入試`,
`学部`), and question words: `量子コンピュータの研究について` keeps
`量子コンピュータ`, `推薦入試はありますか` keeps `推薦入試`, and
`香川先生の研究について` keeps nothing. A term matches a record when it splits
into routing words and two-or-more-character pieces found in that record, so
`推薦入試` matches `学校推薦型選抜`. ASCII terms match whole words only (`AI`
never matches `detail`), and a few synonyms are fixed in code (`AI`/`人工知能`,
`開学`/`創立`/`設立`, `出身地`/`生まれ`). Records that name the term in their
title or opening text rank above records that only mention it.

A routed query receives context only when the retrieved chunks pass domain,
score, query-anchor, content-term, source-priority, and metadata checks. The
content-term check fails only when none of the query's content terms appears in
any included record, because visitors add framing that no record mentions.
Speaker, family, time, and money paraphrases (`私`, `娘`, `今日`, `質問`,
`値段`) and school grades (`高校2年生`) are not content terms. When a check
fails:

- Kagawa questions reached only through generic words (`研究`, `受賞`) with no
  person anchor (`香川`, `学長`, `あなた`, `先生`) go to the normal LLM path;
  the router decides this before entering the grounded path. Instead of
  retrieved context the LLM receives `OUTSIDE_CORPUS_PROMPT_NOTE`: the topic is
  not in the Kagawa/TEU material, answer from general knowledge without linking
  it to Kagawa's research, and do not guess TEU-specific numbers. Without the
  note, swallow-8b attributed quantum-computer cooling research to Kagawa.
  Example: `量子コンピュータの研究について教えてください`.
- Every other routed query (unknown people, unmatched admission names, empty
  retrieval, missing content terms, low confidence) gets the fixed
  `GROUNDED_ANSWER_UNAVAILABLE_REPLY` instead of an ungrounded LLM answer.
  Example: `香川先生の趣味は何ですか`.

The `university` domain (configured keyword `東京工科大学` without a narrower
intent) covers every institutional record, including admissions, faculty,
campus, SEIRAN, and university-news records, but not Kagawa's personal profile.

Open-campus records whose listed dates are all past are tagged
`event_status=ended` in the header and `【終了済み】このオープンキャンパスは終了済みです。`
at the start of the body, with a policy line giving the evaluation date.
swallow-8b ignored the header tag alone and announced the ended August dates
as upcoming; the inline marker fixed that in live checks. A schedule question
(`次のオープンキャンパスはいつですか`) with only ended events in the evidence
gets a fixed reply naming the latest ended event and stating that the next date
is not yet confirmed.

Only domain-matching chunks actually included in the prompt emit structured
citations through Tokkio's citation frame. Each citation has a unique
`document_id` and includes `source_title`, `path`, `source_url`, `chunk_id`,
publisher, publication or corpus-access date, source type/priority, record type,
source format, PDF page, effective year, temporal status, and score. The spoken prompt explicitly keeps URLs, paths, and chunk IDs out of
speech while retaining them in UI/log metadata. Corpus text is treated as data,
not as executable prompt instructions.

For current admissions and faculty evidence, prompt formatting removes corpus
curation notes such as generic instructions to re-check an official site or
admissions guide. The assistant answers directly when current evidence covers
the question. A brief temporal caveat is retained only for historical,
year-unverified, planned, or recruitment-closed evidence.

Run the fixed router/retrieval evaluation with repeated latency samples:

```bash
python3 infra/rag/evaluate_local_rag.py \
  --routing-mode confidence \
  --repeat 20 \
  --output infra/rag/evaluation_results/current.json
```

The evaluation uses the production route keywords
(`DEFAULT_RAG_ROUTE_KEYWORDS` in `infra/tokkio/customize_tokkio_japanese.py`)
and searches the routed domain, as the controller does. It covers domain
accuracy, context injection, expected top-k evidence, domain precision,
citation-ID uniqueness, contextual follow-ups, `must_not_inject` cases (routed
questions whose topic the corpus lacks), and latency. It does not claim to
evaluate hosted-LLM answer wording or browser/audio transport; run a live
controller smoke after syncing for those layers.

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
