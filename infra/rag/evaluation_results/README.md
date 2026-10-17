# Local RAG Evaluation Results

The fixed evaluation set is `../evaluation_cases.json` (63 cases).

| Metric | Before content terms (2026-10-08) | Current |
| --- | ---: | ---: |
| Must-RAG combined retrieval success | 88.2% (45/51) | 100% (51/51) |
| Must-RAG context injected | 98.0% (50/51) | 100% (51/51) |
| Correct evidence in top 3 | 88.2% (45/51) | 100% (51/51) |
| Must-not-RAG context injections | 0/7 | 0/7 |
| Must-not-inject context injections | 4/5 | 0/5 |
| Follow-up context injection | 3/3 | 3/3 |
| Citation-ID duplicate cases | 0 | 0 |
| Search p95 (20 repetitions, paired runs) | 26–28 ms | 38–39 ms |

Both columns use the production route keywords and the same index. The
"before" column is the committed `local_rag.py` evaluated on the 2026-10-08
case set; the 16 cases added that day (開学, 片柳研究所, AI faculty, 推薦入試,
倍率, 出身地, 学長就任, four visitor-style phrasings such as `質問です。…`, and
five `must_not_inject` topics) came from questions outside the earlier set. The latency increase comes from scoring content terms
over every record of the routed domain. The paired latency runs were taken with
the Tokkio stack stopped; `current.json` was re-recorded with the full stack
running on the same host (p95 61 ms), and the in-pod run in
`controller_smoke_20261008.json` measured p95 37 ms.

`baseline_legacy.json` was captured before the implementation change on the
older 16-case set and is retained only as historical evidence; it is not a
like-for-like comparison with the expanded suite. `current.json` uses 63 cases
and 20 search repetitions per case, including tuition, scholarship, generic
research, affiliation, faculty-major, student-support, qualified-admission,
multi-turn follow-up, and outside-the-corpus boundaries.
`controller_smoke.json` records the generated/live hashes and the production
router smoke run performed inside the Ready controller pod.
`controller_smoke_20261008.json` records the 2026-10-08 deployment: deployed
file hashes, the full case set run inside the controller pod (all passing), and
12 live answers from the local swallow-8b LLM for routed, refused, and
outside-the-corpus questions.

This suite evaluates router, retrieval, and citation metadata. Hosted-LLM answer
wording and browser/audio transport require a separate live smoke.
