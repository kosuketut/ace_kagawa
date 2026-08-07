# Local RAG Evaluation Results

The fixed evaluation set is `../evaluation_cases.json`.

| Metric | Current confidence router |
| --- | ---: |
| Must-RAG routing/context success | 100% (18/18) |
| Correct domain | 100% (18/18) |
| Correct evidence in top 3 | 100% (18/18) |
| Combined retrieval success | 100% (18/18) |
| Must-not-RAG context injections | 0/7 |
| Follow-up context injection | 1/1 |
| Citation-ID duplicate cases | 0 |
| Search p95 | 33.251 ms |

`baseline_legacy.json` was captured before the implementation change on the
older 16-case set and is retained only as historical evidence; it is not a
like-for-like comparison with the expanded suite. `current.json` uses 25 cases
and 20 search repetitions per case, including tuition, scholarship, generic
research, affiliation, faculty-major, student-support, qualified-admission, and
multi-turn follow-up boundaries.
`controller_smoke.json` records the generated/live hashes and the production
router smoke run performed inside the Ready controller pod.

This suite evaluates router, retrieval, and citation metadata. Hosted-LLM answer
wording and browser/audio transport require a separate live smoke.
