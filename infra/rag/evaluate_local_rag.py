#!/usr/bin/env python3
"""Evaluate local RAG routing, retrieval grounding, citations, and latency."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time

from local_rag import (
    DEFAULT_DB_PATH,
    assess_local_rag_query,
    citations_for_hits,
    dumps_json,
    filter_hits_for_domain,
    resolve_conversation_query,
    search_index,
)


DEFAULT_CASES_PATH = Path(__file__).with_name("evaluation_cases.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the local Tokkio RAG path")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=1, help="Search repetitions per case for latency measurement")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--routing-mode",
        choices=("legacy-keywords", "confidence"),
        default="confidence",
        help="Use the pre-fix keyword router or the production confidence pipeline",
    )
    parser.add_argument(
        "--route-keywords",
        default=(
            "論文,文献,出典,根拠,資料,ドキュメント,引用,詳細,詳しく,経歴,学歴,職歴,略歴,"
            "役職,現職,職名,学位,所属,生年月日,年齢,プロフィール,誰ですか,専門分野,業績,"
            "研究業績,研究内容,研究,プロジェクト,発表,受賞,特許,EBC,CMC,SiC/SiC,非破壊評価,"
            "学費,入学金,授業料,奨学金,学生支援,専攻"
        ),
    )
    return parser.parse_args()


def percentile_95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * 0.95 + 0.999999) - 1))
    return ordered[index]


def evidence_in_top_k(case: dict, hits: list) -> bool:
    chunk_ids = [hit.chunk_id for hit in hits]
    expected = set(case.get("expected_chunk_ids") or [])
    chunk_match = not expected or bool(expected.intersection(chunk_ids))
    required_pages = {str(page) for page in case.get("required_page_numbers") or []}
    retrieved_pages = {hit.page_number for hit in hits if hit.page_number}
    return chunk_match and required_pages.issubset(retrieved_pages)


def evaluate(args: argparse.Namespace) -> dict:
    cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    legacy_keywords = tuple(keyword.strip().lower() for keyword in args.route_keywords.split(",") if keyword.strip())
    results = []
    latencies_ms: list[float] = []

    for case in cases:
        user_messages = case.get("user_messages") or [case["query"]]
        query = resolve_conversation_query(user_messages, legacy_keywords)
        case_latencies = []
        hits = []
        for _ in range(args.repeat):
            start = time.perf_counter()
            hits = search_index(args.db, query, top_k=args.top_k)
            case_latencies.append((time.perf_counter() - start) * 1000.0)
        latencies_ms.extend(case_latencies)
        search_ms = statistics.median(case_latencies)

        if args.routing_mode == "legacy-keywords":
            routed = any(keyword in query.lower() for keyword in legacy_keywords)
            accepted = routed and bool(hits)
            confidence = None
            reason = "legacy_keyword_match" if routed else "legacy_no_keyword_match"
            domain = case.get("domain", "")
        else:
            assessment = assess_local_rag_query(query, hits, legacy_keywords)
            routed = assessment.route_candidate
            accepted = assessment.accepted
            confidence = assessment.confidence
            reason = assessment.reason
            domain = assessment.domain

        chunk_ids = [hit.chunk_id for hit in hits]
        page_numbers = [hit.page_number for hit in hits]
        effective_years = [hit.effective_year for hit in hits]
        expected_effective_year = str(case.get("expected_effective_year") or "")
        temporal_correct = not expected_effective_year or (
            bool(hits) and all(hit.effective_year == expected_effective_year for hit in hits)
        )
        expected_domain = case.get("domain", "")
        domain_correct = not expected_domain or domain == expected_domain
        domain_hits = filter_hits_for_domain(hits, domain) if domain else []
        domain_precision = len(domain_hits) / len(hits) if hits else 0.0
        citation_ids = [citation["document_id"] for citation in citations_for_hits(domain_hits)]
        citations_unique = len(citation_ids) == len(set(citation_ids))
        results.append(
            {
                "id": case["id"],
                "query": query,
                "user_messages": user_messages,
                "expectation": case["expectation"],
                "domain": domain,
                "expected_domain": expected_domain,
                "domain_correct": domain_correct,
                "route_candidate": routed,
                "context_injected": accepted,
                "confidence": confidence,
                "reason": reason,
                "search_ms": round(search_ms, 3),
                "top_chunk_ids": chunk_ids,
                "top_page_numbers": page_numbers,
                "top_effective_years": effective_years,
                "temporal_correct": temporal_correct,
                "top_scores": [round(hit.score, 3) for hit in hits],
                "top_k_domain_precision": round(domain_precision, 4),
                "citation_ids_unique": citations_unique,
                "grounded_in_top_k": evidence_in_top_k(case, hits) if case["expectation"] == "must_rag" else None,
            }
        )

    must_rag = [result for result in results if result["expectation"] == "must_rag"]
    must_not = [result for result in results if result["expectation"] == "must_not_rag"]
    followups = [result for result in results if len(result["user_messages"]) > 1]
    successful_must_rag = [
        result
        for result in must_rag
        if result["context_injected"]
        and result["domain_correct"]
        and result["grounded_in_top_k"]
        and result["temporal_correct"]
    ]
    summary = {
        "must_rag_count": len(must_rag),
        "must_not_rag_count": len(must_not),
        "must_rag_routing_success_rate": sum(result["context_injected"] for result in must_rag) / len(must_rag),
        "must_rag_top_k_grounding_rate": sum(result["grounded_in_top_k"] for result in must_rag) / len(must_rag),
        "must_rag_domain_accuracy": sum(result["domain_correct"] for result in must_rag) / len(must_rag),
        "must_rag_temporal_accuracy": sum(result["temporal_correct"] for result in must_rag) / len(must_rag),
        "must_rag_retrieval_success_rate": len(successful_must_rag) / len(must_rag),
        "mean_top_k_domain_precision": round(
            statistics.mean(result["top_k_domain_precision"] for result in must_rag),
            4,
        ),
        "must_not_rag_context_injections": sum(result["context_injected"] for result in must_not),
        "citation_id_duplicate_cases": sum(not result["citation_ids_unique"] for result in results),
        "followup_count": len(followups),
        "followup_context_injection_successes": sum(result["context_injected"] for result in followups),
        "search_latency_ms": {
            "min": round(min(latencies_ms), 3),
            "median": round(statistics.median(latencies_ms), 3),
            "p95": round(percentile_95(latencies_ms), 3),
            "max": round(max(latencies_ms), 3),
        },
    }
    return {
        "schema_version": 2,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "routing_mode": args.routing_mode,
        "db_path": str(args.db.resolve()),
        "top_k": args.top_k,
        "repeat": args.repeat,
        "coverage": "router_retrieval_citation_metadata_only",
        "coverage_limitations": [
            "Final hosted-LLM answer wording and browser/audio transport are not evaluated by this script.",
            "Use a live controller smoke test after syncing generated artifacts.",
        ],
        "summary": summary,
        "cases": results,
    }


def main() -> int:
    args = parse_args()
    payload = evaluate(args)
    rendered = dumps_json(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
