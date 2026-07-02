#!/usr/bin/env python3
"""Check an OpenAI-compatible LLM endpoint for Tokkio realtime use."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Iterator
from urllib.parse import urljoin


DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "stockmark/stockmark-2-100b-instruct"


def normalize_openai_base_url(value: str) -> str:
    url = value.strip().rstrip("/")
    if not url:
        raise ValueError("base URL is empty")
    if url.endswith("/v1"):
        return url
    return f"{url}/v1"


def service_root_url(base_url: str) -> str:
    normalized = normalize_openai_base_url(base_url)
    return normalized[: -len("/v1")]


def build_headers(api_key: str = "") -> dict[str, str]:
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "ace-llm-endpoint-check/1.0",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def parse_chat_sse_deltas(lines: Iterable[bytes | str]) -> Iterator[str]:
    for raw_line in lines:
        if isinstance(raw_line, bytes):
            line = raw_line.decode("utf-8", errors="replace")
        else:
            line = raw_line
        line = line.strip()
        if not line or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        message = json.loads(data)
        for choice in message.get("choices", []):
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if content:
                yield str(content)


def summarize_token_timings(*, start_time: float, token_times: list[float]) -> dict[str, float | int | None]:
    if not token_times:
        return {"token_count": 0, "ttft_ms": None, "avg_itl_ms": None, "p95_itl_ms": None}

    intervals_ms = [
        (right - left) * 1000.0
        for left, right in zip(token_times, token_times[1:])
    ]
    p95_itl_ms = None
    if len(intervals_ms) >= 2:
        p95_itl_ms = statistics.quantiles(intervals_ms, n=20, method="inclusive")[18]
    elif intervals_ms:
        p95_itl_ms = intervals_ms[0]

    return {
        "token_count": len(token_times),
        "ttft_ms": (token_times[0] - start_time) * 1000.0,
        "avg_itl_ms": statistics.fmean(intervals_ms) if intervals_ms else None,
        "p95_itl_ms": p95_itl_ms,
    }


def get_json(url: str, *, api_key: str, timeout: float) -> tuple[int, object]:
    request = urllib.request.Request(url, headers=build_headers(api_key))
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
        return response.status, json.loads(body) if body else {}


def post_streaming_chat(
    base_url: str,
    *,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
) -> tuple[list[str], dict[str, float | int | None]]:
    payload = {
        "model": model,
        "stream": True,
        "temperature": 0,
        "top_p": 0.95,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "system",
                "content": (
                    "あなたは日本語で応答する対話型バーチャルアシスタントです。"
                    "あなたの名前は香川です。"
                    "標準語で自然かつ簡潔に、40から120文字、1から3文で答えてください。"
                    "マークダウン、絵文字、内部思考は出さないでください。"
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }
    request = urllib.request.Request(
        urljoin(f"{normalize_openai_base_url(base_url)}/", "chat/completions"),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=build_headers(api_key),
        method="POST",
    )
    start = time.monotonic()
    deltas: list[str] = []
    token_times: list[float] = []
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for delta in parse_chat_sse_deltas(iter(response.readline, b"")):
            deltas.append(delta)
            token_times.append(time.monotonic())
    return deltas, summarize_token_timings(start_time=start, token_times=token_times)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check an OpenAI-compatible LLM endpoint")
    parser.add_argument("--base-url", default=os.environ.get("TOKKIO_LLM_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", default=os.environ.get("TOKKIO_LLM_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--api-key",
        default=(
            os.environ.get("TOKKIO_LLM_API_KEY")
            or os.environ.get("NVIDIA_LLM_API_KEY")
            or os.environ.get("TOKKIO_NVIDIA_API_KEY")
            or os.environ.get("NVIDIA_API_KEY")
            or ""
        ),
    )
    parser.add_argument("--prompt", default="おすすめの昼食を一文で教えてください。")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    base_url = normalize_openai_base_url(args.base_url)
    result: dict[str, object] = {
        "base_url": base_url,
        "model": args.model,
    }

    try:
        status, health = get_json(f"{service_root_url(base_url)}/health", api_key=args.api_key, timeout=args.timeout)
        result["health"] = {"ok": 200 <= status < 300, "status": status, "body": health}
    except urllib.error.HTTPError as exc:
        result["health"] = {"ok": False, "status": exc.code, "detail": str(exc)}
    except Exception as exc:  # noqa: BLE001
        result["health"] = {"ok": False, "detail": str(exc)}

    try:
        status, models = get_json(urljoin(f"{base_url}/", "models"), api_key=args.api_key, timeout=args.timeout)
        model_ids = [
            item.get("id")
            for item in models.get("data", [])
            if isinstance(item, dict)
        ] if isinstance(models, dict) else []
        result["models"] = {"ok": args.model in model_ids, "status": status, "ids": model_ids}
    except urllib.error.HTTPError as exc:
        result["models"] = {"ok": False, "status": exc.code, "detail": str(exc)}
    except Exception as exc:  # noqa: BLE001
        result["models"] = {"ok": False, "detail": str(exc)}

    try:
        deltas, timings = post_streaming_chat(
            base_url,
            api_key=args.api_key,
            model=args.model,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
        result["stream"] = {
            "ok": bool(deltas),
            "text": "".join(deltas),
            "timings": timings,
        }
    except urllib.error.HTTPError as exc:
        result["stream"] = {"ok": False, "status": exc.code, "detail": str(exc)}
    except Exception as exc:  # noqa: BLE001
        result["stream"] = {"ok": False, "detail": str(exc)}

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if all(isinstance(result.get(key), dict) and result[key].get("ok") for key in ("health", "models", "stream")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
