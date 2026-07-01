#!/usr/bin/env python3
"""Basic endpoint and pod checks for a Tokkio deployment."""

from __future__ import annotations

import argparse
import json
import ssl
import subprocess
import urllib.error
import urllib.request
from urllib.parse import urlsplit, urlunsplit


def _fetch_url(url: str, insecure: bool, timeout: float) -> dict[str, object]:
    context = ssl._create_unverified_context() if insecure else None
    request = urllib.request.Request(url, headers={"User-Agent": "tokkio-check/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
            return {
                "ok": 200 <= response.status < 400,
                "status": response.status,
                "url": url,
            }
    except urllib.error.HTTPError as exc:
        return {"ok": False, "status": exc.code, "detail": str(exc), "url": url}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "detail": str(exc), "url": url}


def _http_fallback_url(url: str) -> str | None:
    parsed = urlsplit(url)
    if parsed.scheme != "https":
        return None
    return urlunsplit(("http", parsed.netloc, parsed.path, parsed.query, parsed.fragment))


def _should_retry_over_http(result: dict[str, object]) -> bool:
    detail = str(result.get("detail", ""))
    return "WRONG_VERSION_NUMBER" in detail


def check_url(name: str, url: str, insecure: bool, timeout: float) -> dict[str, object]:
    result = _fetch_url(url, insecure, timeout)
    result["name"] = name
    if result.get("ok"):
        return result

    fallback_url = _http_fallback_url(url)
    if not fallback_url or not _should_retry_over_http(result):
        return result

    fallback = _fetch_url(fallback_url, insecure, timeout)
    fallback["name"] = name
    fallback["requested_url"] = url
    fallback["protocol_mismatch"] = True
    if fallback.get("ok"):
        fallback["detail"] = "HTTPS endpoint responded as plain HTTP"
    return fallback


def check_kubectl(namespace: str, timeout: float) -> dict[str, object]:
    try:
        completed = subprocess.run(  # noqa: S603
            ["kubectl", "get", "pods", "-n", namespace],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "name": "kubectl_pods",
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "detail": (completed.stdout or completed.stderr).strip(),
        }
    except Exception as exc:  # noqa: BLE001
        return {"name": "kubectl_pods", "ok": False, "detail": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Tokkio endpoint checks")
    parser.add_argument("--ui-url", help="Tokkio UI endpoint, for example https://<ip>:30111")
    parser.add_argument("--api-url", help="Tokkio API endpoint, for example http://<ip>:30888")
    parser.add_argument("--grafana-url", help="Grafana endpoint, for example http://<ip>:32300")
    parser.add_argument("--timeout", type=float, default=10.0, help="Timeout in seconds")
    parser.add_argument("--insecure", action="store_true", help="Disable TLS verification")
    parser.add_argument("--kubectl", action="store_true", help="Also run kubectl get pods")
    parser.add_argument("--namespace", default="app", help="Kubernetes namespace for --kubectl")
    args = parser.parse_args()

    checks: list[dict[str, object]] = []
    if args.ui_url:
        checks.append(check_url("ui", args.ui_url, args.insecure, args.timeout))
    if args.api_url:
        checks.append(check_url("api", args.api_url, args.insecure, args.timeout))
    if args.grafana_url:
        checks.append(check_url("grafana", args.grafana_url, args.insecure, args.timeout))
    if args.kubectl:
        checks.append(check_kubectl(args.namespace, args.timeout))

    print(json.dumps(checks, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
