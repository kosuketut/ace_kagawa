#!/usr/bin/env python3
"""Preflight check for Tokkio Helm chart repository access."""

from __future__ import annotations

import argparse
import base64
import json
import ssl
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlencode


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def make_basic_auth(username: str, password: str) -> str:
    token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    return f"Basic {token}"


def check_repo(url: str, api_key: str, insecure: bool, timeout: float) -> dict[str, object]:
    context = ssl._create_unverified_context() if insecure else None
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": make_basic_auth("$oauthtoken", api_key),
            "User-Agent": "tokkio-ngc-check/1.0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
            return {"ok": 200 <= response.status < 300, "status": response.status, "url": url}
    except urllib.error.HTTPError as exc:
        return {"ok": False, "status": exc.code, "detail": str(exc), "url": url}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "detail": str(exc), "url": url}


def check_nvcr_scope(scope: str, api_key: str, insecure: bool, timeout: float) -> dict[str, object]:
    context = ssl._create_unverified_context() if insecure else None
    query = urlencode({"scope": scope, "service": "nvcr.io"})
    url = f"https://nvcr.io/proxy_auth?{query}"
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": make_basic_auth("$oauthtoken", api_key),
            "User-Agent": "tokkio-ngc-check/1.0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
            # NVCR returns an auth token here; never echo it back in diagnostics.
            response.read()
            return {"ok": 200 <= response.status < 300, "status": response.status, "url": url}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace").strip()
        result = {"ok": False, "status": exc.code, "detail": str(exc), "url": url}
        if body:
            result["body"] = body[:400]
        return result
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "detail": str(exc), "url": url}


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Tokkio NGC Helm repo access")
    parser.add_argument("--env-file", required=True, help="Path to generated my-config.env")
    parser.add_argument(
        "--repo-url",
        default="https://helm.ngc.nvidia.com/nvidia/ace/index.yaml",
        help="Tokkio Helm repo index URL",
    )
    parser.add_argument(
        "--ace-image-scope",
        default="repository:nvidia/ace/tokkio-reference-ace-controller:pull",
        help="NVCR scope used to confirm standard ACE image access",
    )
    parser.add_argument(
        "--a2f-image-scope",
        default="repository:nim/nvidia/audio2face-3d:pull",
        help="NVCR scope used to confirm Audio2Face-3D access",
    )
    parser.add_argument("--timeout", type=float, default=10.0, help="Timeout in seconds")
    parser.add_argument("--insecure", action="store_true", help="Disable TLS verification")
    args = parser.parse_args()

    values = parse_env_file(Path(args.env_file).expanduser().resolve())
    checks: list[dict[str, object]] = []

    for name in ("NGC_CLI_API_KEY", "NVIDIA_API_KEY"):
        api_key = values.get(name, "")
        if not api_key:
            checks.append({"name": name.lower(), "ok": False, "detail": "missing"})
            continue
        repo_result = check_repo(args.repo_url, api_key, args.insecure, args.timeout)
        repo_result["name"] = f"{name.lower()}_helm_repo"
        repo_result["key_length"] = len(api_key)
        checks.append(repo_result)

        ace_result = check_nvcr_scope(args.ace_image_scope, api_key, args.insecure, args.timeout)
        ace_result["name"] = f"{name.lower()}_ace_image"
        ace_result["scope"] = args.ace_image_scope
        ace_result["key_length"] = len(api_key)
        checks.append(ace_result)

        a2f_result = check_nvcr_scope(args.a2f_image_scope, api_key, args.insecure, args.timeout)
        a2f_result["name"] = f"{name.lower()}_a2f_image"
        a2f_result["scope"] = args.a2f_image_scope
        a2f_result["key_length"] = len(api_key)
        checks.append(a2f_result)

    print(json.dumps(checks, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
