#!/usr/bin/env python3
"""Prepare controller-side files for Tokkio one-click deployment."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import ipaddress
import json
import os
from pathlib import Path
import subprocess
import sys


DEFAULT_WORKSPACE_DIR = Path(__file__).resolve().parent / "workspace"
DEFAULT_LLM_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_LLM_MODEL = "stockmark/stockmark-2-100b-instruct"
DEFAULT_LLM_API_KEY = ""
DEFAULT_IRODORI_TTS_PORT = "8021"
DEFAULT_IRODORI_TTS_SERVICE = "ace-irodori-tts.service"
DEFAULT_RAG_PORT = "8081"
DEFAULT_RAG_COLLECTION_NAME = "ace_kagawa"
DEFAULT_RAG_SUFFIX_PROMPT = "日本語で簡潔に答えてください。"


@dataclass(frozen=True)
class LlmSettings:
    base_url: str
    model: str
    api_key: str


@dataclass(frozen=True)
class IrodoriTtsSettings:
    enabled: bool
    base_url: str
    health_url: str
    service: str


@dataclass(frozen=True)
class RagSettings:
    enabled: bool
    server_url: str
    collection_name: str
    use_knowledge_base: bool
    max_tokens: int
    suffix_prompt: str


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
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        values[key] = os.path.expanduser(value)
    return values


def require(values: dict[str, str], key: str) -> str:
    value = values.get(key, "").strip()
    if not value:
        raise ValueError(f"{key} is required")
    return value


def validate_ip(value: str, key: str) -> None:
    try:
        ipaddress.ip_address(value)
    except ValueError as exc:
        raise ValueError(f"{key} must be a valid IPv4 or IPv6 address: {value}") from exc


def quote_shell(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def normalize_openai_base_url(value: str) -> str:
    url = value.strip().rstrip("/")
    if not url:
        return url
    if url.endswith("/v1"):
        return url
    return f"{url}/v1"


def resolve_llm_settings(values: dict[str, str]) -> LlmSettings:
    base_url = values.get("TOKKIO_LLM_BASE_URL", "").strip() or DEFAULT_LLM_BASE_URL

    return LlmSettings(
        base_url=normalize_openai_base_url(base_url),
        model=values.get("TOKKIO_LLM_MODEL", "").strip() or DEFAULT_LLM_MODEL,
        api_key=(
            values.get("TOKKIO_LLM_API_KEY", "").strip()
            or values.get("TOKKIO_NVIDIA_API_KEY", "").strip()
            or DEFAULT_LLM_API_KEY
        ),
    )


def normalize_http_base_url(value: str) -> str:
    return value.strip().rstrip("/")


def normalize_rag_server_url(value: str) -> str:
    url = normalize_http_base_url(value)
    if not url:
        return url
    if url.endswith("/v1"):
        return url
    return f"{url}/v1"


def parse_positive_int(values: dict[str, str], key: str, default: int) -> int:
    raw = values.get(key, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer: {raw}") from exc
    if value <= 0:
        raise ValueError(f"{key} must be positive: {raw}")
    return value


def resolve_irodori_tts_settings(values: dict[str, str]) -> IrodoriTtsSettings:
    enabled = is_enabled(values, "TOKKIO_IRODORI_TTS_ENABLED", default=True)
    base_url = values.get("TOKKIO_IRODORI_TTS_BASE_URL", "").strip()
    if not base_url:
        app_host = values.get("TOKKIO_APP_HOST_IPV4_ADDR", "").strip()
        base_url = f"http://{app_host}:{DEFAULT_IRODORI_TTS_PORT}" if app_host else f"http://127.0.0.1:{DEFAULT_IRODORI_TTS_PORT}"

    health_url = values.get("TOKKIO_IRODORI_TTS_HEALTH_URL", "").strip()
    if not health_url:
        health_url = f"http://127.0.0.1:{DEFAULT_IRODORI_TTS_PORT}/healthz"

    service = values.get("TOKKIO_IRODORI_TTS_SERVICE", "").strip() or DEFAULT_IRODORI_TTS_SERVICE
    return IrodoriTtsSettings(
        enabled=enabled,
        base_url=normalize_http_base_url(base_url),
        health_url=health_url,
        service=service,
    )


def resolve_rag_settings(values: dict[str, str]) -> RagSettings:
    enabled = is_enabled(values, "TOKKIO_RAG_ENABLED", default=False)
    server_url = values.get("TOKKIO_RAG_SERVER_URL", "").strip()
    if not server_url:
        app_host = values.get("TOKKIO_APP_HOST_IPV4_ADDR", "").strip()
        server_url = f"http://{app_host}:{DEFAULT_RAG_PORT}/v1" if app_host else f"http://127.0.0.1:{DEFAULT_RAG_PORT}/v1"

    collection_name = values.get("TOKKIO_RAG_COLLECTION_NAME", "").strip() or DEFAULT_RAG_COLLECTION_NAME
    return RagSettings(
        enabled=enabled,
        server_url=normalize_rag_server_url(server_url),
        collection_name=collection_name,
        use_knowledge_base=is_enabled(values, "TOKKIO_RAG_USE_KNOWLEDGE_BASE", default=True),
        max_tokens=parse_positive_int(values, "TOKKIO_RAG_MAX_TOKENS", 1000),
        suffix_prompt=values.get("TOKKIO_RAG_SUFFIX_PROMPT", "").strip() or DEFAULT_RAG_SUFFIX_PROMPT,
    )


def is_enabled(values: dict[str, str], key: str, default: bool = False) -> bool:
    raw = values.get(key)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def build_generated_env(values: dict[str, str]) -> str:
    nvidia_api_key = require(values, "TOKKIO_NVIDIA_API_KEY")
    ngc_api_key = values.get("TOKKIO_NGC_CLI_API_KEY", "").strip() or nvidia_api_key
    openai_api_key = values.get("TOKKIO_OPENAI_API_KEY", "").strip() or "__UNUSED_OPENAI_KEY_FOR_NVIDIA_LLM_MODE__"
    app_host = require(values, "TOKKIO_APP_HOST_IPV4_ADDR")
    app_user = require(values, "TOKKIO_APP_HOST_SSH_USER")
    coturn_host = values.get("TOKKIO_COTURN_HOST_IPV4_ADDR", "").strip() or app_host
    coturn_user = values.get("TOKKIO_COTURN_HOST_SSH_USER", "").strip() or app_user
    llm_settings = resolve_llm_settings(values)

    validate_ip(app_host, "TOKKIO_APP_HOST_IPV4_ADDR")
    validate_ip(coturn_host, "TOKKIO_COTURN_HOST_IPV4_ADDR")

    lines = [
        "# Generated by prepare_tokkio_workspace.py",
        f"export OPENAI_API_KEY={quote_shell(openai_api_key)}",
        f"export NGC_CLI_API_KEY={quote_shell(ngc_api_key)}",
        f"export NVIDIA_API_KEY={quote_shell(nvidia_api_key)}",
        f"export APP_HOST_IPV4_ADDR={quote_shell(app_host)}",
        f"export APP_HOST_SSH_USER={quote_shell(app_user)}",
        f"export COTURN_HOST_IPV4_ADDR={quote_shell(coturn_host)}",
        f"export COTURN_HOST_SSH_USER={quote_shell(coturn_user)}",
        f"export ELEVENLABS_API_KEY={quote_shell(values.get('TOKKIO_ELEVENLABS_API_KEY', '').strip())}",
        f"export NVIDIA_LLM_API_KEY={quote_shell(llm_settings.api_key)}",
        "",
    ]
    return "\n".join(lines)


def maybe_apply_japanese_customization(
    values: dict[str, str],
    ace_repo_dir: Path,
) -> dict[str, str | bool]:
    enabled = is_enabled(values, "TOKKIO_APPLY_JAPANESE_CUSTOMIZATION", default=True)
    result: dict[str, str | bool] = {
        "enabled": enabled,
        "applied": False,
        "reason": "",
    }
    if not enabled:
        result["reason"] = "disabled_by_env"
        return result

    customize_script = Path(__file__).with_name("customize_tokkio_japanese.py")
    if not customize_script.exists():
        result["reason"] = f"script_not_found:{customize_script}"
        return result

    if not ace_repo_dir.exists():
        result["reason"] = f"ace_repo_missing:{ace_repo_dir}"
        return result

    irodori_tts_settings = resolve_irodori_tts_settings(values)
    rag_settings = resolve_rag_settings(values)
    subprocess.run(
        [
            sys.executable,
            str(customize_script),
            "--ace-repo-dir",
            str(ace_repo_dir),
            "--llm-base-url",
            resolve_llm_settings(values).base_url,
            "--llm-model",
            resolve_llm_settings(values).model,
            "--irodori-tts-base-url",
            irodori_tts_settings.base_url,
            "--rag-enabled",
            "true" if rag_settings.enabled else "false",
            "--rag-server-url",
            rag_settings.server_url,
            "--rag-collection-name",
            rag_settings.collection_name,
            "--rag-use-knowledge-base",
            "true" if rag_settings.use_knowledge_base else "false",
            "--rag-max-tokens",
            str(rag_settings.max_tokens),
            "--rag-suffix-prompt",
            rag_settings.suffix_prompt,
        ],
        check=True,
    )
    result["applied"] = True
    result["reason"] = "patched"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Tokkio controller workspace")
    parser.add_argument("--env-file", default=".env", help="Path to infra/tokkio .env file")
    args = parser.parse_args()

    env_path = Path(args.env_file).expanduser().resolve()
    if not env_path.exists():
        raise SystemExit(f"env file not found: {env_path}")

    values = parse_env_file(env_path)
    require(values, "TOKKIO_ACE_BRANCH")
    require(values, "TOKKIO_PROFILE")

    workspace_dir = Path(
        values.get("TOKKIO_WORKSPACE_DIR", str(DEFAULT_WORKSPACE_DIR))
    ).expanduser()
    controller_dir = Path(
        values.get("TOKKIO_CONTROLLER_DIR", str(workspace_dir / "controller"))
    ).expanduser()
    generated_dir = controller_dir / "generated"
    logs_dir = workspace_dir / "logs"
    state_dir = workspace_dir / "state"

    for directory in (workspace_dir, controller_dir, generated_dir, logs_dir, state_dir):
        directory.mkdir(parents=True, exist_ok=True)

    env_file_name = values.get("TOKKIO_ENV_FILE_NAME", "my-config.env").strip() or "my-config.env"
    generated_env_path = generated_dir / env_file_name
    generated_env_path.write_text(build_generated_env(values), encoding="utf-8")

    config_file_name = values.get("TOKKIO_CONFIG_FILE_NAME", "ace-app-config.yml").strip() or "ace-app-config.yml"
    ace_repo_dir = Path(
        values.get("TOKKIO_ACE_REPO_DIR", str(workspace_dir / "NVIDIA-ACE"))
    ).expanduser()
    llm_settings = resolve_llm_settings(values)
    irodori_tts_settings = resolve_irodori_tts_settings(values)
    rag_settings = resolve_rag_settings(values)
    japanese_customization = maybe_apply_japanese_customization(values, ace_repo_dir)
    manifest = {
        "env_file": str(env_path),
        "workspace_dir": str(workspace_dir),
        "controller_dir": str(controller_dir),
        "generated_env_file": str(generated_env_path),
        "config_file": str(controller_dir / config_file_name),
        "ace_repo_dir": str(ace_repo_dir),
        "ace_branch": values["TOKKIO_ACE_BRANCH"],
        "profile": values["TOKKIO_PROFILE"],
        "llm": {
            "base_url": llm_settings.base_url,
            "model": llm_settings.model,
            "api_key_env": "NVIDIA_LLM_API_KEY",
        },
        "japanese_customization": japanese_customization,
        "irodori_tts": {
            "enabled": irodori_tts_settings.enabled,
            "base_url": irodori_tts_settings.base_url,
            "health_url": irodori_tts_settings.health_url,
            "service": irodori_tts_settings.service,
        },
        "rag": {
            "enabled": rag_settings.enabled,
            "server_url": rag_settings.server_url,
            "collection_name": rag_settings.collection_name,
            "use_knowledge_base": rag_settings.use_knowledge_base,
            "max_tokens": rag_settings.max_tokens,
        },
    }
    (generated_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )

    next_steps = "\n".join(
        [
            "Next steps:",
            f"1. Clone NVIDIA/ACE into {manifest['ace_repo_dir']}",
            "2. Run: infra/tokkio/deploy_tokkio.sh init-config --env-file infra/tokkio/.env",
            f"3. Edit: {manifest['config_file']}",
            "4. Japanese Tokkio patch is auto-applied before install/start/reapply when the ACE clone exists",
            "5. Run: infra/tokkio/deploy_tokkio.sh install --env-file infra/tokkio/.env",
            "",
        ]
    )
    (generated_dir / "NEXT_STEPS.txt").write_text(next_steps, encoding="utf-8")

    print(
        json.dumps(
            {
                "ok": True,
                "workspace_dir": str(workspace_dir),
                "controller_dir": str(controller_dir),
                "generated_env_file": str(generated_env_path),
                "config_file": str(controller_dir / config_file_name),
            },
            ensure_ascii=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
