from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

import httpx

from app.service_status import ServiceStatus

if TYPE_CHECKING:
    from app.settings import Settings


class MockLlmClient:
    async def validate_model(self) -> None:
        return None

    async def stream_chat(self, user_text: str) -> AsyncIterator[str]:
        for token in ("はい、", "音声と表情をそろえて応答します。", "少し待ってください。"):
            yield token

    async def healthcheck(self) -> ServiceStatus:
        return ServiceStatus(name="llm", ok=True, detail="mock LLM enabled")


class NvidiaNimChatClient:
    def __init__(self, settings: "Settings") -> None:
        self._settings = settings
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0))

    async def close(self) -> None:
        await self._client.aclose()

    async def validate_model(self) -> None:
        if self._settings.skip_llm_model_validation:
            return
        response = await self._client.get(
            f"{self._openai_base_url()}/models",
            headers=self._headers(),
        )
        response.raise_for_status()
        data = response.json().get("data", [])
        model_ids = {item.get("id") for item in data if isinstance(item, dict)}
        if self._settings.nim_llm_model not in model_ids:
            raise RuntimeError(f"LLM model not found in /v1/models: {self._settings.nim_llm_model}")

    async def healthcheck(self) -> ServiceStatus:
        if not self._settings.nim_api_key:
            return ServiceStatus(name="llm", ok=False, detail="ACE_NIM_API_KEY is not configured")
        if not self._settings.nim_llm_model:
            return ServiceStatus(name="llm", ok=False, detail="ACE_NIM_LLM_MODEL is not configured")
        try:
            await self.validate_model()
            return ServiceStatus(
                name="llm",
                ok=True,
                detail="NVIDIA NIM LLM model is available",
                meta={"base_url": self._openai_base_url(), "model": self._settings.nim_llm_model},
            )
        except Exception as exc:
            return ServiceStatus(
                name="llm",
                ok=False,
                detail=str(exc),
                meta={"base_url": self._openai_base_url(), "model": self._settings.nim_llm_model},
            )

    async def stream_chat(self, user_text: str) -> AsyncIterator[str]:
        if not self._settings.nim_api_key:
            raise RuntimeError("ACE_NIM_API_KEY is not configured")
        if not self._settings.nim_llm_model:
            raise RuntimeError("ACE_NIM_LLM_MODEL is not configured")
        payload = {
            "model": self._settings.nim_llm_model,
            "stream": True,
            "temperature": 0,
            "top_p": 0.95,
            "max_tokens": 512,
            "messages": [
                {"role": "system", "content": self._settings.system_prompt},
                {"role": "user", "content": user_text},
            ],
        }
        url = f"{self._openai_base_url()}/chat/completions"
        async with self._client.stream("POST", url, headers=self._headers(), json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                message = json.loads(data)
                for choice in message.get("choices", []):
                    delta = choice.get("delta", {})
                    content = delta.get("content")
                    if content:
                        yield content

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self._settings.nim_api_key:
            headers["Authorization"] = f"Bearer {self._settings.nim_api_key}"
        return headers

    def _openai_base_url(self) -> str:
        base_url = self._settings.nim_llm_base_url.rstrip("/")
        if base_url.endswith("/v1"):
            return base_url
        return f"{base_url}/v1"


def build_llm_client(settings: "Settings") -> MockLlmClient | NvidiaNimChatClient:
    if settings.mock_llm:
        return MockLlmClient()
    return NvidiaNimChatClient(settings)
