from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Protocol

from app.service_status import ServiceStatus


@dataclass(slots=True)
class AsrEvent:
    kind: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class StreamingAsrSession(Protocol):
    async def push_audio(self, chunk: bytes) -> None:
        ...

    async def end(self) -> None:
        ...

    async def cancel(self) -> None:
        ...

    async def events(self) -> AsyncIterator[AsrEvent]:
        ...


class AsrClient(Protocol):
    def open_stream(self) -> StreamingAsrSession:
        ...

    async def healthcheck(self) -> ServiceStatus:
        ...


class TtsClient(Protocol):
    async def stream_text(self, text: str) -> AsyncIterator[bytes]:
        ...

    async def healthcheck(self) -> ServiceStatus:
        ...


class LlmClient(Protocol):
    async def validate_model(self) -> None:
        ...

    async def stream_chat(self, user_text: str) -> AsyncIterator[str]:
        ...

    async def healthcheck(self) -> ServiceStatus:
        ...
