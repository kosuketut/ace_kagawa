from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket

from app.adapters.asr import build_asr_client
from app.adapters.llm import build_llm_client
from app.adapters.tts import build_tts_client
from app.audio_store import TurnAudioArtifactStore
from app.logging_utils import JsonlTurnLogger
from app.session import ConversationSession
from app.service_status import ServiceStatus
from app.settings import Settings


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    settings.ensure_runtime_dirs()
    llm_client = build_llm_client(settings)
    app.state.settings = settings
    app.state.turn_logger = JsonlTurnLogger(settings.log_dir)
    app.state.audio_store = TurnAudioArtifactStore(settings.audio_dir)
    app.state.asr_client = build_asr_client(settings)
    app.state.llm_client = llm_client
    app.state.tts_client = build_tts_client(settings)
    if settings.validate_externals_on_startup:
        statuses = await collect_service_statuses(app)
        failed = [status for status in statuses if not status.ok]
        if failed:
            details = ", ".join(f"{status.name}: {status.detail}" for status in failed)
            raise RuntimeError(f"external validation failed: {details}")
    try:
        yield
    finally:
        close = getattr(llm_client, "close", None)
        if callable(close):
            await close()


app = FastAPI(title="ACE Orchestrator", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/status")
async def status() -> dict[str, object]:
    statuses = await collect_service_statuses(app)
    return {
        "status": "ok" if all(item.ok for item in statuses) else "degraded",
        "services": [item.as_dict() for item in statuses],
        "config": {
            "mock_asr": app.state.settings.mock_asr,
            "mock_tts": app.state.settings.mock_tts,
            "mock_llm": app.state.settings.mock_llm,
            "save_debug_audio": app.state.settings.save_debug_audio,
            "tts_voice": app.state.settings.tts_voice,
        },
    }


@app.websocket("/ws/session")
async def session_ws(websocket: WebSocket) -> None:
    session = ConversationSession(
        websocket=websocket,
        settings=app.state.settings,
        turn_logger=app.state.turn_logger,
        audio_store=app.state.audio_store,
        asr_client=app.state.asr_client,
        llm_client=app.state.llm_client,
        tts_client=app.state.tts_client,
    )
    await session.run()


async def collect_service_statuses(app: FastAPI) -> list[ServiceStatus]:
    return [
        await app.state.asr_client.healthcheck(),
        await app.state.tts_client.healthcheck(),
        await app.state.llm_client.healthcheck(),
    ]
