from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

from app.adapters.base import AsrEvent
from app.audio_store import TurnAudioArtifactStore
from app.logging_utils import JsonlTurnLogger
from app.protocol import AudioFrameKind, pack_audio_frame, unpack_audio_frame
from app.session import ConversationSession


class FakeWebSocket:
    def __init__(self) -> None:
        self.incoming: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        self.text_messages: list[str] = []
        self.binary_messages: list[bytes] = []
        self.accepted = False

    async def accept(self) -> None:
        self.accepted = True

    async def receive(self) -> dict[str, object]:
        return await self.incoming.get()

    async def send_text(self, data: str) -> None:
        self.text_messages.append(data)

    async def send_bytes(self, data: bytes) -> None:
        self.binary_messages.append(data)


class FakeAsrStream:
    def __init__(self) -> None:
        self.audio = bytearray()
        self.queue: asyncio.Queue[AsrEvent | None] = asyncio.Queue()

    async def push_audio(self, chunk: bytes) -> None:
        self.audio.extend(chunk)
        if len(self.audio) >= 640:
            await self.queue.put(AsrEvent(kind="partial", text="こんにちは"))

    async def end(self) -> None:
        await self.queue.put(AsrEvent(kind="final", text="こんにちは"))
        await self.queue.put(None)

    async def cancel(self) -> None:
        await self.queue.put(None)

    async def events(self):
        while True:
            item = await self.queue.get()
            if item is None:
                break
            yield item


class FakeAsrClient:
    def open_stream(self) -> FakeAsrStream:
        return FakeAsrStream()


class FakeLlmClient:
    async def stream_chat(self, user_text: str):
        yield "はい、"
        yield "応答します。"


class FakeTtsClient:
    async def stream_text(self, text: str):
        yield b"\x00\x00" * 240


class SessionFlowTests(unittest.TestCase):
    def test_conversation_session_runs_complete_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            asyncio.run(self._run_flow(Path(tmp)))

    async def _run_flow(self, root: Path) -> None:
        websocket = FakeWebSocket()
        settings = SimpleNamespace(
            asr_sample_rate_hz=16000,
            asr_frame_ms=20,
            eos_silence_ms=500,
            vad_aggressiveness=2,
            tts_sample_rate_hz=24000,
            save_debug_audio=True,
        )
        session = ConversationSession(
            websocket=websocket,
            settings=settings,
            turn_logger=JsonlTurnLogger(root / "logs"),
            audio_store=TurnAudioArtifactStore(root / "audio"),
            asr_client=FakeAsrClient(),
            llm_client=FakeLlmClient(),
            tts_client=FakeTtsClient(),
        )

        speech_frame = (1000).to_bytes(2, "little", signed=True) * 320
        await websocket.incoming.put(
            {
                "type": "websocket.receive",
                "text": json.dumps(
                    {
                        "type": "session.start",
                        "session_id": None,
                        "timestamp": "2026-04-21T00:00:00Z",
                        "payload": {"locale": "ja-JP"},
                    },
                    ensure_ascii=False,
                ),
            }
        )
        await websocket.incoming.put(
            {
                "type": "websocket.receive",
                "bytes": pack_audio_frame(
                    kind=AudioFrameKind.MIC,
                    sample_rate_hz=16000,
                    channels=1,
                    payload=speech_frame,
                ),
            }
        )
        await websocket.incoming.put(
            {
                "type": "websocket.receive",
                "text": json.dumps(
                    {
                        "type": "mic.end",
                        "session_id": None,
                        "timestamp": "2026-04-21T00:00:01Z",
                        "payload": {},
                    },
                    ensure_ascii=False,
                ),
            }
        )
        await websocket.incoming.put({"type": "websocket.disconnect"})

        await session.run()

        events = [json.loads(message)["type"] for message in websocket.text_messages]
        self.assertIn("asr.partial", events)
        self.assertIn("asr.final", events)
        self.assertIn("llm.delta", events)
        self.assertIn("tts.start", events)
        self.assertIn("tts.end", events)
        self.assertIn("state", events)
        self.assertTrue(websocket.binary_messages)
        decoded = unpack_audio_frame(websocket.binary_messages[0])
        self.assertEqual(decoded.kind, AudioFrameKind.TTS)
        self.assertIsInstance(decoded.turn_id, UUID)


if __name__ == "__main__":
    unittest.main()

