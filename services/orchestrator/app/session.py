from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol
from uuid import UUID, uuid4

from app.adapters.base import AsrClient, AsrEvent, LlmClient, StreamingAsrSession, TtsClient
from app.audio_store import TurnAudioArtifactStore
from app.logging_utils import JsonlTurnLogger, TurnTimer
from app.protocol import (
    AudioFrameKind,
    SessionState,
    pack_audio_frame,
    parse_control_message,
    serialize_control_message,
    unpack_audio_frame,
)
from app.text import SentenceChunker, normalize_tts_text
from app.vad import TurnDetector

if TYPE_CHECKING:
    from app.settings import Settings


LOGGER = logging.getLogger(__name__)


class WebSocketLike(Protocol):
    async def accept(self) -> None:
        ...

    async def receive(self) -> dict[str, Any]:
        ...

    async def send_text(self, data: str) -> None:
        ...

    async def send_bytes(self, data: bytes) -> None:
        ...


@dataclass(slots=True)
class ActiveTurn:
    turn_id: UUID
    timer: TurnTimer
    asr_stream: StreamingAsrSession
    asr_task: asyncio.Task[str]
    mic_audio: bytearray = field(default_factory=bytearray)
    tts_audio: bytearray = field(default_factory=bytearray)


class ConversationSession:
    def __init__(
        self,
        *,
        websocket: WebSocketLike,
        settings: "Settings",
        turn_logger: JsonlTurnLogger,
        audio_store: TurnAudioArtifactStore,
        asr_client: AsrClient,
        llm_client: LlmClient,
        tts_client: TtsClient,
    ) -> None:
        self._websocket = websocket
        self._settings = settings
        self._turn_logger = turn_logger
        self._audio_store = audio_store
        self._asr_client = asr_client
        self._llm_client = llm_client
        self._tts_client = tts_client
        self._session_id = uuid4()
        self._state = SessionState.LISTENING
        self._send_lock = asyncio.Lock()
        self._started = False
        self._locale = "ja-JP"
        self._active_turn: ActiveTurn | None = None
        self._final_transcript = ""
        self._last_partial = ""
        self._finishing_turn = False
        self._vad = TurnDetector(
            sample_rate_hz=self._settings.asr_sample_rate_hz,
            frame_ms=self._settings.asr_frame_ms,
            eos_silence_ms=self._settings.eos_silence_ms,
            aggressiveness=self._settings.vad_aggressiveness,
        )

    async def run(self) -> None:
        await self._websocket.accept()
        try:
            while True:
                message = await self._websocket.receive()
                msg_type = message["type"]
                if msg_type == "websocket.disconnect":
                    break
                if msg_type == "websocket.receive":
                    if message.get("text") is not None:
                        await self._handle_text(message["text"])
                    elif message.get("bytes") is not None:
                        await self._handle_binary(message["bytes"])
        finally:
            await self._cleanup()

    async def _handle_text(self, raw: str) -> None:
        try:
            message = parse_control_message(raw)
        except Exception as exc:
            await self._send_error("invalid_control_frame", str(exc))
            return
        if message.type == "session.start":
            await self._handle_session_start(message)
            return
        if not self._started:
            await self._send_error("session_not_started", "session.start is required before audio streaming")
            return
        if message.type == "mic.end":
            await self._handle_turn_end(reason="client_end")
            return
        await self._send_error("unknown_event", f"unsupported control event: {message.type}")

    async def _handle_session_start(self, message: Any) -> None:
        if self._started:
            return
        requested_session_id = message.session_id or message.payload.get("session_id")
        if requested_session_id:
            try:
                self._session_id = UUID(str(requested_session_id))
            except ValueError:
                LOGGER.warning("Ignoring invalid client session_id: %s", requested_session_id)
        self._locale = str(message.payload.get("locale") or "ja-JP")
        self._started = True
        await self._set_state(SessionState.LISTENING, reason="ready")

    async def _handle_binary(self, raw: bytes) -> None:
        if not self._started:
            await self._send_error("session_not_started", "audio received before session.start")
            return
        try:
            frame = unpack_audio_frame(raw)
        except Exception as exc:
            await self._send_error("invalid_audio_frame", str(exc))
            return
        if frame.kind is not AudioFrameKind.MIC:
            await self._send_error("invalid_audio_kind", "server accepts only mic binary frames from client")
            return
        if frame.sample_rate_hz != self._settings.asr_sample_rate_hz or frame.channels != 1:
            await self._send_error(
                "invalid_audio_format",
                f"expected {self._settings.asr_sample_rate_hz}Hz mono PCM16 mic frames",
            )
            return
        if self._state is not SessionState.LISTENING:
            return

        vad_event = self._vad.feed(frame.payload)
        if vad_event.speech_started and self._active_turn is None:
            await self._start_turn()
        if self._active_turn is not None:
            self._active_turn.mic_audio.extend(frame.payload)
            await self._active_turn.asr_stream.push_audio(frame.payload)
        if vad_event.end_of_utterance:
            await self._handle_turn_end(reason="vad_eou")

    async def _start_turn(self) -> None:
        turn_id = uuid4()
        stream = self._asr_client.open_stream()
        timer = TurnTimer()
        timer.mark("vad_start")
        self._final_transcript = ""
        self._last_partial = ""
        asr_task = asyncio.create_task(self._consume_asr_events(stream))
        self._active_turn = ActiveTurn(turn_id=turn_id, timer=timer, asr_stream=stream, asr_task=asr_task)
        LOGGER.info("turn started session=%s turn=%s", self._session_id, turn_id)

    async def _handle_turn_end(self, *, reason: str) -> None:
        if self._active_turn is None or self._finishing_turn:
            return
        self._finishing_turn = True
        turn = self._active_turn
        turn.timer.mark("eou_detected")
        await turn.asr_stream.end()
        transcript = (await turn.asr_task).strip()
        transcript = transcript or self._last_partial.strip()
        if not transcript:
            await self._finish_turn(turn_id=turn.turn_id, user_text="", assistant_text="", timer=turn.timer)
            return
        try:
            await self._set_state(SessionState.THINKING, reason=reason, turn_id=turn.turn_id)
            assistant_text = await self._run_llm_and_tts(turn.turn_id, transcript, turn.timer)
        except Exception as exc:
            LOGGER.exception("turn failed session=%s turn=%s", self._session_id, turn.turn_id)
            await self._send_error("turn_failure", str(exc), recoverable=True)
            assistant_text = ""
        await self._finish_turn(turn_id=turn.turn_id, user_text=transcript, assistant_text=assistant_text, timer=turn.timer)

    async def _consume_asr_events(self, stream: StreamingAsrSession) -> str:
        final_text = ""
        async for event in stream.events():
            await self._handle_asr_event(event)
            if event.kind == "final":
                final_text = event.text
        return final_text

    async def _handle_asr_event(self, event: AsrEvent) -> None:
        if self._active_turn is None:
            return
        turn_id = self._active_turn.turn_id
        if event.kind == "partial":
            self._last_partial = event.text
            await self._send_control("asr.partial", {"text": event.text}, turn_id=turn_id)
            return
        if event.kind == "final":
            self._final_transcript = event.text
            self._active_turn.timer.mark("asr_final")
            await self._send_control("asr.final", {"text": event.text}, turn_id=turn_id)

    async def _run_llm_and_tts(self, turn_id: UUID, transcript: str, timer: TurnTimer) -> str:
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        tts_task = asyncio.create_task(self._tts_consumer(turn_id, queue, timer))
        chunker = SentenceChunker()
        llm_parts: list[str] = []
        first_token_seen = False
        try:
            async for delta in self._llm_client.stream_chat(transcript):
                if not first_token_seen:
                    timer.mark("llm_first_token")
                    first_token_seen = True
                llm_parts.append(delta)
                await self._send_control("llm.delta", {"text": delta}, turn_id=turn_id)
                for sentence in chunker.push(delta):
                    if sentence:
                        await queue.put(sentence)
            remainder = chunker.flush()
            if remainder:
                await queue.put(remainder)
        finally:
            await queue.put(None)
            await tts_task
        return normalize_tts_text("".join(llm_parts))

    async def _tts_consumer(self, turn_id: UUID, queue: asyncio.Queue[str | None], timer: TurnTimer) -> None:
        tts_started = False
        while True:
            sentence = await queue.get()
            if sentence is None:
                break
            if not sentence:
                continue
            async for chunk in self._tts_client.stream_text(sentence):
                if not tts_started:
                    await self._set_state(SessionState.SPEAKING, reason="tts_start", turn_id=turn_id)
                    await self._send_control(
                        "tts.start",
                        {"sample_rate_hz": self._settings.tts_sample_rate_hz, "channels": 1},
                        turn_id=turn_id,
                    )
                    timer.mark("tts_first_audio")
                    timer.mark("a2f_start")
                    tts_started = True
                if self._active_turn is not None:
                    self._active_turn.tts_audio.extend(chunk)
                await self._send_audio(
                    kind=AudioFrameKind.TTS,
                    payload=chunk,
                    sample_rate_hz=self._settings.tts_sample_rate_hz,
                    turn_id=turn_id,
                )
        if tts_started:
            await self._send_control("tts.end", {}, turn_id=turn_id)

    async def _finish_turn(self, *, turn_id: UUID, user_text: str, assistant_text: str, timer: TurnTimer) -> None:
        metadata: dict[str, Any] = {"locale": self._locale}
        if self._settings.save_debug_audio and self._active_turn is not None:
            metadata["artifacts"] = self._audio_store.write_turn_audio(
                session_id=self._session_id,
                turn_id=turn_id,
                mic_audio=bytes(self._active_turn.mic_audio),
                tts_audio=bytes(self._active_turn.tts_audio),
                tts_sample_rate_hz=self._settings.tts_sample_rate_hz,
            )
        try:
            self._turn_logger.write_turn(
                session_id=self._session_id,
                turn_id=turn_id,
                user_text=user_text,
                assistant_text=assistant_text,
                timer=timer,
                metadata=metadata,
            )
        finally:
            self._active_turn = None
            self._final_transcript = ""
            self._last_partial = ""
            self._finishing_turn = False
            self._vad.reset()
            await self._set_state(SessionState.LISTENING, reason="turn_complete")

    async def _set_state(self, state: SessionState, *, reason: str, turn_id: UUID | None = None) -> None:
        self._state = state
        await self._send_control("state", {"state": state.value, "reason": reason}, turn_id=turn_id)

    async def _send_control(self, event_type: str, payload: dict[str, Any], *, turn_id: UUID | None = None) -> None:
        message = serialize_control_message(
            event_type=event_type,
            session_id=self._session_id,
            turn_id=turn_id,
            payload=payload,
        )
        async with self._send_lock:
            await self._websocket.send_text(message)

    async def _send_audio(
        self,
        *,
        kind: AudioFrameKind,
        payload: bytes,
        sample_rate_hz: int,
        turn_id: UUID | None = None,
    ) -> None:
        frame = pack_audio_frame(
            kind=kind,
            sample_rate_hz=sample_rate_hz,
            channels=1,
            payload=payload,
            turn_id=turn_id,
        )
        async with self._send_lock:
            await self._websocket.send_bytes(frame)

    async def _send_error(self, code: str, message: str, recoverable: bool = True) -> None:
        await self._send_control(
            "error",
            {"code": code, "message": message, "recoverable": recoverable},
            turn_id=self._active_turn.turn_id if self._active_turn else None,
        )

    async def _cleanup(self) -> None:
        if self._active_turn is not None:
            try:
                await self._active_turn.asr_stream.cancel()
            except Exception:
                LOGGER.exception("failed to cancel ASR stream")
