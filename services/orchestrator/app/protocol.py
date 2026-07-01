from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum, IntEnum
from typing import Any
from uuid import UUID


FRAME_MAGIC = b"ACE1"
FRAME_VERSION = 1
PCM_S16LE = 1
FRAME_HEADER = struct.Struct("!4sBBBBII16s")


class SessionState(str, Enum):
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"


class AudioFrameKind(IntEnum):
    MIC = 1
    TTS = 2


@dataclass(slots=True)
class ControlMessage:
    type: str
    session_id: str | None
    turn_id: str | None
    timestamp: str
    payload: dict[str, Any]


@dataclass(slots=True)
class AudioFrame:
    kind: AudioFrameKind
    codec: int
    channels: int
    sample_rate_hz: int
    payload: bytes
    turn_id: UUID | None


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def serialize_control_message(
    *,
    event_type: str,
    session_id: UUID | None,
    turn_id: UUID | None,
    payload: dict[str, Any] | None = None,
) -> str:
    body = {
        "type": event_type,
        "session_id": None if session_id is None else str(session_id),
        "turn_id": None if turn_id is None else str(turn_id),
        "timestamp": utc_timestamp(),
        "payload": payload or {},
    }
    return json.dumps(body, ensure_ascii=False)


def parse_control_message(raw: str) -> ControlMessage:
    data = json.loads(raw)
    if "type" not in data:
        raise ValueError("control frame is missing type")
    return ControlMessage(
        type=str(data["type"]),
        session_id=data.get("session_id"),
        turn_id=data.get("turn_id"),
        timestamp=data.get("timestamp") or utc_timestamp(),
        payload=data.get("payload") or {},
    )


def pack_audio_frame(
    *,
    kind: AudioFrameKind,
    sample_rate_hz: int,
    channels: int,
    payload: bytes,
    turn_id: UUID | None = None,
) -> bytes:
    turn_bytes = (turn_id.bytes if turn_id else bytes(16))
    header = FRAME_HEADER.pack(
        FRAME_MAGIC,
        FRAME_VERSION,
        int(kind),
        PCM_S16LE,
        channels,
        sample_rate_hz,
        len(payload),
        turn_bytes,
    )
    return header + payload


def unpack_audio_frame(raw: bytes) -> AudioFrame:
    if len(raw) < FRAME_HEADER.size:
        raise ValueError("binary frame is too short")
    magic, version, kind, codec, channels, sample_rate_hz, payload_size, turn_bytes = FRAME_HEADER.unpack(
        raw[: FRAME_HEADER.size]
    )
    if magic != FRAME_MAGIC:
        raise ValueError("invalid frame magic")
    if version != FRAME_VERSION:
        raise ValueError(f"unsupported frame version: {version}")
    payload = raw[FRAME_HEADER.size :]
    if len(payload) != payload_size:
        raise ValueError("payload size mismatch")
    turn_id = None if turn_bytes == bytes(16) else UUID(bytes=turn_bytes)
    return AudioFrame(
        kind=AudioFrameKind(kind),
        codec=codec,
        channels=channels,
        sample_rate_hz=sample_rate_hz,
        payload=payload,
        turn_id=turn_id,
    )

