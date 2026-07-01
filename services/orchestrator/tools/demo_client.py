#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
import wave
from pathlib import Path

import websockets

from app.protocol import AudioFrameKind, pack_audio_frame, unpack_audio_frame


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ACE orchestrator demo client")
    parser.add_argument("--url", default="ws://127.0.0.1:8080/ws/session")
    parser.add_argument("--wav", type=Path, default=None, help="16kHz mono PCM16 WAV input")
    parser.add_argument("--mock-audio", action="store_true", help="Generate 1 second of synthetic speech frames")
    parser.add_argument("--realtime", action="store_true", help="Sleep 20ms between mic frames")
    parser.add_argument("--output", type=Path, default=Path("demo-output.wav"))
    return parser


def read_input_frames(path: Path | None) -> list[bytes]:
    if path is None:
        sample = (1200).to_bytes(2, "little", signed=True) * 320
        return [sample for _ in range(50)]
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2 or handle.getframerate() != 16000:
            raise ValueError("input WAV must be 16kHz mono PCM16")
        frame_size = 320
        chunks: list[bytes] = []
        while True:
            chunk = handle.readframes(frame_size)
            if not chunk:
                break
            chunks.append(chunk)
        return chunks


async def run_demo(args: argparse.Namespace) -> None:
    frames = read_input_frames(None if args.mock_audio else args.wav)
    if not frames:
        raise ValueError("no audio frames available")
    session_id = uuid.uuid4()
    tts_audio = bytearray()
    async with websockets.connect(args.url, max_size=None) as websocket:
        await websocket.send(
            json.dumps(
                {
                    "type": "session.start",
                    "session_id": str(session_id),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "payload": {"locale": "ja-JP"},
                },
                ensure_ascii=False,
            )
        )

        async def sender() -> None:
            for frame in frames:
                payload = pack_audio_frame(kind=AudioFrameKind.MIC, sample_rate_hz=16000, channels=1, payload=frame)
                await websocket.send(payload)
                if args.realtime:
                    await asyncio.sleep(0.02)
            await websocket.send(
                json.dumps(
                    {
                        "type": "mic.end",
                        "session_id": str(session_id),
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "payload": {},
                    },
                    ensure_ascii=False,
                )
            )

        async def receiver() -> None:
            speaking = False
            while True:
                message = await websocket.recv()
                if isinstance(message, bytes):
                    frame = unpack_audio_frame(message)
                    if frame.kind is AudioFrameKind.TTS:
                        tts_audio.extend(frame.payload)
                    continue
                event = json.loads(message)
                print(json.dumps(event, ensure_ascii=False))
                event_type = event.get("type")
                if event_type == "tts.start":
                    speaking = True
                elif event_type == "tts.end":
                    speaking = False
                elif event_type == "state" and event.get("payload", {}).get("state") == "LISTENING" and not speaking:
                    break

        await asyncio.gather(sender(), receiver())

    with wave.open(str(args.output), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24000)
        handle.writeframes(bytes(tts_audio))
    print(f"wrote {args.output}")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.wav is None and not args.mock_audio:
        parser.error("either --wav or --mock-audio is required")
    asyncio.run(run_demo(args))


if __name__ == "__main__":
    main()
