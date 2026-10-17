#!/usr/bin/env python3
"""Drive the conversation server like a browser, for end-to-end checks.

Streams a WAV in real time as the microphone (or sends a typed question),
prints the server's events with timestamps, and exits once the server is
listening again after one reply.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import ssl
import sys
import time
import wave
from pathlib import Path

import numpy as np
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parent))
from speech_buffer import resample_int16  # noqa: E402

CHUNK = 320  # 20 ms at 16 kHz, like the page's AudioWorklet


def load_pcm(path: Path, pad_seconds: float) -> bytes:
    with wave.open(str(path)) as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        if w.getnchannels() > 1:
            pcm = pcm.reshape(-1, w.getnchannels()).mean(axis=1).astype(np.int16)
        pcm = resample_int16(pcm, w.getframerate(), 16000)
    silence = np.zeros(int(16000 * pad_seconds), dtype=np.int16)
    return np.concatenate([silence, pcm, silence]).tobytes()


async def probe(url: str, wav: Path | None, text: str | None, timeout: float) -> int:
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    t0 = time.monotonic()
    stamp = lambda: f"{time.monotonic() - t0:6.2f}s"  # noqa: E731
    async with websockets.connect(url, ssl=context, max_size=None) as ws:
        replied = asyncio.Event()
        done = asyncio.Event()
        marks: dict[str, float] = {}

        async def reader():
            async for raw in ws:
                event = json.loads(raw)
                print(stamp(), json.dumps(event, ensure_ascii=False))
                marks.setdefault(event.get("state") or event["type"], time.monotonic() - t0)
                if event["type"] in ("reply", "error"):
                    replied.set()
                if event.get("state") == "listening" and replied.is_set():
                    done.set()
                    return

        task = asyncio.create_task(reader())
        await asyncio.sleep(0.5)
        if text:
            await ws.send(json.dumps({"type": "text", "text": text}))
        else:
            pcm = load_pcm(wav, 1.0)
            for i in range(0, len(pcm), CHUNK * 2):
                await ws.send(pcm[i:i + CHUNK * 2])
                await asyncio.sleep(CHUNK / 16000)
                if replied.is_set() or "recognizing" in marks:
                    break
            sent_end = time.monotonic() - t0
            print(stamp(), "microphone audio ended")
            marks["mic_end"] = sent_end
        try:
            await asyncio.wait_for(done.wait(), timeout)
        finally:
            task.cancel()
        print("marks:", {k: round(v, 2) for k, v in marks.items()})
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="wss://127.0.0.1:18443/mic")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--wav", type=Path)
    group.add_argument("--text")
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    return asyncio.run(probe(args.url, args.wav, args.text, args.timeout))


if __name__ == "__main__":
    raise SystemExit(main())
