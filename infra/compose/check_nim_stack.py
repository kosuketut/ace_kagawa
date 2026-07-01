#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import urllib.error
import urllib.request
import wave
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Health and smoke checks for ASR/TTS NIM services")
    parser.add_argument("--asr-http-url", default="http://127.0.0.1:9000")
    parser.add_argument("--tts-http-url", default="http://127.0.0.1:9001")
    parser.add_argument("--asr-grpc", default=None)
    parser.add_argument("--tts-grpc", default=None)
    parser.add_argument("--asr-wav", type=Path, default=None)
    parser.add_argument("--tts-text", default=None)
    parser.add_argument("--tts-output", type=Path, default=Path("/tmp/ace-tts-smoke.wav"))
    parser.add_argument("--tts-voice", default="Magpie-Multilingual.JA-JP.Aria.Neutral")
    parser.add_argument("--tts-language-code", default="ja-JP")
    parser.add_argument("--asr-language-code", default="multi")
    return parser


async def check_http_ready(name: str, base_url: str) -> dict[str, object]:
    url = f"{base_url.rstrip('/')}/v1/health/ready"
    try:
        await asyncio.to_thread(fetch_ready_url, url)
        return {"name": name, "ok": True, "detail": "ready", "url": url}
    except Exception as exc:
        return {"name": name, "ok": False, "detail": str(exc), "url": url}


def fetch_ready_url(url: str) -> None:
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=5.0) as response:
        status_code = getattr(response, "status", None) or response.getcode()
        if int(status_code) >= 400:
            raise RuntimeError(f"HTTP {status_code}")


def synthesize_tts(tts_grpc: str, text: str, voice: str, language_code: str, output: Path) -> dict[str, object]:
    import riva.client
    from riva.client.proto.riva_audio_pb2 import AudioEncoding

    auth = riva.client.Auth(uri=tts_grpc, use_ssl=False)
    service = riva.client.SpeechSynthesisService(auth)
    payload = bytearray()
    responses = service.synthesize_online(
        [text],
        voice,
        language_code,
        sample_rate_hz=24000,
        encoding=AudioEncoding.LINEAR_PCM,
    )
    for response in responses:
        payload.extend(response.audio)
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24000)
        handle.writeframes(bytes(payload))
    return {"name": "tts_roundtrip", "ok": True, "bytes": len(payload), "output": str(output)}


def transcribe_asr(asr_grpc: str, wav_path: Path, language_code: str) -> dict[str, object]:
    import riva.client

    with wave.open(str(wav_path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2 or handle.getframerate() != 16000:
            raise ValueError("ASR smoke WAV must be 16kHz mono PCM16")
        audio = handle.readframes(handle.getnframes())

    auth = riva.client.Auth(uri=asr_grpc, use_ssl=False)
    service = riva.client.ASRService(auth)
    config = riva.client.StreamingRecognitionConfig(
        config=riva.client.RecognitionConfig(
            language_code=language_code,
            max_alternatives=1,
            profanity_filter=False,
            enable_automatic_punctuation=True,
            verbatim_transcripts=False,
        ),
        interim_results=True,
    )
    frame_size = 320 * 2

    def audio_chunks():
        for start in range(0, len(audio), frame_size):
            yield audio[start : start + frame_size]

    transcript = ""
    responses = service.streaming_response_generator(audio_chunks=audio_chunks(), streaming_config=config)
    for response in responses:
        for result in getattr(response, "results", []):
            alternatives = getattr(result, "alternatives", [])
            if alternatives:
                transcript = alternatives[0].transcript.strip() or transcript
    return {"name": "asr_roundtrip", "ok": True, "transcript": transcript}


async def main() -> None:
    args = build_arg_parser().parse_args()
    results = [
        await check_http_ready("asr_http", args.asr_http_url),
        await check_http_ready("tts_http", args.tts_http_url),
    ]
    if args.tts_grpc and args.tts_text:
        results.append(synthesize_tts(args.tts_grpc, args.tts_text, args.tts_voice, args.tts_language_code, args.tts_output))
    if args.asr_grpc and args.asr_wav:
        results.append(transcribe_asr(args.asr_grpc, args.asr_wav, args.asr_language_code))
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
