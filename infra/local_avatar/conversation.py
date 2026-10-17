"""LLM reply streamed sentence by sentence into TTS.

Ported from ELF-SR2app (mars: scripts/chat_to_wav.py): the reply is split as
it streams, the first sentence is cut at its first comma so the avatar starts
sooner, and one worker synthesizes the pieces in order. The ELF pause of the
LLM during the first TTS is left out: here the LLM (GPU0) and Irodori (GPU1)
do not share a GPU.
"""

from __future__ import annotations

import json
import queue
import sys
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from urllib import error, request

ROOT = Path(__file__).resolve().parents[2]

SENTENCE_END = "。！？!?\n"
CLOSERS = "」』）)】\"'"
SOFT_BREAK = "、，,"
MAX_SENTENCE = 60
MAX_SEGMENTS = 200
FIRST_CLAUSE_MIN = 4  # Characters to speak before the early cut, so "あ、" is not sent alone.
MAX_WAV = 12 * 1024 * 1024
RETRY_HTTP = (429, 500, 502, 503, 504)
HISTORY_MESSAGES = 16


def split_sentences(buffer: str, final: bool = False) -> tuple[list[str], str]:
    """Split streamed text into speakable sentences. Returns (sentences, remainder)."""
    sentences, start, i = [], 0, 0
    while i < len(buffer):
        if buffer[i] in SENTENCE_END:
            end = i + 1
            while end < len(buffer) and buffer[end] in CLOSERS:
                end += 1
            sentences.append(buffer[start:end])
            start = i = end
            continue
        i += 1
    rest = buffer[start:]
    # A long clause without a full stop would delay the first audio: cut at the last soft break.
    if len(rest) > MAX_SENTENCE:
        cut = max(rest.rfind(c, 0, MAX_SENTENCE) for c in SOFT_BREAK)
        cut = cut + 1 if cut > 0 else MAX_SENTENCE
        sentences.append(rest[:cut])
        rest = rest[cut:]
    if final and rest:
        sentences.append(rest)
        rest = ""
    # Drop pieces with nothing to pronounce (bare punctuation, whitespace).
    return [s.strip() for s in sentences if any(c.isalnum() for c in s)], rest


def first_clause(buffer: str) -> tuple[str | None, str]:
    """Cut the reply's first sentence at its first comma: TTS time grows with text length."""
    end = next((i for i, c in enumerate(buffer) if c in SENTENCE_END), len(buffer))
    for i, c in enumerate(buffer[:end]):
        if c in SOFT_BREAK and sum(ch.isalnum() for ch in buffer[:i]) >= FIRST_CLAUSE_MIN:
            return buffer[:i + 1].strip(), buffer[i + 1:]
    return None, buffer


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def fetch(url: str, payload: dict | None = None, limit: int = 1024 * 1024, timeout: float = 120) -> bytes:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with request.build_opener(_NoRedirect()).open(req, timeout=timeout) as response:
        body = response.read(limit + 1)
    if len(body) > limit:
        raise ValueError("Service response exceeds the size limit")
    return body


def stream_chat(url: str, payload: dict, api_key: str = "", limit: int = 1024 * 1024,
                retry_delay: float = 1.0) -> Iterator[tuple[str, str | None]]:
    """Yield (content_delta, finish_reason) from an OpenAI-compatible SSE stream.

    A request refused before the stream opens with a temporary error is sent once more."""
    headers = {"Accept": "text/event-stream", "Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = request.Request(url, data=json.dumps(dict(payload, stream=True), ensure_ascii=False).encode("utf-8"),
                          headers=headers)
    opener = request.build_opener(_NoRedirect())
    for attempt in (1, 2):
        try:
            response = opener.open(req, timeout=120)
            break
        except error.HTTPError as exc:
            if attempt == 2 or exc.code not in RETRY_HTTP:
                raise
        except error.URLError as exc:
            if attempt == 2 or isinstance(exc.reason, TimeoutError):
                raise
        time.sleep(retry_delay)
    received = 0
    with response:
        for raw in response:
            received += len(raw)
            if received > limit:
                raise ValueError("Service response exceeds the size limit")
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                return
            choice = (json.loads(data).get("choices") or [{}])[0]
            yield (choice.get("delta") or {}).get("content") or "", choice.get("finish_reason")


def synthesize(tts_base_url: str, voice: str, text: str) -> bytes:
    return fetch(tts_base_url.rstrip("/") + "/v1/audio/speech",
                 {"input": text, "voice": voice, "response_format": "wav", "stream": False}, limit=MAX_WAV)


def build_messages(system_prompt: str, history: list[dict], text: str) -> list[dict]:
    return [{"role": "system", "content": system_prompt}, *history[-HISTORY_MESSAGES:],
            {"role": "user", "content": text}]


class ReplyStreamer:
    """LLM stream -> sentences -> TTS per sentence -> on_segment(wav, text) as each is ready."""

    def __init__(self, profile: dict,
                 stream: Callable[[dict], Iterable[tuple[str, str | None]]] | None = None,
                 synthesize: Callable[[str], bytes] | None = None) -> None:
        self.profile = profile
        llm = profile["llm_base_url"].rstrip("/") + "/chat/completions"
        self._stream = stream or (lambda payload: stream_chat(llm, payload))
        self._synthesize = synthesize or (
            lambda text: globals()["synthesize"](profile["tts_base_url"], profile["tts_voice"], text))

    def run(self, messages: list[dict], on_segment: Callable[[bytes, str], None]) -> tuple[str, dict]:
        profile = self.profile
        limit = int(profile.get("max_reply_chars", 400))
        start = time.monotonic()
        sentences: queue.Queue[str | None] = queue.Queue()
        state = {"error": None, "first_audio": None, "tts_seconds": 0.0, "segments": 0}

        def speak() -> None:
            # One TTS request at a time, in sentence order, overlapping with LLM streaming.
            while (sentence := sentences.get()) is not None:
                if state["error"]:
                    continue
                try:
                    began = time.monotonic()
                    audio = self._synthesize(sentence)
                    state["tts_seconds"] += time.monotonic() - began
                    on_segment(audio, sentence)
                    state["segments"] += 1
                    if state["first_audio"] is None:
                        state["first_audio"] = time.monotonic() - start
                except Exception as exc:  # Surfaced to the caller after the worker drains.
                    state["error"] = exc

        reply, pending, finish, first_token = "", "", None, None
        early = profile.get("split_first_clause", True)
        queued = 0
        worker = threading.Thread(target=speak, daemon=True, name="tts-worker")
        worker.start()
        try:
            payload = {"model": profile["llm_model"], "messages": messages,
                       "temperature": float(profile.get("temperature", 0.0)),
                       "max_tokens": int(profile.get("max_tokens", 256))}
            for delta, finish_reason in self._stream(payload):
                if state["error"]:
                    break
                if delta and first_token is None:
                    first_token = time.monotonic() - start
                finish = finish_reason or finish
                if not delta:
                    continue
                reply += delta
                pending += delta
                if "<think" in reply.lower() or len(reply) > limit:
                    raise ValueError("LLM reply is too long or contains reasoning markup; speech stopped")
                if early:
                    head, pending = first_clause(pending)
                    if head:
                        sentences.put(head)
                        queued += 1
                        early = False
                done, pending = split_sentences(pending)
                for sentence in done:
                    if queued >= MAX_SEGMENTS:
                        raise ValueError("Reply has too many segments")
                    sentences.put(sentence)
                    queued += 1
                    early = False
            if not state["error"]:
                for sentence in split_sentences(pending, final=True)[0]:
                    sentences.put(sentence)
        finally:
            sentences.put(None)
            worker.join()
        if state["error"]:
            raise state["error"]
        if finish != "stop":
            raise ValueError(f"LLM did not finish a normal text reply (finish_reason={finish})")
        if not reply.strip() or not state["segments"]:
            raise ValueError("LLM reply is empty; no speech generated")
        return reply.strip(), {
            "llm_first_token_seconds": round(first_token or 0.0, 3),
            "first_audio_seconds": round(state["first_audio"], 3),
            "total_seconds": round(time.monotonic() - start, 3),
            "tts_seconds": round(state["tts_seconds"], 3),
            "segments": state["segments"],
        }


def tokkio_avatar_prompt(variant_root: Path = ROOT) -> str:
    """The system prompt the Tokkio controller uses for this variant (single source of truth)."""
    sys.path.insert(0, str(variant_root / "infra" / "tokkio"))
    try:
        import customize_tokkio_japanese as custom
        import yaml
    finally:
        sys.path.pop(0)
    context = yaml.safe_load(custom.CONFIG_YAML)["OpenAILLMContext"]
    return context["prompt"].replace("{name}", context["name"])
