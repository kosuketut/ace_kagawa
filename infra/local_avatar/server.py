#!/usr/bin/env python3
"""Tokkio-free conversation server: browser microphone in, avatar out.

One HTTPS origin serves the page, the microphone WebSocket, the Pixel
Streaming player and a proxy to its signalling server, so the browser allows
microphone access and does not block the player as mixed content.

A turn follows ELF-SR2app's automatic mode (mars: scripts/voice_chat.py):
listen with VAD -> recognize (started early on a short pause) -> LLM reply
spoken sentence by sentence -> wait until the avatar has finished plus a short
tail -> listen again. Microphone audio that arrives outside listening is
dropped (half duplex, no barge-in).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import uvicorn
import webrtcvad
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import a2f_speak
import avatar_output
import conversation
import listening

LOG = logging.getLogger("local_avatar")
HERE = Path(__file__).resolve().parent


def load_profile(path: Path) -> dict:
    profile = json.loads(path.read_text(encoding="utf-8"))
    if profile.get("system_prompt") == "tokkio":
        profile["system_prompt"] = conversation.tokkio_avatar_prompt()
    return profile


class Turns:
    """Shared services for the one active conversation."""

    def __init__(self, args: argparse.Namespace, profile: dict) -> None:
        self.args = args
        self.profile = profile
        asr = profile["asr"]
        LOG.info("loading ASR %s", asr["model_dir"])
        self.asr = listening.WhisperASR(asr["model_dir"], device_index=int(asr.get("device_index", 0)),
                                        compute_type=asr.get("compute_type", "float16"),
                                        beam_size=int(asr.get("beam_size", 5)))
        self.streamer = conversation.ReplyStreamer(profile)
        self.voice = avatar_output.AvatarVoice(
            speak=lambda pcm: a2f_speak.speak(pcm, args.a2f, args.relay),
            status=lambda: avatar_output.relay_status(args.relay_status),
        )
        self.active = False
        self.session_dir = args.log_dir / datetime.now().strftime("%Y%m%d-%H%M%S")


def create_app(turns: Turns) -> FastAPI:
    args = turns.args
    app = FastAPI()
    app.mount("/ps", StaticFiles(directory=args.ps_www), name="ps")

    @app.get("/")
    async def index():
        return FileResponse(HERE / "web" / "index.html")

    @app.get("/mic-worklet.js")
    async def worklet():
        return FileResponse(HERE / "web" / "mic-worklet.js", media_type="text/javascript")

    @app.get("/healthz")
    async def healthz():
        try:
            relay = await asyncio.to_thread(avatar_output.relay_status, args.relay_status)
        except OSError as exc:
            relay = {"error": type(exc).__name__}
        return JSONResponse({"status": "ok", "relay": relay, "conversation_active": turns.active})

    @app.websocket("/ps-signal")
    async def ps_signal(client: WebSocket):
        await client.accept()
        async with websockets.connect(args.ps_signalling, max_size=None) as upstream:
            async def down():
                async for message in upstream:
                    await (client.send_text(message) if isinstance(message, str) else client.send_bytes(message))

            async def up():
                while True:
                    message = await client.receive()
                    if message["type"] == "websocket.disconnect":
                        return
                    await upstream.send(message.get("text") if message.get("text") is not None
                                        else message.get("bytes"))

            tasks = [asyncio.create_task(down()), asyncio.create_task(up())]
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in tasks:
                task.cancel()

    @app.websocket("/mic")
    async def mic(ws: WebSocket):
        await ws.accept()
        if turns.active:
            await ws.send_json({"type": "error", "message": "別の画面で会話中です。"})
            await ws.close()
            return
        turns.active = True
        try:
            await Conversation(turns, ws).run()
        finally:
            turns.active = False

    return app


class Conversation:
    def __init__(self, turns: Turns, ws: WebSocket) -> None:
        self.turns = turns
        self.ws = ws
        self.loop = asyncio.get_running_loop()
        self.vad = webrtcvad.Vad(2)
        self.chunker = listening.FrameChunker()
        self.early = listening.EarlyASR(turns.asr)
        options = turns.profile.get("auto_voice", {})
        self.reply_tail = int(options.get("reply_tail_ms", 500)) / 1000
        self.detector = listening.Utterance(int(options.get("end_silence_ms", listening.END_SILENCE_MS)),
                                            on_pause=self.early.pause, on_resume=self.early.resume,
                                            early_ms=int(options.get("early_asr_ms", listening.EARLY_ASR_MS)))
        self.listening = False
        self.history: list[dict] = []
        self.turn_no = 0
        self.task: asyncio.Task | None = None
        self.speaking_sent = False

    async def send(self, **event) -> None:
        try:
            await self.ws.send_json(event)
        except (WebSocketDisconnect, RuntimeError):
            pass

    async def listen(self) -> None:
        self.detector.reset()
        self.chunker.reset()
        self.early.resume()
        self.listening = True
        await self.send(type="state", state="listening")

    async def run(self) -> None:
        await self.listen()
        try:
            while True:
                message = await self.ws.receive()
                if message["type"] == "websocket.disconnect":
                    return
                if message.get("bytes") is not None:
                    self.on_audio(message["bytes"])
                elif message.get("text"):
                    data = json.loads(message["text"])
                    if data.get("type") == "text" and self.listening and data.get("text", "").strip():
                        self.start_turn(text=data["text"].strip())
                    elif data.get("type") == "reset":
                        self.history.clear()
                        await self.send(type="info", message="会話履歴を消去しました。")
        finally:
            if self.task:
                self.task.cancel()
            self.early.close()

    def on_audio(self, data: bytes) -> None:
        if not self.listening:
            return
        for frame in self.chunker.push(data):
            was_active = self.detector.active
            try:
                pcm = self.detector.feed(frame, self.vad.is_speech(frame, listening.RATE))
            except listening.Discarded as exc:
                asyncio.ensure_future(self.send(type="info", message=str(exc)))
                continue
            if self.detector.active and not was_active:
                asyncio.ensure_future(self.send(type="state", state="hearing"))
            if pcm is not None:
                self.start_turn(pcm=pcm)
                return

    def start_turn(self, pcm: bytes | None = None, text: str | None = None) -> None:
        self.listening = False
        self.task = asyncio.create_task(self.turn(pcm, text))

    async def turn(self, pcm: bytes | None, text: str | None) -> None:
        turns = self.turns
        record = {"time": datetime.now().isoformat(timespec="seconds")}
        replied = False
        self.speaking_sent = False
        try:
            if pcm is not None:
                await self.send(type="state", state="recognizing")
                began = time.monotonic()
                text = await asyncio.to_thread(self.early.result, pcm)
                record["asr_seconds"] = round(time.monotonic() - began, 3)
            record["input"] = text
            await self.send(type="transcript", text=text)
            await self.send(type="state", state="thinking")
            messages = conversation.build_messages(turns.profile["system_prompt"], self.history, text)
            replied = True
            reply, meta = await asyncio.to_thread(turns.streamer.run, messages, self.say)
            record.update(reply=reply, **meta)
            self.history += [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
            await self.send(type="reply", text=reply, first_audio_seconds=meta["first_audio_seconds"])
        except listening.Unheard as exc:
            await self.send(type="info", message=str(exc))
        except Exception as exc:
            LOG.exception("turn failed")
            record["error"] = f"{type(exc).__name__}: {exc}"
            await self.send(type="error", message=f"このターンは失敗しました（{type(exc).__name__}）。")
        finally:
            if replied:
                if not self.speaking_sent:
                    await self.send(type="state", state="speaking")
                try:
                    await asyncio.to_thread(turns.voice.wait_idle)
                except Exception as exc:
                    LOG.warning("avatar did not finish: %s", exc)
                if turns.voice.error is not None:
                    LOG.error("avatar output failed: %r", turns.voice.error)
                    record["avatar_error"] = f"{type(turns.voice.error).__name__}: {turns.voice.error}"
                    turns.voice.error = None
                    await self.send(type="error", message="アバターへの音声・表情の送信に失敗しました（A2F・中継を確認）。")
                await asyncio.sleep(self.reply_tail)
            if "input" in record:
                self.turn_no += 1
                turns.session_dir.mkdir(parents=True, exist_ok=True)
                (turns.session_dir / f"{self.turn_no:03}.json").write_text(
                    json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            await self.listen()

    def say(self, wav: bytes, text: str) -> None:
        self.turns.voice.say(wav)
        if not self.speaking_sent:
            # The avatar starts with the first sentence while the LLM may still be writing.
            self.speaking_sent = True
            asyncio.run_coroutine_threadsafe(self.send(type="state", state="speaking"), self.loop)
        asyncio.run_coroutine_threadsafe(self.send(type="segment", text=text), self.loop)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=HERE / "profiles" / "kagawa.json")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18443)
    parser.add_argument("--cert", type=Path, default=Path("/data/ACE/ace_kagawa/local_avatar/tls/cert.pem"))
    parser.add_argument("--key", type=Path, default=Path("/data/ACE/ace_kagawa/local_avatar/tls/key.pem"))
    parser.add_argument("--ps-www", type=Path,
                        default=Path("/data/ACE/ace_kagawa/local_avatar/ps-signalling/SignallingWebServer/www"))
    parser.add_argument("--ps-signalling", default="ws://127.0.0.1:18080")
    parser.add_argument("--a2f", default="127.0.0.1:18200")
    parser.add_argument("--relay", default="127.0.0.1:18100")
    parser.add_argument("--relay-status", default="http://127.0.0.1:18102/status")
    parser.add_argument("--log-dir", type=Path, default=Path("/data/ACE/ace_kagawa/local_avatar/sessions"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    turns = Turns(args, load_profile(args.profile))
    app = create_app(turns)
    uvicorn.run(app, host=args.host, port=args.port, ssl_certfile=str(args.cert), ssl_keyfile=str(args.key),
                log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
