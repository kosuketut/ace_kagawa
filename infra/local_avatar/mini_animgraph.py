#!/usr/bin/env python3
"""Minimal stand-in for the Animation Graph microservice.

The Tokkio Unreal renderer pulls face animation and audio with
``AnimationDataService.PullAnimationDataStream``. This server answers that
pull with one long-lived stream: a header, then frames at a fixed rate.

Speech enters with ``PushAnimationDataStream`` in the Audio2Face-3D output
format (header, then blend shape frames and audio chunks). Utterances play in
the order they were pushed and can start before the push has finished.
Between utterances the stream carries neutral weights and no audio.
"""

from __future__ import annotations

import argparse
import http.server
import json
import logging
import sys
import threading
import time
import wave
from collections import deque
from concurrent import futures
from pathlib import Path

import grpc
import numpy as np

GEN_DIR = Path("/data/ACE/ace_kagawa/local_avatar/gen")
sys.path.insert(0, str(GEN_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import nvidia_ace.animation_data.v1_pb2 as anim_pb  # noqa: E402
import nvidia_ace.animation_id.v1_pb2 as ids_pb  # noqa: E402
import nvidia_ace.audio.v1_pb2 as audio_pb  # noqa: E402
import nvidia_ace.services.animation_data.v1_pb2_grpc as svc_grpc  # noqa: E402
import nvidia_ace.status.v1_pb2 as status_pb  # noqa: E402
from speech_buffer import SpeechBuffer  # noqa: E402

LOG = logging.getLogger("mini_animgraph")

# Names emitted by Audio2Face-3D 1.3 (52 ARKit + head + 16 tongue).
A2F_BLENDSHAPES = [
    "EyeBlinkLeft", "EyeLookDownLeft", "EyeLookInLeft", "EyeLookOutLeft", "EyeLookUpLeft",
    "EyeSquintLeft", "EyeWideLeft", "EyeBlinkRight", "EyeLookDownRight", "EyeLookInRight",
    "EyeLookOutRight", "EyeLookUpRight", "EyeSquintRight", "EyeWideRight", "JawForward",
    "JawLeft", "JawRight", "JawOpen", "MouthClose", "MouthFunnel", "MouthPucker", "MouthLeft",
    "MouthRight", "MouthSmileLeft", "MouthSmileRight", "MouthFrownLeft", "MouthFrownRight",
    "MouthDimpleLeft", "MouthDimpleRight", "MouthStretchLeft", "MouthStretchRight",
    "MouthRollLower", "MouthRollUpper", "MouthShrugLower", "MouthShrugUpper", "MouthPressLeft",
    "MouthPressRight", "MouthLowerDownLeft", "MouthLowerDownRight", "MouthUpperUpLeft",
    "MouthUpperUpRight", "BrowDownLeft", "BrowDownRight", "BrowInnerUp", "BrowOuterUpLeft",
    "BrowOuterUpRight", "CheekPuff", "CheekSquintLeft", "CheekSquintRight", "NoseSneerLeft",
    "NoseSneerRight", "TongueOut", "HeadRoll", "HeadPitch", "HeadYaw", "TongueTipUp",
    "TongueTipDown", "TongueTipLeft", "TongueTipRight", "TongueRollUp", "TongueRollDown",
    "TongueRollLeft", "TongueRollRight", "TongueUp", "TongueDown", "TongueLeft", "TongueRight",
    "TongueIn", "TongueStretch", "TongueWide", "TongueNarrow",
]

DUMMY_JOINT = "mini_animgraph_unused"
ZERO3 = anim_pb.Float3(x=0.0, y=0.0, z=0.0)
ONE3 = anim_pb.Float3(x=1.0, y=1.0, z=1.0)
IDENTITY_QUAT = anim_pb.QuatF(real=1.0, i=0.0, j=0.0, k=0.0)
EMPTY_AUDIO = np.zeros(0, dtype=np.int16)


def load_demo_clip(frames_json: Path, wav_path: Path, fps: int, sample_rate: int) -> SpeechBuffer:
    data = json.loads(frames_json.read_text())
    with wave.open(str(wav_path)) as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError("demo WAV must be mono PCM16")
        buf = SpeechBuffer(A2F_BLENDSHAPES, fps, sample_rate)
        buf.set_source(data["blendshape_names"], w.getframerate())
        buf.add_audio(w.readframes(w.getnframes()))
    for frame in data["frames"]:
        buf.add_weights(frame["weights"])
    buf.finish()
    return buf


class SpeechQueue:
    def __init__(self) -> None:
        self._items: deque[SpeechBuffer] = deque()
        self._lock = threading.Lock()

    def put(self, item: SpeechBuffer) -> None:
        with self._lock:
            self._items.append(item)

    def peek(self) -> SpeechBuffer | None:
        with self._lock:
            return self._items[0] if self._items else None

    def pop(self) -> None:
        with self._lock:
            self._items.popleft()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


class AnimationDataService(svc_grpc.AnimationDataServiceServicer):
    def __init__(self, queue: SpeechQueue, fps: int, sample_rate: int, prebuffer_frames: int) -> None:
        self.queue = queue
        self.fps = fps
        self.sample_rate = sample_rate
        self.prebuffer_frames = prebuffer_frames
        self.speaking = False
        self.clients = 0

    def status(self) -> dict:
        return {"speaking": self.speaking, "queued": len(self.queue), "clients": self.clients}

    def PushAnimationDataStream(self, request_iterator, context):
        buf = SpeechBuffer(A2F_BLENDSHAPES, self.fps, self.sample_rate)
        queued = False
        frames = 0
        try:
            for msg in request_iterator:
                part = msg.WhichOneof("stream_part")
                if part == "animation_data_stream_header":
                    header = msg.animation_data_stream_header
                    rate = header.audio_header.samples_per_second or self.sample_rate
                    if header.audio_header.bits_per_sample not in (0, 16) or header.audio_header.channel_count > 1:
                        return status_pb.Status(code=status_pb.Status.ERROR,
                                                message="audio must be mono PCM16")
                    buf.set_source(list(header.skel_animation_header.blend_shapes), rate)
                    if not queued:
                        self.queue.put(buf)
                        queued = True
                elif part == "animation_data":
                    if not queued:
                        return status_pb.Status(code=status_pb.Status.ERROR,
                                                message="header must come first")
                    data = msg.animation_data
                    for row in data.skel_animation.blend_shape_weights:
                        buf.add_weights(row.values)
                        frames += 1
                    if data.audio.audio_buffer:
                        buf.add_audio(data.audio.audio_buffer)
        finally:
            buf.finish()
        LOG.info("push: %d frames, %.2f s audio", frames, buf.audio_samples() / self.sample_rate)
        return status_pb.Status(code=status_pb.Status.SUCCESS, message="queued")

    def PullAnimationDataStream(self, request, context):
        LOG.info("pull: stream_id=%s", request.stream_id)
        ids = ids_pb.AnimationIds(
            request_id=request.request_id or "local",
            stream_id=request.stream_id,
            target_object_id=request.target_object_id or "avatar",
        )
        yield anim_pb.AnimationDataStream(
            animation_data_stream_header=anim_pb.AnimationDataStreamHeader(
                animation_ids=ids,
                source_service_id="mini-animgraph",
                audio_header=audio_pb.AudioHeader(
                    audio_format=audio_pb.AudioHeader.AUDIO_FORMAT_PCM,
                    channel_count=1,
                    samples_per_second=self.sample_rate,
                    bits_per_sample=16,
                ),
                skel_animation_header=anim_pb.SkelAnimationHeader(
                    blend_shapes=A2F_BLENDSHAPES, joints=[DUMMY_JOINT]
                ),
                start_time_code_since_epoch=time.time(),
            )
        )
        self.clients += 1
        try:
            yield from self._frames(context)
        finally:
            self.clients -= 1
        LOG.info("pull ended: stream_id=%s", request.stream_id)

    def _next_frame(self, state: dict) -> tuple[np.ndarray, np.ndarray]:
        neutral = state["neutral"]
        current: SpeechBuffer | None = state["current"]
        if current is None:
            candidate = self.queue.peek()
            if candidate is None or not candidate.ready_to_start(self.prebuffer_frames):
                return neutral, EMPTY_AUDIO
            current, state["current"], state["k"] = candidate, candidate, 0
            self.speaking = True
            LOG.info("speech start")
        k = state["k"]
        if current.available(k):
            weights, audio = current.frame(k)
            state["k"] = k + 1
            state["last"] = weights
            return weights, audio
        if current.exhausted(k):
            self.queue.pop()
            state["current"] = None
            self.speaking = False
            LOG.info("speech end: %d frames", k)
            return self._next_frame(state)
        # Underrun: hold the face, send no audio, and retry this frame next tick.
        state["underruns"] += 1
        return state["last"], EMPTY_AUDIO

    def _frames(self, context):
        frame_dt = 1.0 / self.fps
        neutral = np.zeros(len(A2F_BLENDSHAPES), dtype=np.float32)
        state = {"current": None, "k": 0, "last": neutral, "neutral": neutral, "underruns": 0}
        frame_no = 0
        start = time.monotonic()
        # Time codes are wall-clock seconds so they keep increasing across
        # relay restarts: the renderer keeps its audio sample count per stream.
        epoch0 = time.time()
        while context.is_active():
            # Idle frames carry no audio: the plugin pads silence up to the
            # animation time code itself. Sending silence instead lets the
            # backlog buffered before the client starts reading arrive as one
            # oversized audio chunk, which shifts audio against animation.
            weights, chunk = self._next_frame(state)
            t = epoch0 + frame_no * frame_dt
            # The ACE plugin's animgraph client reads one time code from each of
            # blend shapes, audio, translations, rotations and scales: it
            # crashes when a list is empty and drops frames whose joint values
            # are empty. UE only uses blend shapes and audio from this stream,
            # so one identity transform for a dummy joint satisfies it.
            yield anim_pb.AnimationDataStream(
                animation_data=anim_pb.AnimationData(
                    skel_animation=anim_pb.SkelAnimation(
                        blend_shape_weights=[
                            anim_pb.FloatArrayWithTimeCode(time_code=t, values=weights.tolist())
                        ],
                        translations=[anim_pb.Float3ArrayWithTimeCode(time_code=t, values=[ZERO3])],
                        rotations=[anim_pb.QuatFArrayWithTimeCode(time_code=t, values=[IDENTITY_QUAT])],
                        scales=[anim_pb.Float3ArrayWithTimeCode(time_code=t, values=[ONE3])],
                    ),
                    audio=anim_pb.AudioWithTimeCode(time_code=t, audio_buffer=chunk.tobytes()),
                )
            )
            frame_no += 1
            if state["underruns"] and frame_no % (self.fps * 10) == 0:
                LOG.warning("underrun frames so far: %d", state["underruns"])
            sleep = start + frame_no * frame_dt - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)


def start_status_server(service: AnimationDataService, host: str, port: int) -> None:
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/status":
                self.send_error(404)
                return
            body = json.dumps(service.status()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args) -> None:
            pass

    httpd = http.server.ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True, name="status-http").start()
    LOG.info("status on http://%s:%d/status", host, port)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18100)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--status-port", type=int, default=18102,
                        help="HTTP GET /status on the same host: is the avatar speaking")
    parser.add_argument("--prebuffer-frames", type=int, default=6,
                        help="frames to buffer before a pushed utterance starts playing")
    parser.add_argument("--demo-frames", type=Path, help="A2F frames JSON for the demo loop")
    parser.add_argument("--demo-wav", type=Path, help="WAV matching --demo-frames")
    parser.add_argument("--demo-interval", type=float, default=10.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    queue = SpeechQueue()
    # SO_REUSEPORT would let a stale instance keep serving the same port.
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=8), options=[("grpc.so_reuseport", 0)]
    )
    service = AnimationDataService(queue, args.fps, args.sample_rate, args.prebuffer_frames)
    svc_grpc.add_AnimationDataServiceServicer_to_server(service, server)
    server.add_insecure_port(f"{args.host}:{args.port}")
    server.start()
    LOG.info("listening on %s:%d", args.host, args.port)
    start_status_server(service, args.host, args.status_port)

    if args.demo_frames and args.demo_wav:
        def demo_loop() -> None:
            while True:
                time.sleep(args.demo_interval)
                # Keep at most one pending clip so the demo never builds a backlog.
                if len(queue) == 0:
                    queue.put(load_demo_clip(args.demo_frames, args.demo_wav, args.fps, args.sample_rate))

        threading.Thread(target=demo_loop, daemon=True).start()

    server.wait_for_termination()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
