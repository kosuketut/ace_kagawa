#!/usr/bin/env python3
"""Make the avatar say a WAV: Audio2Face-3D animates it, mini_animgraph plays it.

Audio2Face-3D's bidirectional ``ProcessAudioStream`` answers with the same
``AnimationData`` messages the relay accepts, so its output is forwarded to the
relay's ``PushAnimationDataStream`` as it arrives.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import wave
from collections.abc import Iterator
from pathlib import Path

import grpc
import numpy as np

GEN_DIR = Path("/data/ACE/ace_kagawa/local_avatar/gen")
sys.path.insert(0, str(GEN_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import nvidia_ace.a2f.v1_pb2 as a2f_pb  # noqa: E402
import nvidia_ace.animation_data.v1_pb2 as anim_pb  # noqa: E402
import nvidia_ace.animation_id.v1_pb2 as ids_pb  # noqa: E402
import nvidia_ace.audio.v1_pb2 as audio_pb  # noqa: E402
import nvidia_ace.controller.v1_pb2 as ctrl_pb  # noqa: E402
import nvidia_ace.services.a2f_controller.v1_pb2_grpc as a2f_grpc  # noqa: E402
import nvidia_ace.services.animation_data.v1_pb2_grpc as relay_grpc  # noqa: E402
from speech_buffer import resample_int16  # noqa: E402

LOG = logging.getLogger("a2f_speak")
A2F_RATE = 16000
CHUNK_SECONDS = 0.1


def read_wav_16k(path: Path) -> bytes:
    with wave.open(str(path)) as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: PCM16 required")
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        if w.getnchannels() > 1:
            pcm = pcm.reshape(-1, w.getnchannels()).mean(axis=1).astype(np.int16)
        return resample_int16(pcm, w.getframerate(), A2F_RATE).tobytes()


def audio_requests(pcm16: bytes) -> Iterator[ctrl_pb.AudioStream]:
    yield ctrl_pb.AudioStream(
        audio_stream_header=ctrl_pb.AudioStreamHeader(
            audio_header=audio_pb.AudioHeader(
                audio_format=audio_pb.AudioHeader.AUDIO_FORMAT_PCM,
                channel_count=1, samples_per_second=A2F_RATE, bits_per_sample=16,
            )
        )
    )
    step = int(A2F_RATE * CHUNK_SECONDS) * 2
    for i in range(0, len(pcm16), step):
        yield ctrl_pb.AudioStream(audio_with_emotion=a2f_pb.AudioWithEmotion(audio_buffer=pcm16[i:i + step]))
    yield ctrl_pb.AudioStream(end_of_audio=ctrl_pb.AudioStream.EndOfAudio())


def relay_messages(a2f_responses, stats: dict) -> Iterator[anim_pb.AnimationDataStream]:
    for msg in a2f_responses:
        part = msg.WhichOneof("stream_part")
        if part == "animation_data_stream_header":
            h = msg.animation_data_stream_header
            stats["blend_shapes"] = len(h.skel_animation_header.blend_shapes)
            stats["first_header_s"] = time.monotonic() - stats["t0"]
            yield anim_pb.AnimationDataStream(
                animation_data_stream_header=anim_pb.AnimationDataStreamHeader(
                    animation_ids=ids_pb.AnimationIds(request_id="a2f", stream_id="a2f", target_object_id="avatar"),
                    audio_header=h.audio_header,
                    skel_animation_header=h.skel_animation_header,
                    start_time_code_since_epoch=h.start_time_code_since_epoch,
                )
            )
        elif part == "animation_data":
            stats["frames"] += len(msg.animation_data.skel_animation.blend_shape_weights)
            stats["audio_bytes"] += len(msg.animation_data.audio.audio_buffer)
            yield anim_pb.AnimationDataStream(animation_data=msg.animation_data)
        elif part == "event":
            if msg.event.event_type == ctrl_pb.END_OF_A2F_AUDIO_PROCESSING:
                return
        elif part == "status" and msg.status.code != 0:
            raise RuntimeError(f"A2F status {msg.status.code}: {msg.status.message}")


def speak(pcm16: bytes, a2f_target: str, relay_target: str) -> dict:
    stats = {"t0": time.monotonic(), "frames": 0, "audio_bytes": 0}
    a2f = a2f_grpc.A2FControllerServiceStub(grpc.insecure_channel(a2f_target))
    relay = relay_grpc.AnimationDataServiceStub(grpc.insecure_channel(relay_target))
    responses = a2f.ProcessAudioStream(audio_requests(pcm16))
    status = relay.PushAnimationDataStream(relay_messages(responses, stats))
    stats["seconds"] = time.monotonic() - stats.pop("t0")
    stats["relay_status"] = status.code
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", type=Path)
    parser.add_argument("--a2f", default="127.0.0.1:18200")
    parser.add_argument("--relay", default="127.0.0.1:18100")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    pcm = read_wav_16k(args.wav)
    stats = speak(pcm, args.a2f, args.relay)
    stats["audio_in_s"] = len(pcm) / 2 / A2F_RATE
    print(stats)
    return 0 if stats["relay_status"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
