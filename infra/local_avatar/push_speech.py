#!/usr/bin/env python3
"""Push one utterance to mini_animgraph.py in the Audio2Face-3D output format.

Replays saved A2F frames (JSON with ``blendshape_names`` and ``frames``) with
the matching WAV, a few frames at a time, the way Audio2Face-3D streams them.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from pathlib import Path

import grpc

GEN_DIR = Path("/data/ACE/ace_kagawa/local_avatar/gen")
sys.path.insert(0, str(GEN_DIR))

import nvidia_ace.animation_data.v1_pb2 as anim_pb  # noqa: E402
import nvidia_ace.animation_id.v1_pb2 as ids_pb  # noqa: E402
import nvidia_ace.audio.v1_pb2 as audio_pb  # noqa: E402
import nvidia_ace.services.animation_data.v1_pb2_grpc as svc_grpc  # noqa: E402


def replay_messages(frames_json: Path, wav_path: Path, chunk_frames: int, chunk_delay: float):
    data = json.loads(frames_json.read_text())
    with wave.open(str(wav_path)) as w:
        rate = w.getframerate()
        pcm = w.readframes(w.getnframes())
    frames = data["frames"]
    fps = round(1.0 / (frames[1]["time"] - frames[0]["time"])) if len(frames) > 1 else 30
    yield anim_pb.AnimationDataStream(
        animation_data_stream_header=anim_pb.AnimationDataStreamHeader(
            animation_ids=ids_pb.AnimationIds(request_id="push", stream_id="push", target_object_id="avatar"),
            audio_header=audio_pb.AudioHeader(
                audio_format=audio_pb.AudioHeader.AUDIO_FORMAT_PCM,
                channel_count=1, samples_per_second=rate, bits_per_sample=16,
            ),
            skel_animation_header=anim_pb.SkelAnimationHeader(blend_shapes=data["blendshape_names"]),
            start_time_code_since_epoch=time.time(),
        )
    )
    bytes_per_frame = 2 * rate / fps
    for i in range(0, len(frames), chunk_frames):
        part = frames[i:i + chunk_frames]
        a0 = int(round(i * bytes_per_frame / 2)) * 2
        a1 = int(round((i + len(part)) * bytes_per_frame / 2)) * 2
        if i + chunk_frames >= len(frames):
            a1 = len(pcm)
        yield anim_pb.AnimationDataStream(
            animation_data=anim_pb.AnimationData(
                skel_animation=anim_pb.SkelAnimation(
                    blend_shape_weights=[
                        anim_pb.FloatArrayWithTimeCode(time_code=f["time"], values=f["weights"]) for f in part
                    ]
                ),
                audio=anim_pb.AudioWithTimeCode(time_code=part[0]["time"], audio_buffer=pcm[a0:a1]),
            )
        )
        time.sleep(chunk_delay)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="127.0.0.1:18100")
    parser.add_argument("--frames-json", type=Path, required=True)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--chunk-frames", type=int, default=10)
    parser.add_argument("--chunk-delay", type=float, default=0.05,
                        help="seconds between chunks (0.05 s per 10 frames is faster than real time)")
    args = parser.parse_args()
    stub = svc_grpc.AnimationDataServiceStub(grpc.insecure_channel(args.target))
    status = stub.PushAnimationDataStream(
        replay_messages(args.frames_json, args.wav, args.chunk_frames, args.chunk_delay)
    )
    print(f"status={status.code} message={status.message}")
    return 0 if status.code == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
