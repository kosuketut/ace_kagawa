#!/usr/bin/env python3
"""Isolated Irodori CUDA / speech.cpp Vulkan benchmark; never changes services.

Run with Irodori-TTS/.venv/bin/python. Outputs (including audio) belong under /data.
GPU telemetry is device-wide: run without other workloads on the selected GPU.
"""
import argparse
import base64
from dataclasses import replace
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import wave

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "infra/tts"))
sys.path.insert(0, str(ROOT / "Irodori-TTS"))

MODEL_IDS = {
    "small": "Aratako/Irodori-TTS-v4.1-Small",
    "mf": "Aratako/Irodori-TTS-v4.1-Small-MF",
    "large": "Aratako/Irodori-TTS-v4-Large",
    "v3": "Aratako/Irodori-TTS-500M-v3",
}

CASES = [
    ("short", "少々お待ちください。", 2.4),
    ("profile", "私の専門分野は材料強度学、複合材料、高信頼性材料です。", 6.4),
    ("access", "八王子キャンパスへは八王子駅から無料スクールバスで約10分です。", 6.4),
    ("numbers", "オープンキャンパスは2026年8月1日の10:00から16:00までです。", 10.4),
    ("long", "東京工科大学では、専門分野の知識だけでなく、実際の社会で課題を見つけて解決する力を身につけるため、実践的な学びを大切にしています。", 14.4),
]


def read_json_line(stream, output):
    try:
        for line in stream:
            output.put((time.perf_counter(), json.loads(line)))
    except Exception as exc:
        output.put((time.perf_counter(), {"type": "reader_error", "error": str(exc)}))
    finally:
        output.put((time.perf_counter(), {"type": "eof"}))


class Telemetry:
    def __init__(self, gpu, path):
        self.samples = []
        self.file = path.open("w")
        self.proc = subprocess.Popen([
            "nvidia-smi", "-i", str(gpu),
            "--query-gpu=utilization.gpu,memory.used,power.draw",
            "--format=csv,noheader,nounits", "-lms", "100",
        ], stdout=subprocess.PIPE, text=True)
        self.thread = threading.Thread(target=self.read, daemon=True)
        self.thread.start()

    def read(self):
        for line in self.proc.stdout:
            now = time.perf_counter()
            values = [float(x.strip()) for x in line.split(",")]
            sample = dict(t=now, gpu_pct=values[0], memory_mib=values[1], power_w=values[2])
            self.samples.append(sample)
            self.file.write(json.dumps(sample) + "\n")
            self.file.flush()

    def close(self):
        self.proc.terminate()
        self.proc.wait(timeout=10)
        self.thread.join(timeout=10)
        self.file.close()


def metrics(samples, start, end):
    selected = [s for s in samples if start <= s["t"] <= end]
    result = {"telemetry_samples": len(selected)}
    for key in ("gpu_pct", "memory_mib", "power_w"):
        vals = [s[key] for s in selected]
        result[key + "_mean"] = sum(vals) / len(vals) if vals else None
        result[key + "_peak"] = max(vals) if vals else None
    return result


def cpu_seconds(pid):
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")


def write_wav(path, pcm, sample_rate):
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["python", "cpp"], required=True)
    p.add_argument("--model", choices=MODEL_IDS, required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--speech", type=Path)
    p.add_argument("--gguf", type=Path)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--voice-file", type=Path)
    p.add_argument("--max-ref-seconds", type=float,
                   help="Python reference cap; omit for the checkpoint default the service uses (120 s for v4.1)")
    p.add_argument("--gpu", type=int, default=0, help="nvidia-smi physical GPU index for telemetry")
    p.add_argument("--cpp-device", help="speech devices name; Vulkan indices need not match CUDA/nvidia-smi")
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--warmup-rounds", type=int, default=2)
    p.add_argument("--fixed", action="store_true", help="Same fixed duration, no tail trimming, for implementation comparison")
    p.add_argument("--short-cap", action="store_true",
                   help="cpp only: give short texts the service's mora duration cap as a fixed length")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        s = [dict(t=1, gpu_pct=20, memory_mib=100, power_w=40), dict(t=2, gpu_pct=80, memory_mib=200, power_w=60)]
        assert metrics(s, 1, 2)["gpu_pct_mean"] == 50
        assert metrics(s, 1.5, 2)["memory_mib_peak"] == 200
        assert metrics(s, 3, 4)["gpu_pct_mean"] is None
        assert cpu_seconds(os.getpid()) >= 0
        assert MODEL_IDS["large"] == "Aratako/Irodori-TTS-v4-Large"
        assert MODEL_IDS["v3"] == "Aratako/Irodori-TTS-500M-v3"
        print("self-test passed")
        return
    if args.steps < 1 or args.rounds < 1 or args.warmup_rounds < 1 or not args.reference.is_file():
        p.error("positive steps/rounds/warmup-rounds and an existing reference WAV required")
    if args.backend == "cpp" and args.model == "v3":
        p.error("speech.cpp v0.8.7 supports Irodori v4/v4.1, not 500M-v3")
    if args.backend == "cpp" and (not args.cpp_device or not all(x and x.is_file() for x in (args.speech, args.gguf, args.voice_file))):
        p.error("cpp requires --cpp-device and existing --speech, --gguf, --voice-file")
    if args.short_cap and (args.backend != "cpp" or args.fixed):
        p.error("--short-cap is for cpp without --fixed")
    args.output.mkdir(parents=True, exist_ok=False)
    from irodori_server import IrodoriSettings, IrodoriSynthesizer, normalize_speech_input
    settings = replace(IrodoriSettings(),
        hf_checkpoint=MODEL_IDS[args.model],
        reference_wav=args.reference, num_steps=args.steps, max_ref_seconds=args.max_ref_seconds,
        short_cache_enabled=False,
    )
    synth = IrodoriSynthesizer(settings)
    telemetry = Telemetry(args.gpu, args.output / "gpu.jsonl")
    worker = None
    log_file = None
    try:
        load_start = time.perf_counter()
        if args.backend == "python":
            import torch
            from irodori_tts.inference_runtime import SamplingRequest
            torch.set_num_threads(8)
            runtime = synth.load_runtime()
            ready = {"watermark": runtime.watermarker.ready, "default_max_ref_seconds": runtime.default_max_ref_seconds}
            pid = os.getpid()
        else:
            log_file = (args.output / "worker.stderr").open("w")
            worker = subprocess.Popen([
                str(args.speech), "worker", str(args.gguf), "--device", args.cpp_device,
                "--threads", "8", "--add-voice", f"kagawa={args.voice_file}",
            ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log_file, text=True, bufsize=1)
            messages = queue.Queue()
            threading.Thread(target=read_json_line, args=(worker.stdout, messages), daemon=True).start()
            _, ready = messages.get(timeout=300)
            if ready.get("type") != "ready":
                raise RuntimeError(ready)
            pid = worker.pid
        load_s = time.perf_counter() - load_start
        metadata = {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    "load_ready_s": load_s, "ready": ready, "cases": CASES,
                    "python": sys.version, "pid": pid}
        (args.output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
        # Warm every shape twice: a single prompt misses first-use kernel/allocator costs.
        requests = [(f"warmup-{name}-{i}", text, seconds, 0)
                    for i in range(args.warmup_rounds) for name, text, seconds in CASES]
        requests += [(name, text, seconds, seed) for seed in range(1, args.rounds + 1) for name, text, seconds in CASES]
        with (args.output / "requests.jsonl").open("w") as rows:
            for name, text, seconds, seed in requests:
                normalized = normalize_speech_input(text)
                start_cpu = cpu_seconds(pid)
                start = time.perf_counter()
                if args.backend == "python":
                    torch.cuda.synchronize()
                    req = SamplingRequest(text=normalized, ref_wav=str(args.reference), max_ref_seconds=args.max_ref_seconds,
                                          num_steps=args.steps, seed=seed,
                                          max_seconds=synth._max_seconds_for(normalized))
                    if args.fixed:
                        req.seconds, req.max_seconds, req.trim_tail = seconds, 30.0, False
                    result = runtime.synthesize(req, log_fn=None)
                    raw_audio_s = result.audio.shape[-1] / result.sample_rate
                    pcm = synth._speech_pcm16(result.audio, source_sample_rate=result.sample_rate, normalized_text=normalized)
                    torch.cuda.synchronize()
                    end = first = time.perf_counter()
                    sample_rate = settings.response_sample_rate_hz
                    extra = {"stages_s": dict(result.stage_timings), "raw_audio_s": raw_audio_s,
                             "messages": result.messages}
                else:
                    request = dict(type="synthesize", id=f"{name}-{seed}", text=normalized, voice="kagawa",
                                   seed=seed, steps=args.steps)
                    # Official SamplingRequest defaults to a linear schedule (Sway disabled).
                    if args.model != "mf":
                        request["sway_coeff"] = 0.0
                    if args.fixed:
                        request.update(seconds=seconds, keep_tail=True)
                    elif args.short_cap and len(normalized) <= settings.short_duration_max_chars:
                        # speech.cpp has no max_seconds for Irodori; its predicted length of
                        # short phrases exceeds the cap (babble), so a wrapper fixes the length.
                        request["seconds"] = synth._max_seconds_for(normalized)
                    worker.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
                    worker.stdin.flush()
                    parts, first = [], None
                    while True:
                        received, message = messages.get(timeout=300)
                        kind = message["type"]
                        if kind == "chunk":
                            data = base64.b64decode(message["pcm"], validate=True)
                            if data and first is None:
                                first = received
                            parts.append(data)
                        elif kind == "end":
                            end = received
                            break
                        elif kind != "progress":
                            raise RuntimeError(message)
                    pcm, sample_rate = b"".join(parts), 48000
                    if len(pcm) // 2 != message["samples"] or first is None:
                        raise RuntimeError("invalid or empty worker audio")
                    extra = {"stop": message.get("stop"), "raw_audio_s": len(pcm) / 2 / sample_rate}
                end_cpu = cpu_seconds(pid)
                if not pcm or len(pcm) % 2:
                    raise RuntimeError("invalid PCM output")
                audio_s = len(pcm) / 2 / sample_rate
                row = dict(case=name, seed=seed, text=text, normalized=normalized,
                           start=start, end=end, latency_s=end-start, ttfa_s=first-start,
                           audio_s=audio_s, rtfx=audio_s/(end-start),
                           cpu_core_pct=100*(end_cpu-start_cpu)/(end-start), **extra,
                           **metrics(telemetry.samples, start, end))
                # A request shorter than the 100 ms sampling can have no sample; judge only sampled ones.
                if row["memory_mib_peak"] is not None and row["memory_mib_peak"] < 500:
                    raise RuntimeError("No model memory on telemetry GPU: verify CUDA/Vulkan device mapping")
                write_wav(args.output / f"{name}-{seed}.wav", pcm, sample_rate)
                rows.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows.flush()
                print(json.dumps(row, ensure_ascii=False), flush=True)
    finally:
        if worker is not None:
            worker.stdin.close()
            try:
                worker.wait(timeout=20)
            except subprocess.TimeoutExpired:
                worker.terminate()
                worker.wait(timeout=10)
        if log_file:
            log_file.close()
        telemetry.close()


if __name__ == "__main__":
    main()
