#!/usr/bin/env python3
"""Generate the symposium opening directly with Qwen3-TTS voice cloning."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "data/models/Qwen3-TTS-12Hz-1.7B-Base"
DEFAULT_REFERENCE_AUDIO = ROOT / "voice/kagawa_EN.MP3"
DEFAULT_REFERENCE_TEXT = ROOT / "voice/qwen_tts_voice_clone_english_script.md"
DEFAULT_SCRIPT = ROOT / "voice/symposium_opening_script.md"
DEFAULT_OUTPUT = ROOT / "data/output/qwen/symposium_opening_kagawa_en.wav"


def markdown_body(path: Path) -> str:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    if lines and lines[0].lstrip().startswith("#"):
        lines = lines[1:]
    return "\n".join(lines).strip()


def sentence_plan(text: str) -> list[tuple[str, bool]]:
    plan: list[tuple[str, bool]] = []
    for paragraph in re.split(r"\n\s*\n", text.strip()):
        sentences = re.split(r"(?<=[.!?])\s+", " ".join(paragraph.splitlines()).strip())
        sentences = [sentence for sentence in sentences if sentence]
        plan.extend((sentence, index == len(sentences) - 1) for index, sentence in enumerate(sentences))
    return plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--reference-audio", type=Path, default=DEFAULT_REFERENCE_AUDIO)
    parser.add_argument("--reference-text", type=Path, default=DEFAULT_REFERENCE_TEXT)
    parser.add_argument("--script", type=Path, default=DEFAULT_SCRIPT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--sentence-gap", type=float, default=0.3)
    parser.add_argument("--paragraph-gap", type=float, default=0.65)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script = args.script.read_text(encoding="utf-8").strip()
    plan = sentence_plan(script)
    if not plan:
        raise ValueError("script is empty")
    if args.plan_only:
        print(json.dumps([text for text, _ in plan], ensure_ascii=False, indent=2))
        return 0

    import torch
    from qwen_tts import Qwen3TTSModel

    started = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(
        str(args.model),
        device_map=args.device,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    reference_audio, reference_sample_rate = sf.read(args.reference_audio, dtype="float32")
    if reference_audio.ndim == 2:
        reference_audio = reference_audio.mean(axis=1)
    prompt = model.create_voice_clone_prompt(
        ref_audio=(reference_audio, reference_sample_rate),
        ref_text=markdown_body(args.reference_text),
        x_vector_only_mode=False,
    )

    frames: list[np.ndarray] = []
    sample_rate: int | None = None
    for index, (sentence, paragraph_end) in enumerate(plan, start=1):
        wavs, generated_sample_rate = model.generate_voice_clone(
            text=sentence,
            language="English",
            voice_clone_prompt=prompt,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        if not wavs:
            raise RuntimeError(f"Qwen3-TTS returned no audio for sentence {index}")
        if sample_rate is None:
            sample_rate = int(generated_sample_rate)
        elif sample_rate != int(generated_sample_rate):
            raise RuntimeError("Qwen3-TTS returned inconsistent sample rates")
        audio = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
        frames.append(audio)
        if index < len(plan):
            gap_seconds = args.paragraph_gap if paragraph_end else args.sentence_gap
            frames.append(np.zeros(round(sample_rate * gap_seconds), dtype=np.float32))
        print(f"[{index}/{len(plan)}] {sentence}", flush=True)

    assert sample_rate is not None
    combined = np.concatenate(frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_name(f"{args.output.stem}.tmp.wav")
    sf.write(temporary_output, combined, sample_rate, subtype="PCM_16")
    temporary_output.replace(args.output)
    args.output.with_suffix(".txt").write_text(script + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "output": str(args.output),
                "sample_rate": sample_rate,
                "sentences": len(plan),
                "duration_seconds": round(len(combined) / sample_rate, 3),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
