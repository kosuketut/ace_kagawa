from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, TextIO

DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clone an authorized Japanese voice with Qwen3-TTS."
    )
    parser.add_argument("--reference-audio", type=Path, required=True)
    parser.add_argument("--reference-text", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--attn-implementation",
        choices=("sdpa", "eager", "flash_attention_2"),
        default="sdpa",
        help="Attention backend; sdpa avoids a flash-attn requirement.",
    )
    return parser


def _load_model(model_name: str, device: str, attn_implementation: str) -> Any:
    try:
        import torch
        from qwen_tts import Qwen3TTSModel
    except ImportError as exc:
        raise RuntimeError(
            "qwen-tts and its isolated dependencies are required; see tts/README.md"
        ) from exc

    return Qwen3TTSModel.from_pretrained(
        model_name,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
    )


def _audio_samples(audio: Any) -> int:
    shape = getattr(audio, "shape", None)
    if shape is not None:
        return int(shape[-1])
    return len(audio)


def _write_wav(output_path: Path, audio: Any, sample_rate: int) -> None:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("soundfile is required to write the Qwen WAV output") from exc

    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio, sample_rate)


def run(
    *,
    reference_audio: Path,
    reference_text: str,
    text: str,
    output: Path,
    model_name: str = DEFAULT_MODEL,
    device: str = "cuda:0",
    attn_implementation: str = "sdpa",
) -> dict[str, Any]:
    load_started = time.perf_counter()
    with redirect_stdout(sys.stderr):
        model = _load_model(model_name, device, attn_implementation)
    load_seconds = time.perf_counter() - load_started

    synthesis_started = time.perf_counter()
    with redirect_stdout(sys.stderr):
        wavs, sample_rate = model.generate_voice_clone(
            text=text,
            language="Japanese",
            ref_audio=str(reference_audio),
            ref_text=reference_text,
            x_vector_only_mode=False,
        )
    synthesis_seconds = time.perf_counter() - synthesis_started

    if not wavs:
        raise RuntimeError("Qwen3-TTS returned no audio")
    audio = wavs[0]
    sample_rate = int(sample_rate)
    _write_wav(output, audio, sample_rate)
    audio_seconds = _audio_samples(audio) / sample_rate

    return {
        "model": model_name,
        "output": str(output),
        "sample_rate": sample_rate,
        "load_seconds": round(load_seconds, 3),
        "synthesis_seconds": round(synthesis_seconds, 3),
        "audio_seconds": round(audio_seconds, 3),
        "RTF": round(synthesis_seconds / audio_seconds, 3) if audio_seconds else 0.0,
    }


def main(argv: list[str] | None = None, *, stdout: TextIO | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.reference_audio.is_file():
        parser.error(f"reference audio does not exist: {args.reference_audio}")

    metadata = run(
        reference_audio=args.reference_audio,
        reference_text=args.reference_text,
        text=args.text,
        output=args.output,
        model_name=args.model,
        device=args.device,
        attn_implementation=args.attn_implementation,
    )
    print(json.dumps(metadata, ensure_ascii=False), file=stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
