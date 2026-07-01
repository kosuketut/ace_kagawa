from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, TextIO

from ace_tts.downloader import download_youtube_audio
from ace_tts.sarashina import synthesize_with_sarashina

DownloadAudio = Callable[[str], Path]


def main(
    argv: list[str] | None = None,
    *,
    download_audio: Callable[..., Path] = download_youtube_audio,
    synthesize: Callable[..., Path] = synthesize_with_sarashina,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "download-reference":
        reference_audio = download_audio(
            args.url,
            output_dir=args.output_dir,
            output_name=args.output_name,
            start=args.start,
            duration=args.duration,
        )
        print(reference_audio, file=stdout)
        return 0

    if args.command == "synthesize":
        return _run_synthesize(args, synthesize=synthesize, stdout=stdout, stderr=stderr)

    if args.command == "clone-from-youtube":
        reference_audio = download_audio(
            args.url,
            output_dir=args.work_dir,
            output_name="reference",
            start=args.start,
            duration=args.duration,
        )
        output_path = synthesize(
            reference_audio=reference_audio,
            reference_text=args.reference_text,
            text=args.text,
            output_path=args.output,
            model_name=args.model_name,
        )
        print(output_path, file=stdout)
        return 0

    parser.print_help(stderr)
    return 2


def _run_synthesize(
    args: argparse.Namespace,
    *,
    synthesize: Callable[..., Path],
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    if not args.reference_audio.exists():
        print(f"reference audio does not exist: {args.reference_audio}", file=stderr)
        return 2

    output_path = synthesize(
        reference_audio=args.reference_audio,
        reference_text=args.reference_text,
        text=args.text,
        output_path=args.output,
        model_name=args.model_name,
    )
    print(output_path, file=stdout)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ace-tts",
        description="Create speech with sarashina2.2-tts from authorized reference audio.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser(
        "download-reference",
        help="Download authorized YouTube audio as a wav reference file.",
    )
    download.add_argument("url", help="YouTube URL to download.")
    download.add_argument("--output-dir", type=Path, default=Path("data"))
    download.add_argument("--output-name", default="reference")
    _add_clip_args(download)

    synthesize = subparsers.add_parser(
        "synthesize",
        help="Generate speech from an existing reference wav.",
    )
    synthesize.add_argument(
        "--reference-audio",
        type=Path,
        required=True,
        help="Path to the authorized reference wav.",
    )
    _add_synthesis_args(synthesize)

    clone = subparsers.add_parser(
        "clone-from-youtube",
        help="Download authorized YouTube audio and use it as the voice reference.",
    )
    clone.add_argument("url", help="YouTube URL to download.")
    clone.add_argument("--work-dir", type=Path, default=Path("data"))
    _add_clip_args(clone)
    _add_synthesis_args(clone)

    return parser


def _add_synthesis_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--reference-text",
        required=True,
        help="Transcript matching the reference audio.",
    )
    parser.add_argument("--text", required=True, help="Text to synthesize.")
    parser.add_argument("--output", type=Path, default=Path("output/cloned.wav"))
    parser.add_argument("--model-name", default="sbintuitions/sarashina2.2-tts")


def _add_clip_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--start",
        type=float,
        default=None,
        help="Start time in seconds for the reference clip.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Reference clip duration in seconds.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
