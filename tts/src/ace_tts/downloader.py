from __future__ import annotations

from pathlib import Path
import subprocess


def download_youtube_audio(
    url: str,
    *,
    output_dir: Path,
    output_name: str,
    start: float | None = None,
    duration: float | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{output_name}.wav"
    downloaded_name = f"{output_name}.download" if start is not None or duration is not None else output_name
    downloaded_path = output_dir / f"{downloaded_name}.wav"

    try:
        from yt_dlp import YoutubeDL
    except ImportError as exc:
        raise RuntimeError("yt-dlp is required to download YouTube audio") from exc

    options = {
        "format": "bestaudio/best",
        "outtmpl": str(output_dir / f"{downloaded_name}.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
        "noplaylist": True,
        "quiet": False,
    }

    with YoutubeDL(options) as ydl:
        ydl.download([url])

    if not downloaded_path.exists():
        raise RuntimeError(
            f"download finished but wav was not created: {downloaded_path}. "
            "Install ffmpeg and retry."
        )

    if downloaded_path != output_path:
        _trim_audio(downloaded_path, output_path, start=start, duration=duration)
        downloaded_path.unlink(missing_ok=True)

    return output_path


def _trim_audio(
    input_path: Path,
    output_path: Path,
    *,
    start: float | None,
    duration: float | None,
) -> None:
    command = ["ffmpeg", "-y"]
    if start is not None:
        command.extend(["-ss", str(start)])
    command.extend(["-i", str(input_path)])
    if duration is not None:
        command.extend(["-t", str(duration)])
    command.extend(["-acodec", "pcm_s16le", "-ar", "24000", "-ac", "1", str(output_path)])
    subprocess.run(command, check=True)
