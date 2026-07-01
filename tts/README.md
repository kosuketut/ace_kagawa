# ACE TTS

Local workflow for creating speech with `sbintuitions/sarashina2.2-tts` from authorized reference audio.

## Requirements

- Permission from the speaker and rights holder for the reference audio.
- `uv`
- `ffmpeg`
- `yt-dlp`, installed automatically by `uv`
- The official `sarashina2.2-tts` package, installed from GitHub by `uv`

The Sarashina model is distributed under the Sarashina Model NonCommercial License Agreement. Generated audio includes the model's inaudible AI-generated audio watermark by default.

## Download Authorized YouTube Audio

```bash
uv run ace-tts download-reference \
  'https://www.youtube.com/watch?v=2Zb0AiL3IO0&list=PLlO7zjtaf1FSfF2HQ3RvNcpKKcjQki7IY&index=13' \
  --output-dir data \
  --output-name reference
```

This creates:

```text
data/reference.wav
```

For a short reference clip, pass start time and duration in seconds:

```bash
uv run ace-tts download-reference \
  'https://www.youtube.com/watch?v=2Zb0AiL3IO0&list=PLlO7zjtaf1FSfF2HQ3RvNcpKKcjQki7IY&index=13' \
  --output-dir data \
  --output-name reference_clip \
  --start 60 \
  --duration 12
```

## Install Dependencies

This project pins the official package from GitHub in `pyproject.toml`. Install all dependencies with:

```bash
uv sync
```

The first synthesis run downloads model files into `pretrained_models/`.

## Synthesize From Downloaded Reference Audio

`--reference-text` must be the transcript of the reference audio. For best results, use a short clean segment and a transcript that exactly matches that segment.

```bash
uv run ace-tts synthesize \
  --reference-audio data/reference.wav \
  --reference-text 'ここに参照音声の文字起こしを入れてください。' \
  --text '生成したい文章をここに入れてください。' \
  --output output/cloned.wav
```

The generated test file in this workspace was created with:

```bash
uv run ace-tts synthesize \
  --reference-audio data/reference_clip.wav \
  --reference-text '感じ茶色っぽいやつやけどさ先がすれてて、だんだんはげてきてて、はげてるみたいな。こうとんがってる感じで、なんかこうこう。' \
  --text 'こんにちは。これはサラシナTTSを使った音声クローンのテストです。' \
  --output output/cloned.wav
```

## One-Step Download And Synthesis

```bash
uv run ace-tts clone-from-youtube \
  'https://www.youtube.com/watch?v=2Zb0AiL3IO0&list=PLlO7zjtaf1FSfF2HQ3RvNcpKKcjQki7IY&index=13' \
  --reference-text 'ここに参照音声の文字起こしを入れてください。' \
  --text '生成したい文章をここに入れてください。' \
  --work-dir data \
  --start 60 \
  --duration 12 \
  --output output/cloned.wav
```
