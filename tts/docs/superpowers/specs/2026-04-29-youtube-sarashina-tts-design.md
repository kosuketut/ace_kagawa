# YouTube Sarashina TTS Design

## Goal

Build a local workflow that downloads authorized YouTube audio and uses it as the reference voice for `sbintuitions/sarashina2.2-tts` zero-shot speech generation.

## Approach

The project exposes a small CLI with three commands:

- `download-reference`: downloads a single YouTube video audio track and converts it to wav.
- `synthesize`: generates speech from an existing reference wav and matching transcript.
- `clone-from-youtube`: downloads the reference wav, then runs synthesis in one command.

The implementation keeps YouTube acquisition, CLI parsing, and Sarashina generation in separate modules so tests can cover orchestration without loading the model.

## Constraints

The workflow assumes the speaker and rights holder have granted permission to use the audio. The model is non-commercial, and generated audio should keep the model's default watermark.

## Testing

Unit tests mock the downloader and generator. They verify CLI argument handling, missing reference-file validation, and command orchestration without downloading media or loading model weights.
