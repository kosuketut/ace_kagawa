# YouTube Sarashina TTS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local CLI that downloads authorized YouTube audio and uses it as a `sarashina2.2-tts` reference voice.

**Architecture:** Split the workflow into CLI orchestration, YouTube download, and Sarashina synthesis modules. Keep the heavy model import behind the synthesis boundary so tests can run without GPU dependencies.

**Tech Stack:** Python 3.10+, `uv`, `yt-dlp`, `ffmpeg`, `sarashina2.2-tts`.

---

### Task 1: CLI Tests

**Files:**
- Create: `tests/test_cli.py`
- Create: `pyproject.toml`
- Create: `README.md`

- [x] Write unit tests for `download-reference`, `synthesize`, and `clone-from-youtube`.
- [x] Run `uv run python -m unittest tests/test_cli.py -v` and confirm tests fail because `ace_tts` is missing.

### Task 2: CLI Implementation

**Files:**
- Create: `src/ace_tts/__init__.py`
- Create: `src/ace_tts/cli.py`
- Create: `src/ace_tts/downloader.py`
- Create: `src/ace_tts/sarashina.py`

- [x] Implement argparse subcommands and dependency-injected boundaries.
- [x] Implement `yt-dlp` wav download.
- [x] Implement Sarashina generator calls based on the official README usage.
- [x] Run `uv run python -m unittest tests/test_cli.py -v` and confirm all tests pass.

### Task 3: Real Download

**Files:**
- Create: `data/reference.wav`

- [x] Run `uv run ace-tts download-reference '<YouTube URL>' --output-dir data --output-name reference`.
- [x] Confirm `data/reference.wav` is created.

### Task 4: Documentation

**Files:**
- Modify: `README.md`

- [x] Document requirements, download command, Sarashina installation, synthesis command, and one-step command.
