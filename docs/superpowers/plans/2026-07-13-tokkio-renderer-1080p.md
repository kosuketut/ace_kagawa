# Tokkio Renderer 1080p Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render the CC5 Tokkio avatar natively at 1920x1080 while retaining the existing VST 1080p bitrate profile.

**Architecture:** Keep the repository-managed Helm override as the source of truth. Change only the Unreal Renderer window environment variables, render and validate the focused StatefulSet, then roll out that StatefulSet and verify the actual Unreal command line.

**Tech Stack:** Tokkio 5.0, Helm 3, Kubernetes, Python unittest, PyYAML, Unreal Engine 5.5 Pixel Streaming.

---

### Task 1: Lock the renderer resolution in a regression test

**Files:**
- Modify: `infra/tests/test_tokkio_cc5_override.py`
- Test: `infra/tests/test_tokkio_cc5_override.py`

- [ ] Add `test_override_renders_cc5_at_1080p`, reading the `ms` container environment and asserting `IAUEMS_WINDOW_WIDTH == "1920"` and `IAUEMS_WINDOW_HEIGHT == "1080"`.
- [ ] Run `python3 -m unittest infra.tests.test_tokkio_cc5_override.TokkioCc5OverrideTests.test_override_renders_cc5_at_1080p -v` and confirm it fails because the current values are `1280` and `720`.

### Task 2: Change the source Helm override

**Files:**
- Modify: `infra/tokkio/overrides/cc5-unreal-renderer.values.yaml`

- [ ] Change `IAUEMS_WINDOW_WIDTH` from `1280` to `1920`.
- [ ] Change `IAUEMS_WINDOW_HEIGHT` from `720` to `1080`.
- [ ] Leave `IAUEMS_DEPLOYMENT_ADDITIONAL_STARTUP_ARGS=-PixelStreamingEncoderCodec=VP8` and the existing VST bitrate profile unchanged.

### Task 3: Verify locally

**Files:**
- Test: `infra/tests/test_tokkio_cc5_override.py`

- [ ] Run the focused resolution test and confirm it passes.
- [ ] Run `python3 -m unittest infra.tests.test_tokkio_cc5_override -v` and confirm the full focused suite passes.
- [ ] Run `git diff --check` and confirm no whitespace errors.
- [ ] Render the Tokkio Helm chart with the override and confirm the Renderer StatefulSet contains `1920` and `1080`.

### Task 4: Roll out and verify live runtime

**Files:**
- Runtime resource: `statefulset/ia-unreal-renderer-microservice-deployment` in namespace `app`

- [ ] Apply only the validated Renderer StatefulSet, avoiding unrelated chart resources.
- [ ] Wait for `statefulset/ia-unreal-renderer-microservice-deployment` rollout completion.
- [ ] Confirm the Pod is `3/3 Running` with zero container restarts.
- [ ] Confirm the live StatefulSet environment contains `IAUEMS_WINDOW_WIDTH=1920` and `IAUEMS_WINDOW_HEIGHT=1080`.
- [ ] Confirm the Unreal log command line contains `-ResX=1920 -ResY=1080`.
- [ ] Record that browser-side bitrate, packet loss, and FPS still require a connected Chrome session in `chrome://webrtc-internals`.
