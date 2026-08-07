# CC5 Tokkio Local Package Implementation Plan

> **For agentic workers:** Execute inline in the user-specified checkout. Do not commit or push.

**Goal:** Replace the Tokkio Unreal Renderer Aki asset with the UE 5.5.4 CC5 package using an idempotent local-copy initContainer and the existing expanded PVC.

**Architecture:** An official one-click Helm user override mounts the host package read-only and stages it into the existing asset PVC. Capacity, three SHA-256 hashes, ownership, atomic promotion, and rollback are enforced before the renderer starts.

**Tech Stack:** Tokkio 5.0, Helm 3, Kubernetes 1.31, Bash, Python unittest/PyYAML, Unreal Engine 5.5 Pixel Streaming.

---

### Task 1: Add failing configuration and staging tests

**Files:**
- Create: `infra/tests/test_tokkio_cc5_override.py`
- Create later: `infra/tokkio/overrides/cc5-unreal-renderer.values.yaml`

- [ ] Write tests that require the override to use hostPath read-only, an 8Gi existing claim, no NGC downloader, UE/signalling 5.5, GPU 1, version/hash constants, capacity guard, idempotent installed-hash check, `.next`, `.prev`, and rollback trap.
- [ ] Run `python3 -m unittest infra.tests.test_tokkio_cc5_override -v` and confirm it fails because the override is absent.

### Task 2: Implement the Helm override

**Files:**
- Create: `infra/tokkio/overrides/cc5-unreal-renderer.values.yaml`

- [ ] Add a root local-copy initContainer using the renderer 0.1.3 image.
- [ ] Mount the exact CC5 source read-only and the existing asset PVC read-write.
- [ ] Implement capacity gating, source and staged hashes, idempotency, ownership, staging, atomic promotion, and rollback.
- [ ] Set the PVC request to 8Gi and UE/signalling metadata and environments to 5.5.
- [ ] Run the focused test and confirm it passes.

### Task 3: Register the official one-click override

**Files:**
- Modify: `infra/tokkio/workspace/controller/ace-app-config.yml`

- [ ] Add `spec.app.configs.app_settings.helm_chart.repo.user_value_override_files` with the absolute override path while preserving all secrets and unrelated settings.
- [ ] Validate YAML parsing and focused tests.

### Task 4: Render and server-side dry run

- [ ] Render chart 5.0.0-GA with the base profile values and CC5 override.
- [ ] Assert the rendered StatefulSet has no NGC downloader, has UE/signalling 5.5, preserves GPU 1, mounts the hostPath read-only, and requests 8Gi on the same PVC name.
- [ ] Run `kubectl apply --dry-run=server` against the rendered manifest and resolve any schema/admission errors before deployment.

### Task 5: Deploy with the existing one-click path

- [ ] Reconfirm current renderer readiness and endpoints.
- [ ] Run `infra/tokkio/deploy_tokkio.sh install --env-file infra/tokkio/.env`.
- [ ] Wait for PVC expansion and the single renderer Pod rollout.
- [ ] Inspect init logs for capacity values, hash success, staging, permissions, and promotion.

### Task 6: Runtime and end-to-end validation

- [ ] Verify renderer Pod `3/3 Ready` and all supporting current replicas.
- [ ] Verify the live executable is `/home/unreal-renderer/assembledProject/cc5_tokkio/Binaries/Linux/cc5_tokkio`.
- [ ] Verify renderer `/health` returns 200.
- [ ] Verify signalling registers the CC5 streamer and WebRTC establishes a player connection.
- [ ] Verify Tokkio UI shows the CC5 character.
- [ ] Trigger speech and verify facial animation receives and applies audio-driven animation.
- [ ] Record Pod states, logs, endpoints, rendered values, and changed files.

### Task 7: Roll back on failure

- [ ] If init or rollout fails, collect evidence before mutation.
- [ ] Restore the previous project from `.prev` or restore the saved Helm revision and values.
- [ ] Verify the Aki baseline renderer, health, signalling, and UI recover.
