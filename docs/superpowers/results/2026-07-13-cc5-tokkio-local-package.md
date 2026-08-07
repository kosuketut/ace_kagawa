# CC5 package integration result

Date: 2026-07-13 (Asia/Tokyo)

## Outcome

The CC5 UE 5.5.4 Linux package `2026-07-11` is running in the Tokkio 5.0 Unreal Renderer on node `mercury`. The renderer keeps the existing assets PVC and receives the package through a read-only `hostPath` plus a local copy init container. NGC Resource Downloader is not used.

The deployed renderer uses UE/signalling version 5.5, exposes the expected health and Pixel Streaming endpoints, renders `CC5TokkioAvatar_1`, and completed an ASR -> Irodori TTS -> Audio2Face/Animation Graph -> CC5/WebRTC interaction.

## Deployment evidence

- Helm release before and after the targeted deployment: `tokkio-app`, revision `1`, chart `tokkio-1stream-with-ui-5.0.0-GA`.
- Renderer StatefulSet: generation `3`, observed generation `3`, `1/1` ready.
- Renderer Pod: `3/3 Running`, restart counts `0,0,0`, node `mercury`, IP `192.168.33.45`.
- VMS Pod: Running; its earlier single restart did not increase during the successful final interaction.
- PVC/PV: `ia-unreal-renderer-microservice-assets`, Bound, request/status/PV capacity all `8Gi`.
- Installed project owner: UID/GID `1000:1000`; launcher and executable are mode `0775`.
- Rollback project remains at `/home/unreal-renderer/unrealEngineProject.prev`.

The initial capacity gate measured:

- Existing asset usage: `3,958,760,794` bytes
- CC5 source: `1,701,680,607` bytes
- Safety margin: `536,870,912` bytes
- Required logical capacity: `6,197,312,313` bytes
- PVC logical capacity: `8,589,934,592` bytes

The second init run verified the source and installed files, then logged `CC5 2026-07-11 is already installed and verified; skipping copy`.

## Integrity evidence

The source, staged copy, and deployed copy were checked against these SHA-256 values before promotion:

- `cc5_tokkio.sh`: `031bc7c70cd6287bd1cc4d83ab78c830a91ae6b93fc4b161d2041723a036b1b9`
- `cc5_tokkio/Binaries/Linux/cc5_tokkio`: `55f4e7f71a823392d80a7d91560294430fb3cf7c84ee14de453ba44177e7dcd0`
- `cc5_tokkio/Content/Paks/cc5_tokkio-Linux.ucas`: `e267c048d92a28bb9924fd22001f10c4b03ca12f0bb27873f92e8244a5c72161`

## Validation commands

The main validation commands were:

```bash
python3 -m unittest infra.tests.test_tokkio_cc5_override -v
helm template tokkio-app /tmp/tokkio-chart-inspect/tokkio-1stream-with-ui \
  --namespace app \
  -f infra/tokkio/overrides/cc5-unreal-renderer.values.yaml
kubectl apply --dry-run=server -f renderer-pvc-cc5.yaml
kubectl apply --dry-run=server -f renderer-statefulset-cc5.yaml
kubectl -n app get pod ia-unreal-renderer-microservice-deployment-0
kubectl -n app logs ia-unreal-renderer-microservice-deployment-0 -c stage-cc5-project
kubectl -n app exec ia-unreal-renderer-microservice-deployment-0 -c ms -- \
  sha256sum <launcher> <executable> <ucas>
```

All seven focused unit/integration tests passed. `git diff --check` passed. The renderer PVC and StatefulSet each passed server-side dry-run.

The full chart server-side dry-run remains unsuitable for an all-resource apply because the existing chart renders an invalid container-level `securityContext.fsGroup` in `tokkio-ingress`; it also collides with already allocated fixed NodePorts when rendered as a fresh release. These unrelated resources were not changed. Only the validated renderer PVC and StatefulSet were applied.

## Runtime and endpoint evidence

- Renderer HTTP API inside the Pod: `GET /health` -> `HTTP/1.1 200`
- Signalling NodePort: `http://10.209.1.12:30080/` -> `200`
- Tokkio UI: `https://10.209.1.12:30111/` -> `200`
- Ingress lifecycle health: `http://10.209.1.12:30801/status` -> `200`
- Signalling 5.5 listened on streamer port `8888` and player port `8080`, registered the CC5 streamer, and accepted the final player connection.
- The CC5 process ran with UE HTTP API port `8021`, AnimGraph port `8100`, Pixel Streaming `ws://localhost:8888`, 1280x720, VP8, and GPU assignment `NVIDIA_VISIBLE_DEVICES=1`.

## Voice and facial-animation evidence

The final browser interaction ran from `2026-07-13T02:59:22Z` to `03:00:40Z` with a fake microphone containing Japanese speech.

- Both inbound and outbound WebRTC ICE/peer connections reached `connected`.
- Video `animated-avatar` stayed at `readyState=4`, 1280x720, unpaused, and advanced continuously for 75 seconds.
- ASR final transcript: `こんにちは あなたのお名前を教えてください`
- Irodori TTS response: `私は香川豊です。`
- Controller sent 16 kHz TTS audio frames to Audio2Face and received Animation Graph playback completion.
- Unreal reported `2460 animation samples, 1316829 audio samples` for `/Game/Maps/L_CC5_Tokkio...CC5TokkioAvatar_1`, then completed the stream successfully.
- The UI displayed the CC5 character and the response subtitle during playback.

## Saved state

- Pre-change Helm history, values, manifest, renderer Pod/StatefulSet/PVC, and rendered target resources:
  `infra/tokkio/workspace/state/cc5-integration/20260713T112812+0900-pre-change`
- Post-change renderer Pod/StatefulSet/PVC/PV, init log, filtered E2E logs, and UI captures:
  `infra/tokkio/workspace/state/cc5-integration/20260713T120500+0900-post-change`

No API key, token, or password was added to Git-managed files. No commit or push was performed. The pre-existing `.gitignore` change was preserved.
