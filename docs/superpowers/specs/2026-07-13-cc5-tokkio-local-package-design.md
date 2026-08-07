# CC5 Tokkio Local Package Integration Design

## Goal

Run the UE 5.5.4 CC5 Linux package in the existing Tokkio 5.0 Unreal Renderer without using NGC Resource Downloader or replacing the existing renderer PVC.

## Current baseline

- Helm release: `tokkio-app`, chart `tokkio-1stream-with-ui-5.0.0-GA`, revision 1.
- Renderer: `ia-unreal-renderer-microservice-deployment-0`, GPU 1, three ready containers.
- Existing asset PVC: `ia-unreal-renderer-microservice-assets`, `mdx-local-path`, initially 5000Mi.
- Current project: Aki under `/home/unreal-renderer/unrealEngineProject`.
- CC5 source: `/home2/ko66/ace-sandbox/tokkio/unreal-resources/cc5_tokkio/2026-07-11`.
- Source size: 1,701,680,607 bytes. Current PVC use: approximately 3,958,865,920 bytes.

## Architecture

The official one-click `user_value_override_files` mechanism supplies a repository-managed Helm values override. The override replaces the NGC download initContainer with a local staging initContainer. The CC5 source is mounted read-only by `hostPath`; the existing PVC remains mounted at `/home/unreal-renderer`.

The PVC request is expanded in place from 5000Mi to 8Gi. No PVC deletion or replacement is allowed. The initContainer stages the source as `unrealEngineProject.next`, validates capacity and hashes, fixes ownership and permissions, then atomically rotates `unrealEngineProject` to `unrealEngineProject.prev` and promotes `.next`. A trap restores `.prev` if promotion fails.

## UE 5.5 alignment

- Set `unrealEngine.version: "5.5"` as explicit compatibility metadata.
- Set `unrealEngine.signallingServerVersion: "5.5"`; tag availability was verified.
- Add `UE_VERSION=5.5` and `UNREAL_ENGINE_VERSION=5.5` to the renderer and signalling container environments.
- Keep the existing renderer microservice image and GPU 1 assignment.

## Integrity contract

The following source hashes are immutable for version `2026-07-11`:

- `cc5_tokkio.sh`: `031bc7c70cd6287bd1cc4d83ab78c830a91ae6b93fc4b161d2041723a036b1b9`
- `cc5_tokkio/Binaries/Linux/cc5_tokkio`: `55f4e7f71a823392d80a7d91560294430fb3cf7c84ee14de453ba44177e7dcd0`
- `cc5_tokkio/Content/Paks/cc5_tokkio-Linux.ucas`: `e267c048d92a28bb9924fd22001f10c4b03ca12f0bb27873f92e8244a5c72161`

The staged copies must match all three values before promotion. The installed copy is checked again when the version marker matches; a matching version and matching hashes skip the copy. A version marker alone is insufficient.

## Capacity gate

The script calculates the byte size of the source and current PVC content. It requires enough logical 8Gi capacity for current content, the complete `.next` copy, and a 512Mi safety margin. It also checks filesystem free bytes. Failure exits before changing the current project.

## Staging and rollback

1. Validate source files and source hashes.
2. If marker and installed hashes match, exit successfully without copying.
3. Validate logical PVC and filesystem capacity.
4. Remove only stale `.next`, then copy the complete source into `.next`.
5. Validate the three staged hashes.
6. Set staged ownership to UID/GID 1000 and permissions to user-writable, world-readable/traversable.
7. Write the version and hash marker inside `.next`.
8. Remove an older `.prev`, rename current project to `.prev`, and rename `.next` to current.
9. On any promotion error, restore `.prev`.
10. Retain `.prev` through runtime validation so the previous Aki project remains recoverable.

## Validation and deployment

Before mutation:

- Save Helm history, all values, manifest, renderer Pod YAML, StatefulSet YAML, and PVC YAML under the ignored `infra/tokkio/workspace/state/cc5-integration/` directory with mode 0700.
- Run unit tests for the override and staging script.
- Render the chart with `helm template` and inspect the renderer StatefulSet.
- Run `kubectl apply --dry-run=server` on the rendered manifest.

Deployment uses the existing one-click install path with the added override file. Only the renderer Pod is expected to restart.

Runtime validation covers PVC capacity, init logs, Pod readiness, the `cc5_tokkio` process, `/health`, signalling streamer registration, WebRTC, Tokkio UI rendering, and facial motion during speech. If renderer readiness, health, or streaming fails, restore the saved revision or `.prev` project and verify the Aki baseline.

## Security and repository scope

- No API key, token, password, Helm secret, or TLS private key enters a tracked file.
- Sensitive pre-change snapshots remain under ignored workspace state with mode 0700.
- Existing unrelated changes are preserved.
- No commit or push is performed.
