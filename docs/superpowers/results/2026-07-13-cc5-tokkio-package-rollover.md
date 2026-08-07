# CC5 Tokkio package rollover result

Date: 2026-07-13 14:27 JST

## Outcome

The live Tokkio Unreal Renderer was switched from the CC5 package `2026-07-11`
to `/home2/ko66/ace-sandbox/tokkio/unreal-resources/cc5_tokkio/2026-07-13`.
The existing 8 GiB PVC was retained, and the previous `2026-07-11` project is
available at `/home/unreal-renderer/unrealEngineProject.prev` for rollback.

## Package integrity

- `cc5_tokkio.sh`: `031bc7c70cd6287bd1cc4d83ab78c830a91ae6b93fc4b161d2041723a036b1b9`
- `cc5_tokkio/Binaries/Linux/cc5_tokkio`: `8b6c99fd17c7db9d125f8b72d643aa3ac2ae5d8cc4d787235170b5a7e9461768`
- `cc5_tokkio/Content/Paks/cc5_tokkio-Linux.ucas`: `6d0f5a78b9672cca0db0ec1452c6cc886f732461959cf194a9ddb82939e3125d`

The init container verified the source and staged hashes before promotion. Its
capacity gate measured `7,898,835,384` required bytes against an
`8,589,934,592` byte logical PVC capacity.

## Runtime verification

- Renderer StatefulSet: `1/1` ready.
- Renderer Pod: `3/3 Running`, zero restarts, node `mercury`.
- Installed marker: `2026-07-13` with the three hashes above.
- Unreal executable: `/home/unreal-renderer/assembledProject/cc5_tokkio/Binaries/Linux/cc5_tokkio`.
- Renderer health: HTTP 200, `Health Check Successful!`.
- Signalling endpoint: HTTP 200; the new Pixel Streaming streamer registered.
- Tokkio UI: HTTPS 200.
- Ingress lifecycle endpoint: HTTP 200.
- Unreal loaded `/Game/Maps/L_CC5_Tokkio` and selected `CC5TokkioAvatar_1` as the stream actor.

Automated in-app visual inspection was unavailable in this session. No new
speech/facial-animation interaction was run as part of this package-only
rollover. No commit or push was performed.
