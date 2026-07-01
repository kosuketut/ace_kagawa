# Tokkio 5.0 Reference Stack

## Goal

`Tokkio 5.0` を使って、Linux RTX 上で `browser-first` の digital human workflow を立ち上げるための基準構成です。対象は `WebRTC 配信`、`Kubernetes ベースの multi-service deployment`、`vision を含む製品寄りの検証` です。`Unreal` は単体アプリの中心ではなく、必要に応じて `Unreal Renderer Microservice` 配下の avatar renderer として扱います。

## Official Baseline

NVIDIA 公式ドキュメントで押さえるべき主な前提は以下です。

- Quickstart は `controller instance` と `application instance` を前提にする
- application 側は `Ubuntu 22.04`、`700 GB` 以上、`2xL4` または `2xA10` 以上を前提にする
- one-click deployment は `NVIDIA/ACE` repo の `workflows/tokkio/5.0.0-ga/scripts/one-click/baremetal` を使う
- ブラウザ接続は `Tokkio UI + WebRTC + coturn + VST`
- 実行系は `ACE Controller + Audio2Face-3D + Animation Graph + Unreal Renderer Microservice + SDR + Tokkio Ingress`

## Runtime View

高レベルの流れは以下です。

1. ブラウザの `Tokkio UI` が `WebRTC` と `WebSocket signaling` で接続する
2. `VST` が新しい stream を publish する
3. `SDR` が stream を各 microservice にルーティングする
4. `ACE Controller` が ASR / LLM / TTS / multimodal 制御を行う
5. `Audio2Face-3D` と `Animation Graph` が顔と全身の animation data を生成する
6. `Unreal Renderer Microservice` が avatar を描画し、ブラウザへ戻す

## Repo Mapping

この repo で管理する責務は `Tokkio 本体の再実装` ではなく、Tokkio を運用しやすくするための周辺物です。

- `infra/tokkio`
  - controller 側 `.env` 雛形
  - `my-config.env` 自動生成
  - `/home/kyano/workspace/ACE/ace_kagawa/infra/tokkio/workspace` 配下の永続領域準備
  - `envbuild.sh` 呼び出しラッパ
  - UI/API/Grafana/Kubernetes の簡易疎通確認
- `docs/architecture`
  - Tokkio を主系にした運用メモ
- `infra/compose`, `services/orchestrator`, `unreal`
  - Tokkio 非採用時の Unreal 直結研究用。Tokkio の主経路ではない

## Persistent Storage

重量物は Git 管理に入れず、以下に寄せます。Tokkio の controller state と公式 clone は repo 配下の ignored workspace に置きます。

- `/home/kyano/workspace/ACE/ace_kagawa/infra/tokkio/workspace/controller`
- `/home/kyano/workspace/ACE/ace_kagawa/infra/tokkio/workspace/logs`
- `/home/kyano/workspace/ACE/ace_kagawa/infra/tokkio/workspace/state`
- `/home2/ko66/ace-sandbox/nim-cache`
- `/home2/ko66/ace-sandbox/docker`

## Recommended Flow

1. `infra/tokkio/.env.example` を `infra/tokkio/.env` にコピーする
2. `prepare_tokkio_workspace.py` で controller 用 `my-config.env` を生成する
3. `NVIDIA/ACE` repo を `/home/kyano/workspace/ACE/ace_kagawa/infra/tokkio/workspace/NVIDIA-ACE` に clone する
4. `deploy_tokkio.sh init-config` で公式 `config-template.yml` を controller ディレクトリへ退避する
5. `ace-app-config.yml` を編集して、必要な avatar / workflow / endpoint を確定する
6. `deploy_tokkio.sh install` で one-click deployment を実行する
7. install の出力で得た `ui_endpoint` と `api_endpoint` を `check_tokkio_endpoints.py` で確認する

## Notes

- Tokkio quickstart の既定 TTS は `ElevenLabs` です。Speech NIM をローカル compose で立てる既存構成とは前提が違います
- `MetaHuman` を Tokkio に入れる場合は `Unreal Renderer Microservice` 側のフローに寄せる
- Web ブラウザ向け完成形を早く触るなら、custom FastAPI orchestrator を主経路にしない方がよいです

## References

- https://docs.nvidia.com/ace/tokkio/5.0/quickstart-guide.html
- https://docs.nvidia.com/ace/tokkio/5.0/overview/architecture.html
- https://docs.nvidia.com/ace/tokkio/5.0/reference-workflow/tokkio-ue.html
