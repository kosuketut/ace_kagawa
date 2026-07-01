# ACE Linux RTX Sandbox

Linux RTX 上で NVIDIA ACE 系の digital human 構成を試すためのリポジトリです。現在の主経路は `Tokkio 5.0` を使った `browser-first / Kubernetes / multi-service` 構成で、既存の `Audio2Face + MetaHuman + ASR/LLM/TTS` 直結サンドボックスは Unreal 中心の研究用サブパスとして残しています。

大きなキャッシュ、生成音声、Docker bind mount、UE の重い生成物は repo の外に置きます。LLM の model / cache / engine は `/data/ACE` 配下に置き、repo 側からは `./data` symlink で参照します。Tokkio の作業ディレクトリは `infra/tokkio/workspace` に置きますが、Git 管理外です。既定の音声・Tokkio 補助用外部永続領域は `/home2/ko66/ace-sandbox` です。

## Layout

- `docs/architecture/tokkio-reference-stack.md`: Tokkio 5.0 を主系にした構成メモ
- `infra/tokkio`: Tokkio one-click deployment 用の前提チェック、環境変数雛形、実行ラッパ
- `infra/llm`: LLM NIM / OpenAI 互換 endpoint 検証と、任意の local TensorRT-LLM 補助手順
- `docs/architecture/ace-sandbox.md`: 全体アーキテクチャ、状態遷移、WebSocket プロトコル、運用前提
- `infra/compose/docker-compose.yml`: `GPU1` に固定した ASR/TTS NIM 起動雛形
- `services/orchestrator`: FastAPI + asyncio ベースの half-duplex 会話オーケストレータ
- `unreal/Plugins/ACEConversation`: UE 5.6 プロジェクトへ組み込むための C++ コンポーネント雛形

## Persistent Directories

初回に以下を作成します。

```bash
mkdir -p /data/ACE/{data,hf-cache,models,logs}
ln -s /data/ACE/data ./data
mkdir -p /home2/ko66/ace-sandbox/{nim-cache/{asr,tts},logs,audio,ue-ddc,docker}
mkdir -p infra/tokkio/workspace/{controller,logs,state}
```

`./data` が既に存在する場合は、実体が `/data/ACE/data` に向いていることを確認してから使います。大容量 model, checkpoint, TensorRT engine, Hugging Face cache は repo 直下に置きません。

## LLM NIM

Tokkio の LLM は NVIDIA NIM の hosted OpenAI-compatible endpoint を使い、既定では Stockmark の `stockmark/stockmark-2-100b-instruct` に接続します。API key は tracked file には置かず、`infra/tokkio/.env` の `TOKKIO_NVIDIA_API_KEY` または `TOKKIO_LLM_API_KEY` で渡します。

疎通確認は次の形です。

```bash
export NVIDIA_API_KEY=<your-nvidia-nim-api-key>
python3 infra/llm/check_llm_endpoint.py \
  --base-url https://integrate.api.nvidia.com/v1 \
  --model stockmark/stockmark-2-100b-instruct
```

ローカル TensorRT-LLM で Osaka-Swallow を出す旧経路は、研究用の任意サブパスとして `infra/llm/README.md` に残しています。

## RAG

Tokkio の裏側で RAG を使う場合は、host 側で NVIDIA RAG Blueprint を起動し、controller の `llm_processor` を `NvidiaRAGService` に切り替えます。この variant の既定 collection 名は `ace_kagawa` です。

```bash
export NVIDIA_API_KEY=<your-nvidia-nim-api-key>
infra/rag/manage_rag.sh init
infra/rag/manage_rag.sh start
mkdir -p data/rag/corpus
infra/rag/ingest_local_corpus.sh --collection ace_kagawa
python3 infra/tokkio/prepare_tokkio_workspace.py --env-file infra/tokkio/.env
infra/tokkio/manage_tokkio.sh sync-controller --env-file infra/tokkio/.env
```

RAG Blueprint の checkout と Docker runtime は `/data/ACE/rag`、文書 corpus は `data/rag/corpus` に置きます。詳細は `infra/rag/README.md` を参照してください。

## Tokkio 5.0

`Tokkio` を主系で使う場合は、まず `infra/tokkio` を使って controller 側の準備を行います。ここで行うのは `my-config.env` 生成、永続ディレクトリの作成、`envbuild.sh` 実行ラッパの提供です。Tokkio 本体は NVIDIA 配布の `NVIDIA/ACE` repo 側にあります。

```bash
cp infra/tokkio/.env.example infra/tokkio/.env
python3 infra/tokkio/prepare_tokkio_workspace.py --env-file infra/tokkio/.env
```

`infra/tokkio/.env` の LLM 既定値は `TOKKIO_LLM_BASE_URL=https://integrate.api.nvidia.com/v1`、`TOKKIO_LLM_MODEL=stockmark/stockmark-2-100b-instruct` です。`TOKKIO_LLM_API_KEY` が空の場合は `TOKKIO_NVIDIA_API_KEY` を Tokkio の `NVIDIA_LLM_API_KEY` として再利用します。

その後、NVIDIA 公式 quickstart に沿って `NVIDIA/ACE` を `infra/tokkio/workspace/NVIDIA-ACE` に取得し、`infra/tokkio/deploy_tokkio.sh` で `init-config` と `install` を実行します。

```bash
infra/tokkio/deploy_tokkio.sh init-config --env-file infra/tokkio/.env
$EDITOR /home/kyano/workspace/ACE/ace_kagawa/infra/tokkio/workspace/controller/ace-app-config.yml
infra/tokkio/deploy_tokkio.sh install --env-file infra/tokkio/.env
python3 infra/tokkio/check_tokkio_endpoints.py --insecure --kubectl --ui-url https://<app-ip>:30111 --api-url https://<app-ip>:30888
```

詳細は `infra/tokkio/README.md` と `docs/architecture/tokkio-reference-stack.md` にまとめています。

## Unreal-Centric Sandbox

以下は Tokkio ではなく、Unreal 直結の研究用サンドボックスです。

### Speech NIM

1. `infra/compose/.env.example` を `infra/compose/.env` にコピーして、NGC とイメージ名を埋めます。
2. 起動します。

```bash
docker compose --env-file infra/compose/.env -f infra/compose/docker-compose.yml up -d
```

ASR は host の `50051/9000`、TTS は `50052/9001` を使う前提です。

### Orchestrator

```bash
cd services/orchestrator
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
python3 tools/init_storage.py
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

`ACE_NIM_API_KEY`、Speech NIM の接続先を `.env` で調整してください。LLM は既定で `https://integrate.api.nvidia.com/v1` の `stockmark/stockmark-2-100b-instruct` を使い、起動時に `/v1/models` でモデル存在確認を行います。

起動後は `GET /status` で外部依存を含む状態確認ができます。モックモードの疎通確認には `services/orchestrator/tools/demo_client.py` を使えます。

### Unreal

`unreal/README.md` に組み込み手順をまとめています。重要なポイントは次のとおりです。

- UE 5.6 の C++ プロジェクトを作成する
- MetaHuman と ACE Unreal Plugin を入れる
- `Face_AnimBP` に `Apply ACE Face Animations` を追加する
- `mh_arkit_mapping_pose_A2F` を使う
- `Mouth Close` の既定カーブ干渉を無効化する
- `ConversationBridgeComponent` で WebSocket と PCM の受け渡しを接続する
- `ACEAudioPlaybackComponent` で受信 PCM を `USoundWaveProcedural` へ流す

## Verification

Tokkio の確認は `infra/tokkio/check_tokkio_endpoints.py` を使います。Unreal 中心のサンドボックスを確認する場合は、依存が入った環境で以下を順に確認します。

```bash
python3 -m unittest discover -s services/orchestrator/tests
python3 services/orchestrator/tools/demo_client.py --url ws://127.0.0.1:8080/ws/session --mock-audio
curl -s http://127.0.0.1:8080/status
```
