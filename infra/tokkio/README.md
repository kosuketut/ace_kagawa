# Tokkio 5.0 Helpers

このディレクトリは `Tokkio 5.0` を controller 側から起動するための補助物です。Tokkio 本体はここには含めず、NVIDIA 公式の `NVIDIA/ACE` repo を利用します。

## What This Adds

- `prepare_tokkio_workspace.py`
  - `.env` を読んで `/home/kyano/workspace/ACE/ace_kagawa/infra/tokkio/workspace` 配下を初期化
  - one-click script 用の `my-config.env` を生成
  - `TOKKIO_OPENAI_API_KEY` が空でも、Tokkio 5.0 の過剰な secret validation を避けるために placeholder を補完
  - `TOKKIO_LLM_BASE_URL` / `TOKKIO_LLM_MODEL` を日本語向け `llm-rag` config に反映
  - `TOKKIO_RAG_ENABLED=true` の場合は `TOKKIO_RAG_MODE=auto|always|off`、`TOKKIO_RAG_PROVIDER=local|nvidia`、RAG collection/local DB 設定を `llm-rag` config に反映
  - `NVIDIA-ACE` clone が存在すれば、日本語・標準語向け `llm-rag` パッチを自動適用
- `deploy_tokkio.sh`
  - `envbuild.sh` の `init-config`, `install`, `info`, `uninstall` を叩くラッパ
- `manage_tokkio.sh`
  - install 済み Tokkio 環境の `start`, `stop`, `status`, `restart`, `reapply`, `restart-controller`, `logs` をまとめる運用ラッパ
- `check_tokkio_endpoints.py`
  - UI/API/Grafana と `kubectl get pods -n app` の簡易確認
  - `https://...:30888` が plain HTTP を返す環境では自動で HTTP fallback して protocol mismatch を表示
- `check_tokkio_ngc_access.py`
  - 生成済み `my-config.env` を使って Tokkio chart repo へのアクセス可否を先に確認
  - `Audio2Face-3D` イメージの NVCR 取得権限も確認し、NGC ブラウザ上での license 未承認を早期に検出

## Role Split

- `deploy_tokkio.sh`
  - `init-config`, `install`, `info`, `uninstall` のための controller/deploy ラッパ
- `manage_tokkio.sh`
  - install 済み環境の日常運用で使う起動停止・状態確認ラッパ

初回構築やフル再構築は `deploy_tokkio.sh` と runbook を使い、日々の起動停止や再確認は `manage_tokkio.sh` を使います。

## Quick Start

### Initial Deployment

```bash
cp .env.example .env
python3 prepare_tokkio_workspace.py --env-file .env
```

生成後、`/home/kyano/workspace/ACE/ace_kagawa/infra/tokkio/workspace/controller/generated/my-config.env` ができます。

`TOKKIO_OPENAI_API_KEY` を空欄にした場合、生成される `my-config.env` には placeholder 値が入ります。Tokkio 5.0 chart の default `llm_processor` は `NvidiaLLMService` なので、NVIDIA LLM 経路を使う限りこの placeholder は実行時に参照されません。`OpenAILLMService` を使う場合は実際の OpenAI key に差し替えてください。

Tokkio の LLM は既定で hosted NVIDIA NIM の OpenAI-compatible endpoint を使います。`.env` の `TOKKIO_LLM_BASE_URL` は `https://integrate.api.nvidia.com/v1`、`TOKKIO_LLM_MODEL` は `nvidia/nemotron-3-ultra-550b-a55b` です。`TOKKIO_LLM_API_KEY` を空にすると、`TOKKIO_NVIDIA_API_KEY` が Tokkio controller の `NVIDIA_LLM_API_KEY` として使われます。

音声対話では長い内部推論を待たないよう、Nemotron 3 Ultra に `enable_thinking=false`、`temperature=0` を指定します。複雑な推論を優先する別用途では、用途ごとのクライアントから reasoning を明示的に有効化してください。

Hosted model の採用前には、`https://build.nvidia.com/<provider>/<model>` に終了予定の表示がないことを確認してください。`/v1/models` は現時点の提供有無だけを示し、将来の終了予定までは返しません。現在の既定モデルは 2026-07-17 時点で Free Endpoint が利用可能です。以前の `stockmark/stockmark-2-100b-instruct` は hosted Free Endpoint が Deprecated のため、既定値には戻しません。

Hosted NIM を使う場合、`manage_tokkio.sh start|stop|restart` で別途 LLM コンテナを起動・停止する必要はありません。

```bash
TOKKIO_LLM_BASE_URL=https://integrate.api.nvidia.com/v1
TOKKIO_LLM_MODEL=nvidia/nemotron-3-ultra-550b-a55b
TOKKIO_LLM_API_KEY=
```

RAG を使う場合、既定の `TOKKIO_RAG_PROVIDER=local` では `data/rag/corpus` から作った SQLite index を controller に同期し、検索結果だけを `NvidiaLLMService` 相当の NIM 直呼びへ渡します。`TOKKIO_RAG_MODE=auto` では通常会話は直接 NIM で応答し、資料・論文・詳細質問だけ RAG に回します。`always` にすると全発話を RAG に通し、`off` にすると RAG を使いません。NVIDIA RAG Blueprint を使う場合は `TOKKIO_RAG_PROVIDER=nvidia` にします。

```bash
TOKKIO_RAG_ENABLED=true
TOKKIO_RAG_MODE=auto
TOKKIO_RAG_PROVIDER=local
TOKKIO_RAG_SERVER_URL=http://10.209.1.12:8081/v1
TOKKIO_RAG_COLLECTION_NAME=ace_kagawa
TOKKIO_RAG_MAX_TOKENS=64
TOKKIO_RAG_VDB_TOP_K=12
TOKKIO_RAG_RERANKER_TOP_K=5
TOKKIO_RAG_MULTIMODAL_RERANKER_TOP_K=10
TOKKIO_RAG_ENABLE_RERANKER=true
TOKKIO_RAG_ROUTE_KEYWORDS=論文,文献,出典,根拠,資料,ドキュメント,引用,詳細,詳しく,経歴,業績,研究業績,研究内容,プロジェクト,発表,受賞,特許,EBC,CMC,SiC/SiC,非破壊評価
TOKKIO_RAG_FALLBACK_TO_LLM_ON_ERROR=false
TOKKIO_RAG_HEALTH_URL=http://127.0.0.1:8081/v1/health?check_dependencies=true
TOKKIO_LOCAL_RAG_DB=data/rag/local/local_rag.sqlite
TOKKIO_LOCAL_RAG_RUNTIME_DB_PATH=/code/configs/local_rag.sqlite
TOKKIO_LOCAL_RAG_TOP_K=3
TOKKIO_LOCAL_RAG_MAX_CONTEXT_CHARS=2800
```

RAG対象の知識質問で検索結果が得られない場合やRAG処理が失敗した場合は、直接LLMへフォールバックせず、参照情報を確認できない旨を応答します。例外は、`研究`・`受賞` などの一般語だけで香川先生ドメインに振り分けられ（`香川`・`学長`・`あなた`・`先生` を含まない）、質問の話題語がコーパスにない場合です（例: `量子コンピュータの研究について教えてください`）。これは RAG 対象外としてルーティング段階で通常の LLM に回し、「話題が資料にない・自分の研究と関係づけない・大学固有の数値は推測しない」という注記を付けます。判定の詳細は `infra/rag/README.md` の Routing, Confidence, And Citations を参照してください。

local RAG index の作成は repo root から次を使います。

```bash
python3 infra/rag/build_local_index.py \
  --corpus data/rag/corpus \
  --db data/rag/local/local_rag.sqlite
```

NVIDIA RAG Blueprint を使う場合だけ、repo root から次を使います。

```bash
infra/rag/manage_rag.sh start
infra/rag/ingest_local_corpus.sh --collection ace_kagawa
```

LLM endpoint の疎通確認は repo root から次を使います。

```bash
export NVIDIA_API_KEY=<your-nvidia-nim-api-key>
python3 infra/llm/check_llm_endpoint.py \
  --base-url https://integrate.api.nvidia.com/v1 \
  --model nvidia/nemotron-3-ultra-550b-a55b
```

次に NVIDIA 公式 repo を用意します。

```bash
git clone https://github.com/NVIDIA/ACE.git /home/kyano/workspace/ACE/ace_kagawa/infra/tokkio/workspace/NVIDIA-ACE
```

公式の `config-template.yml` を controller 側へコピーします。

```bash
./deploy_tokkio.sh init-config --env-file .env
```

`/home/kyano/workspace/ACE/ace_kagawa/infra/tokkio/workspace/controller/ace-app-config.yml` を編集したら、install を実行します。

```bash
python3 check_tokkio_ngc_access.py --env-file /home/kyano/workspace/ACE/ace_kagawa/infra/tokkio/workspace/controller/generated/my-config.env
./deploy_tokkio.sh install --env-file .env
```

`check_tokkio_ngc_access.py` では Helm repo だけでなく `nvcr.io/nim/nvidia/audio2face-3d` の pull 権限も見ます。`412 Precondition Failed` や `Please accept license on the browser` が返る場合は、NGC/NVIDIA のブラウザ画面で `Audio2Face-3D` NIM の利用規約承認がまだ済んでいません。

デプロイ後に install 出力の endpoint を使って確認します。

```bash
python3 check_tokkio_endpoints.py \
  --insecure \
  --kubectl \
  --ui-url https://<app-ip>:30111 \
  --api-url http://<app-ip>:30888 \
  --grafana-url http://<app-ip>:32300 \
  --tts-base-url http://<app-ip>:8021
```

`--tts-base-url` を付けると、TTS の `/healthz` と短い文（既定「確認しています」）の `/v1/audio/speech` ストリームを確認します。HTTP 200、空でない偶数長の PCM、`X-Sample-Rate-Hz: 16000` を満たさなければ失敗です。`--tts-health-only` で合成を省き、`/healthz` だけを見ます。どれかの確認に失敗すると終了コードは 1 です。

`manage_tokkio.sh` の `status`・`start`・`restart`・`reapply` も、日本語モードでは同じ TTS 確認を行います。合成まで確かめたくない場合は `.env` に `TOKKIO_TTS_SYNTHESIS_CHECK_ENABLED=false` を書きます。ElevenLabs gateway（`:8022`）へ向けている場合は、実行のたびに課金対象の合成リクエストが 1 回発生するため、`false` を推奨します。

`start`・`restart`・`reapply` では、復元した Pod（`tokkio-ingress`、`riva`、`triton` など）が controller より遅れて起動します。そのため最終確認の前に、UI・API・Grafana の確認だけを `TOKKIO_ENDPOINT_WAIT_ATTEMPTS` 回（既定 36）、`TOKKIO_ENDPOINT_WAIT_SECONDS` 秒（既定 5）間隔で繰り返します。待機中は TTS を確認せず、待機後の最終確認で合成を 1 回だけ行います。各 URL の HTTP タイムアウトは 10 秒なので、接続を拒否せずに黙る host では待機が最長で約 20 分になります。`status` は待たずに 1 回だけ確認します。

`start`・`restart` は TTS や RAG の起動確認に失敗しても先に app workload を戻し、最後に終了コード 1 を返します。`reapply` も最終確認に失敗すると終了コード 1 を返します。原因は `[WARN]` 行と、endpoint 確認の JSON で `"ok": false` になっている項目で確認します。

### Day-2 Operations

既存インストール済み環境の起動停止は次を使います。

```bash
./manage_tokkio.sh start --env-file .env
./manage_tokkio.sh status --env-file .env
./manage_tokkio.sh stop --env-file .env
```

`start` は `containerd`, `kubelet`, `nginx`, `coturn` を起動し、endpoint が応答するまで待ってから `kubectl get pods -n app` と endpoint 確認まで流します。`stop` は非破壊停止のみで、`deploy_tokkio.sh uninstall` は呼びません。

現在の `stop` は host 側サービス停止の前に `app` namespace の `Deployment` / `StatefulSet` を `replicas=0` へ落として GPU を使う Tokkio workload を静止します。退避した replica 数は `/home/kyano/workspace/ACE/ace_kagawa/infra/tokkio/workspace/controller/generated/app-workload-replicas.tsv` に保存され、次の `start` で復元されます。

停止待ちでは、`Failed` / `Succeeded` の Pod（以前の ReplicaSet が残した `Error`・`ContainerStatusUnknown`・`Completed` の Pod）は数えません。これらは replica を 0 にしても消えないため、数えると毎回タイムアウトしていました。それでも動作中の Pod が残った場合、`stop` は TTS と host 側サービスの停止まで続けてから、終了コード 1 で終わります。vLLM コンテナ（`ace-vllm-llama31-swallow-8b`）は `stop` では止まりません。止めるときは `containerd` を止める前に `docker stop` してください（後に回すと docker が終了を検知できず、状態が `Up` のまま残ります）。

起動だけで戻らない場合:

```bash
./manage_tokkio.sh reapply --env-file .env
```

`ElevenLabs` など secret を更新して controller に再読込させたい場合:

```bash
./manage_tokkio.sh restart-controller --env-file .env
./manage_tokkio.sh logs controller --env-file .env
```

## Important Notes

- quickstart の既定構成では `TTS` に `ElevenLabs` を使います
- controller と application を別マシンに分けるのが公式前提ですが、同一ワークステーションでも `SSH` と `passwordless sudo` の条件を満たせば運用できます
- Tokkio の作業領域は `infra/tokkio/workspace` 配下に置きます。このディレクトリは Git 管理外です
- この環境では `UI` は `https://<app-ip>:30111`、`Grafana` は `http://<app-ip>:32300`、`API` は `http://<app-ip>:30888` で応答しました。install 出力が `https` を出しても、そのままでは false negative になる場合があります

## Japanese Customization

`Tokkio 5.0` の `ace-controller` は、公式 image を起動しつつ `/code` を `app-storage-volume` にマウントし、`ACE Configurator` がその volume を `HotReload` します。この repo には `NVIDIA-ACE` clone の `llm-rag` ソースへ日本語標準語向けの `ASR / LLM / TTS` 設定を当てる補助スクリプト [`customize_tokkio_japanese.py`](/home/kyano/workspace/ACE/ace_kagawa/infra/tokkio/customize_tokkio_japanese.py) を追加してあり、`prepare_tokkio_workspace.py` からも自動実行されます。

```bash
python3 infra/tokkio/customize_tokkio_japanese.py \
  --ace-repo-dir /home/kyano/workspace/ACE/ace_kagawa/infra/tokkio/workspace/NVIDIA-ACE \
  --llm-base-url https://integrate.api.nvidia.com/v1 \
  --llm-model nvidia/nemotron-3-ultra-550b-a55b
```

通常は個別実行は不要で、`./manage_tokkio.sh start --env-file .env` または `./manage_tokkio.sh reapply --env-file .env` の前に `prepare_tokkio_workspace.py` が自動で同じパッチを当てます。

### オープンキャンパス挨拶

マイクまたはテキスト入力から「オープンキャンパスのご挨拶をお願いします」「オープニングをお願いします」「台本を読んでください」のいずれかを伝えると、AI学長が八王子キャンパス向けの挨拶を文章単位で読み上げます。この定型発話は LLM や RAG を呼ばないため、原稿が言い換えられることはありません。

冒頭の「皆さん、こんにちは。」だけは、`/data/ACE/irodori/fixed-phrases/open_campus_greeting_16k_mono.pcm` が存在する場合に事前生成済み音声を返します。ファイルがない場合は通常のIrodori生成へフォールバックします。

詳細な手順と前提は [`docs/operations/tokkio-japanese-customization.md`](/home/kyano/workspace/ACE/ace_kagawa/docs/operations/tokkio-japanese-customization.md) を参照してください。

### 英語シンポジウム挨拶

英語音声へ切り替える場合は `.env` で `TOKKIO_TTS_PROCESSOR=QwenTTSService` を指定し、`TOKKIO_QWEN_TTS_ENABLED` は `true` にするか空にします（空なら `true` 扱い。`QwenTTSService` なのに明示的に `false` だと `start`・`restart` はエラーで止まります）。互換性のためselector名はQwenのままですが、実backendは `en-tts/indextts_tts_server.py` のIndexTTS-2.5です。香川先生の英語参照音声を使い、サーバー側で22.05 kHzの生成音声をAudio2Faceの16 kHz契約へ帯域制限つきでresampleして返します。日本語モードに戻したときに `ace-indextts-tts.service` が動いていると、GPU1 を使い続けるため `status` が警告します。

```bash
en-tts/install_indextts_tts_service.sh
infra/tokkio/manage_tokkio.sh reapply --env-file infra/tokkio/.env
```

英語モードではASRも `en-US` に切り替わり、自動の挨拶・退出音声は無効になります。WebUIで `Start` と発話または入力すると固定英語原稿を7文に分けてIndexTTSへ渡し、読み上げ中の `Stop` は即時中断して返答音声を出しません。`Please read the symposium opening.` も利用できます。事前生成WAVの直接再生は使いません。
