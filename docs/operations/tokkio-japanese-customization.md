# Tokkio Japanese Customization

`Tokkio 5.0` の quickstart は、そのままだと `ASR=英語`, `TTS=ElevenLabs 前提`, `LLM プロンプト=英語` の寄り方が強いです。日本語化し、LLM を hosted NVIDIA NIM の Stockmark model に固定するには `ace-controller` 側の `llm-rag` ソースを調整する必要があります。

この repo には、そのための補助スクリプトとして [`infra/tokkio/customize_tokkio_japanese.py`](/home/kyano/workspace/ACE/ace_kagawa/infra/tokkio/customize_tokkio_japanese.py) を追加しています。

現在の Tokkio では `ace-controller` の StatefulSet が公式 image `nvcr.io/nvidia/ace/tokkio-reference-ace-controller:5.0.0` を使いながら、実行コードを `/code` の PVC `ace-controller-app-storage` から読み込みます。さらに `ACE Configurator` がこの `app-storage-volume` を `HotReload` するので、今回の日本語化は custom image ではなく `NVIDIA-ACE` clone 側の `src/llm-rag` を更新するのが正しい経路です。

## What It Changes

- `workflows/tokkio/5.0.0-ga/src/llm-rag/src/config.py`
  - `RivaASRService`, `RivaTTSService`, `ElevenLabsTTSService` の設定項目を追加
- `workflows/tokkio/5.0.0-ga/src/llm-rag/configs/config.yaml`
  - welcome / farewell / proactivity / system prompt を日本語化
  - system prompt を短い自然な標準語応答向けに変更
  - `OpenAILLMContext.name="香川豊"` とし、`香川先生` / `香川豊先生` などの呼びかけを自分への呼びかけとして扱う
  - system prompt に、東京工科大学 学長、片柳研究所 教授、セラミックス複合材料センター長、材料強度学・複合材料・CMC・SiC/SiC 複合材料などの固定プロフィールを入れる
  - RAG suffix prompt でも、香川先生について聞かれた場合は一人称の `私は` / `私の` で答えるよう補強する
  - `NvidiaLLMService.base_url` を hosted NVIDIA NIM の OpenAI-compatible endpoint に変更
  - `NvidiaLLMService.model` を `stockmark/stockmark-2-100b-instruct` に変更
  - `TOKKIO_RAG_ENABLED=true` かつ `TOKKIO_RAG_MODE=auto` の場合は `Pipeline.llm_processor` を `NvidiaLLMRAGRouterService` にし、通常会話は NIM 直呼び、資料・論文・詳細質問だけ RAG に回す
  - `TOKKIO_RAG_PROVIDER=local` の場合は `/code/configs/local_rag.sqlite` を controller 内で検索し、上位 chunk を NIM prompt に注入する
  - `TOKKIO_RAG_PROVIDER=nvidia` の場合は host-side NVIDIA RAG Blueprint の `/generate` API を使う
  - `TOKKIO_RAG_MODE=always` の場合は従来どおり `Pipeline.llm_processor` を `NvidiaRAGService` に変更
  - `TOKKIO_RAG_MODE=off` または `TOKKIO_RAG_ENABLED=false` の場合は `Pipeline.llm_processor` を `NvidiaLLMService` に変更
  - `NvidiaRAGService.rag_server_url` / `collection_name` を host-side NVIDIA RAG Blueprint に合わせる
  - `NvidiaRAGService.vdb_top_k=12` / `reranker_top_k=5` / `enable_reranker=true` で検索精度を保ちながら LLM に渡す context 量を抑える
  - 表・図・画像系の質問では `multimodal_reranker_top_k=10` に自動で広げる
  - `NvidiaRAGService.max_tokens=128` と短文プロンプトで音声対話向けの生成長に制限する
  - `Pipeline.time_delay=6.0` で、通常のRAG応答では filler phrase より本回答を優先する
  - `ASR language=ja-JP`
  - `ASR model=nvidia/nemotron-3.5-asr-streaming-0.6b`
  - `TTS processor=RivaTTSService`
  - `Riva TTS voice=Magpie-Multilingual.JA-JP.Aria.Neutral`
- `workflows/tokkio/5.0.0-ga/src/llm-rag/src/bot.py`
  - `NvidiaLLMService` の API key として `NVIDIA_LLM_API_KEY`, `LLM_API_KEY`, `NVIDIA_API_KEY` の順に使う
  - ASR / TTS を config 駆動に変更
  - TTS を `Riva` と `ElevenLabs` で切り替え可能に変更

## Apply

```bash
python3 infra/tokkio/customize_tokkio_japanese.py \
  --ace-repo-dir /home/kyano/workspace/ACE/ace_kagawa/infra/tokkio/workspace/NVIDIA-ACE \
  --llm-base-url https://integrate.api.nvidia.com/v1 \
  --llm-model stockmark/stockmark-2-100b-instruct \
  --rag-enabled true \
  --rag-mode auto \
  --rag-provider local \
  --local-rag-db data/rag/local/local_rag.sqlite \
  --rag-server-url http://10.209.1.12:8081/v1 \
  --rag-collection-name ace_kagawa
```

`infra/tokkio/prepare_tokkio_workspace.py` からも同じパッチを自動適用するようにしてあるため、通常は `start` / `reapply` 前に個別実行しなくても構いません。

`.env` の `TOKKIO_LLM_BASE_URL` が空の場合、`prepare_tokkio_workspace.py` は `https://integrate.api.nvidia.com/v1` を使います。`TOKKIO_LLM_API_KEY` が空の場合は `TOKKIO_NVIDIA_API_KEY` を LLM API key として再利用します。

Hosted NIM を使う場合、`manage_tokkio.sh start|stop|restart` で local LLM コンテナを制御する必要はありません。local TensorRT-LLM を使う場合だけ [`infra/llm/README.md`](/home/kyano/workspace/ACE/ace_kagawa/infra/llm/README.md) の手順を参照してください。

`TOKKIO_RAG_PROVIDER=local` では、controller を同期または再起動する前に `python3 infra/rag/build_local_index.py --corpus data/rag/corpus --db data/rag/local/local_rag.sqlite` で index を更新してください。`TOKKIO_RAG_PROVIDER=nvidia` かつ `TOKKIO_RAG_MODE=always` では、host-side NVIDIA RAG Blueprint を起動し、`http://127.0.0.1:8081/v1/health?check_dependencies=true` が通ることを確認してください。RAG の起動・索引作成・文書取り込みは [`infra/rag/README.md`](/home/kyano/workspace/ACE/ace_kagawa/infra/rag/README.md) を参照してください。

## After Applying

この変更は `ACE clone` の source を書き換えるだけなので、停止中または稼働中の Tokkio へ反映するには `ACE Configurator` に同期させる必要があります。通常は次で十分です。

1. `./manage_tokkio.sh start --env-file infra/tokkio/.env`
2. 反映が鈍い場合は `./manage_tokkio.sh reapply --env-file infra/tokkio/.env`
3. controller だけ再読込したい場合は `./manage_tokkio.sh restart-controller --env-file infra/tokkio/.env`
4. 接続先の `Riva / Speech NIM` 側でも日本語 `ASR` と `TTS` が使える状態にする
5. `https://integrate.api.nvidia.com/v1` で `/models` と `/chat/completions` streaming が使える API key を `.env` に設定する

## Notes

- `RivaASRService.model` は `nvidia/nemotron-3.5-asr-streaming-0.6b` に固定しています。接続先の Riva / Speech NIM 側でも同じ ASR model id が利用できる状態にしてください
- `ElevenLabs` を使い続けたい場合は `Pipeline.tts_processor` を `ElevenLabsTTSService` に戻し、`ElevenLabsTTSService.voice_id` に日本語対応 voice id を設定してください
- この repo の `infra/compose/check_nim_stack.py` は既に `ja-JP` と `Magpie-Multilingual.JA-JP.Aria.Neutral` を既定にしているので、日本語 Speech NIM の疎通確認に使えます
- Hosted NIM / local LLM の検証は `infra/llm/check_llm_endpoint.py` を使います
