# Tokkio なしのローカル会話（kagawa）

Tokkio（Kubernetes・VST・SDR・Animation Graph）を使わずに、香川先生のアバターとブラウザで会話する。
アバターは Tokkio 用に作った CC5 の Linux パッケージ（`cc5_tokkio` 2026-07-15）をそのまま使う。
会話の進め方は Windows 版 ELF-SR2app（mars: `D:\ACE\ELF-SR2app`）の自動会話モードを移植した。

```
ブラウザ（https://<mercury>:18443/）
  ├─ マイク 16kHz PCM ──WebSocket /mic──▶ server.py
  │                                         VAD → kotoba-whisper → LLM（vLLM :8011）
  │                                         → 文ごとに Irodori（:8021）→ Audio2Face-3D（:18200）
  │                                         → mini_animgraph.py（:18100、Animation Graph の代役）
  └─ 映像・音声 ◀──WebRTC── Unreal renderer（cc5_tokkio、画面なし）◀── 表情＋音声を Pull
```

| 部品 | 実体 | ポート | GPU |
|---|---|---|---|
| 会話サーバー | `server.py`（HTTPS。ページ・マイク・Pixel Streaming 画面とシグナリングの中継） | 0.0.0.0:18443 | GPU1（kotoba-whisper 約 2GB） |
| 中継 | `mini_animgraph.py`（PullAnimationDataStream を返す） | 127.0.0.1:18100、状態 127.0.0.1:18102 | なし |
| Audio2Face-3D | NIM 1.3.16（james_v2.3、表情設定は Tokkio と同じ） | 127.0.0.1:18200 | GPU0 |
| Unreal renderer | `cc5_tokkio`（`-RenderOffScreen`、Pixel Streaming） | HTTP API 0.0.0.0:18021 | GPU1 |
| シグナリング | Pixel Streaming 5.5（docker イメージから取り出してホストの node で実行） | 0.0.0.0:18888 / 18080 / 8889 | なし |
| LLM・TTS | Tokkio と共用（`ace-vllm-llama31-swallow-8b`、`ace-irodori-tts`） | 8011 / 8021 | GPU0 |

実行物・ログ・会話記録は `/data/ACE/ace_kagawa/local_avatar/`（`logs/`、`sessions/`）。

## 使い方

前提: Tokkio は停止中（renderer の GPU と TTS を共有するため同時に動かさない）、containerd・LLM・TTS は起動済み。

```bash
docker start ace-vllm-llama31-swallow-8b        # LLM（未起動なら）
sudo systemctl start ace-irodori-tts             # TTS（未起動なら）
infra/local_avatar/local_avatar.sh start         # A2F・シグナリング・中継・renderer・会話サーバー
infra/local_avatar/local_avatar.sh status
infra/local_avatar/local_avatar.sh stop
```

ブラウザで `https://10.209.1.12:18443/` を開く。自己署名証明書の警告を一度許可し、「会話を始める」でマイクを許可する。
映像の上の再生ボタンを押すと音が出る。話し終えて約 0.7 秒で送信され、返答中はマイクを止める（半二重）。

文字での確認（マイクなし）:

```bash
/data/ACE/ace_kagawa/local_avatar/.venv/bin/python infra/local_avatar/ws_probe.py --text "先生のご専門は何ですか。"
/data/ACE/ace_kagawa/local_avatar/.venv/bin/python infra/local_avatar/ws_probe.py --wav question.wav
```

## 初回準備（済み。作り直すときの記録）

- venv: `/data/ACE/ace_kagawa/local_avatar/.venv`（Python 3.11、grpcio・numpy・fastapi・uvicorn・webrtcvad-wheels・faster-whisper・websockets）
- proto: `proto/`（NVIDIA ACE、Apache-2.0）から `gen/` に生成。
  `python -m grpc_tools.protoc -I proto --python_out=<gen> --grpc_python_out=<gen> proto/*.proto`
- renderer: `/home2/ko66/ace-sandbox/tokkio/unreal-resources/cc5_tokkio/2026-07-15` を `renderer/` にコピー（Tokkio と同じ sha256 で照合）。
- シグナリング: `docker create` + `docker cp` で `pixelstreamingunofficial/pixel-streaming-signalling-server:5.5` の `/SignallingWebServer` を `ps-signalling/` へ。
- ASR: `kotoba-tech/kotoba-whisper-v2.0-faster` を `/data/ACE/models/kotoba-whisper-v2.0-faster` へ。
- A2F: `docker pull nvcr.io/nim/nvidia/audio2face-3d:1.3.16`（k8s の containerd からの書き出しは圧縮レイヤーが無く失敗する）。
  初回だけ `NGC_API_KEY` を渡して起動し、モデルを `a2f-cache/` に取得する。以後は不要。
- TLS: `tls/cert.pem`・`tls/key.pem`（自己署名、SAN は 10.209.1.12・127.0.0.1・mercury・localhost、1 年）。

## 守ること・分かっていること

- **中継と renderer は必ず一緒に再起動する**（`local_avatar.sh start` はそうする）。
  UE はストリームごとに受信した音声サンプル数を覚えていて、時刻コードが巻き戻ると無音の埋め合わせが巨大になり落ちる。
- ACE プラグイン（NVIGI の animgraph クライアント）は、1 フレームに表情・音声・関節の時刻コード 5 つを読み、
  関節の値が空だとクラッシュするかフレームを捨てる。中継はダミー関節 1 つを送る。待機中は音声を送らない（UE が無音を補う）。
- A2F-3D の NIM は `/apps/configs/` の設定を読み、コマンド引数の設定ファイルは無視される。HTTP 8000 番も開くので `--network host` にしない。
- 認証なしで LAN に開いているポートがある（18443、18021、18080、18888、8889）。試作用。
- RAG（kagawa の local_rag）はまだつないでいない。資料にない質問は Tokkio 版より不正確になる。

## テスト

```bash
python3 -m unittest infra/tests/test_local_avatar_conversation.py infra/tests/test_local_avatar_speech_buffer.py
```
