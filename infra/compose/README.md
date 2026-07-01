# Speech NIM Compose

`docker-compose.yml` は `GPU1` に ASR/TTS NIM を固定するための最小構成です。大きなキャッシュとログは `/home2/ko66/ace-sandbox` へ逃がします。

## Start

```bash
cp .env.example .env
docker compose --env-file .env -f docker-compose.yml up -d
```

## Verify

```bash
python3 check_nim_stack.py --asr-http-url http://127.0.0.1:9000 --tts-http-url http://127.0.0.1:9001
python3 check_nim_stack.py --tts-grpc 127.0.0.1:50052 --tts-text "こんにちは、音声合成の確認です。"
```

ASR 実測を行う場合は `16kHz mono PCM16 WAV` を指定します。

```bash
python3 check_nim_stack.py --asr-grpc 127.0.0.1:50051 --asr-wav /path/to/input.wav
```

