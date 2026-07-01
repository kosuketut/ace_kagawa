# Orchestrator

FastAPI ベースの WebSocket オーケストレータです。役割は以下です。

- Unreal から `16kHz mono PCM16` を受ける
- VAD で EOS を検出する
- ASR をストリーミングし `partial/final transcript` を返す
- NVIDIA NIM API へ会話プロンプトを送り、streaming delta を返す
- 文単位に区切って TTS を streaming し、受信 PCM を Unreal へ返す

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
python3 tools/init_storage.py
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

`.env.example` の LLM 既定値は hosted NVIDIA NIM の `https://integrate.api.nvidia.com/v1` と `stockmark/stockmark-2-100b-instruct` です。`ACE_NIM_API_KEY` だけ実運用値に差し替えてください。

## Control Frames

JSON text frame の envelope は以下です。

```json
{
  "type": "state",
  "session_id": "a79e3ab3-4fbf-4ffc-b0fa-ad16bb6f8139",
  "turn_id": "eb799a38-19f1-4eb5-ad54-96bdf9518efe",
  "timestamp": "2026-04-21T07:00:00Z",
  "payload": {
    "state": "SPEAKING"
  }
}
```

### Required Client Flow

1. `session.start` を送る
2. `kind=mic` の binary frame を 20ms ごとに送る
3. クライアント側で明示的に終端が分かる場合は `mic.end` を送る
4. `asr.partial` / `asr.final` / `llm.delta` / `tts.start` / binary `tts` / `tts.end` を受ける

## Binary Audio Frames

- magic: `ACE1`
- version: `1`
- kind: `1=mic`, `2=tts`
- codec: `1=PCM_S16LE`
- channels: `1`
- sample rate: big-endian uint32
- payload size: big-endian uint32
- turn id bytes: 16 bytes
- payload: raw PCM16

## Notes

- `ACE_MOCK_ASR=true`
- `ACE_MOCK_LLM=true`
- `ACE_MOCK_TTS=true`

を指定すると、外部サービスなしで WebSocket の疎通確認だけ先に進められます。

## Utilities

- `tools/init_storage.py`: `/home2/ko66/ace-sandbox` 配下の永続ディレクトリを初期化する
- `tools/demo_client.py`: WAV もしくは疑似マイク音声で orchestrator と会話フローを検証する

## HTTP Endpoints

- `GET /healthz`: プロセス生存確認
- `GET /status`: ASR / TTS / LLM の接続状態とモック設定を返す
