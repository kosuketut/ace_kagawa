# Linux RTX Audio2Face Sandbox Architecture

`Tokkio` を主経路にする場合は [tokkio-reference-stack.md](/home/kyano/workspace/ACE/ace_kagawa/docs/architecture/tokkio-reference-stack.md) を参照してください。この文書は Unreal 直結の研究サンドボックス用です。

## Goal

対象は `Linux RTX` 単一ワークステーション上の `リアルタイム half-duplex` 会話アバターです。`GPU0` を Unreal/MetaHuman 表示用、`GPU1` を Speech NIM 用に固定します。会話状態は `LISTENING -> THINKING -> SPEAKING -> LISTENING` の 4 状態に限定し、`THINKING` と `SPEAKING` 中の新規ユーザー発話は処理しません。

## Runtime Split

- `Unreal Engine 5.6`
  - MetaHuman 描画
  - マイク入力取得
  - orchestrator との WebSocket 接続
  - TTS PCM の再生
  - ACE `Audio2Face-3D` への PCM 入力
- `services/orchestrator`
  - WebSocket セッション管理
  - WebRTC VAD による EOS 判定
  - ASR ストリーミング
  - NVIDIA NIM API への LLM ストリーミング
  - sentence chunking 後の TTS ストリーミング
  - ターン単位の JSONL ロギング
- `infra/compose`
  - `GPU1` 固定の `ASR NIM` / `TTS NIM`

## Audio / Data Contracts

- 上り音声
  - `16kHz`
  - `mono`
  - `PCM16`
  - `20ms` フレーム
- 下り音声
  - `24kHz`
  - `mono`
  - `PCM16`
  - `40-80ms` チャンク想定

## WebSocket Protocol

1 本の WebSocket 上で `JSON text frame` と `binary audio frame` を混在させます。

### JSON Control Frame

```json
{
  "type": "session.start",
  "session_id": "0f5416eb-6fc2-4b6e-ae22-44ab2df79374",
  "turn_id": null,
  "timestamp": "2026-04-21T07:00:00Z",
  "payload": {
    "locale": "ja-JP"
  }
}
```

### Event Types

- `session.start`
- `mic.end`
- `asr.partial`
- `asr.final`
- `llm.delta`
- `tts.start`
- `tts.end`
- `state`
- `error`

`mic.chunk` と `tts.chunk` は binary frame で搬送します。

### Binary Audio Frame

ヘッダは 32 bytes です。

- bytes `0..3`: magic = `ACE1`
- byte `4`: version = `1`
- byte `5`: kind = `1: mic`, `2: tts`
- byte `6`: codec = `1: PCM_S16LE`
- byte `7`: channels
- bytes `8..11`: sample rate, big-endian uint32
- bytes `12..15`: payload size, big-endian uint32
- bytes `16..31`: turn id bytes, optional
- bytes `32..`: raw PCM payload

## Latency Markers

各ターンで以下を JSONL 出力します。

- `vad_start_ms`
- `eou_detected_ms`
- `asr_final_ms`
- `llm_first_token_ms`
- `tts_first_audio_ms`
- `a2f_start_ms`
- `turn_total_ms`

出力先は `/home2/ko66/ace-sandbox/logs/turns-YYYYMMDD.jsonl` です。

## Unreal Integration Notes

- `Audio2Face-3D` は Linux でローカル推論に寄せず、`ACE Unreal Plugin` の `RemoteA2F` プロバイダを前提にする
- `Animation Stream` は使わない
- 受信 TTS PCM を `USoundWaveProcedural` と `FACERuntimeModule::Get().AnimateFromAudioSamples()` の両方へ送る
- セッション終端時は `EndAudioSamples()`、切断時は `CancelAnimationGeneration()` を呼ぶ

## Persistent Storage

repo 外の大きな永続領域は以下を使います。

- `/home2/ko66/ace-sandbox/nim-cache/asr`
- `/home2/ko66/ace-sandbox/nim-cache/tts`
- `/home2/ko66/ace-sandbox/logs`
- `/home2/ko66/ace-sandbox/audio`
- `/home2/ko66/ace-sandbox/ue-ddc`
- `/home2/ko66/ace-sandbox/docker`
