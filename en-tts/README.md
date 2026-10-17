# IndexTTS-2.5 English voice clone for Tokkio

香川先生の英語参照音声を使い、IndexTTS-2.5をTokkioの通常TTSとして提供します。モデル、cache、生成音声は `data -> /data/ACE/en-tts` に保存します。

## Service

```bash
./install_indextts_tts_service.sh
curl -fsS http://127.0.0.1:8023/healthz
```

installerはunitを起動しますが、boot時の自動起動（enable）はしません。boot時に起動すると、日本語モードでもIndexTTSがGPU1を占有するためです。以前の `enable --now` 版で導入済みの場合は、一度だけ `sudo systemctl disable ace-indextts-tts.service` を実行します。

HTTP契約はTokkioの既存Irodori clientと同じです。IndexTTSのnative出力は22.05 kHzですが、serverが `INDEXTTS_TTS_RESPONSE_SAMPLE_RATE_HZ`（unitでは16000、Audio2Faceの16 kHz契約）へband-limited resample（soxr VHQ）して返します。`X-Sample-Rate-Hz` と `/healthz` の `sample_rate_hz` も実際の出力rateを示します。Tokkio clientのresampleは、serverが16 kHz以外を返した場合の予備です。

英語の数字・年・時刻・`%`・`$`・`Dr.` などは、推論前にIndexTTS付属のTextNormalizerで読みへ展開します（例: `1:30 p.m.` → `one thirty PM`）。英字を含まない文や日本語を含む文は、Chinese normalizerへ回さないよう展開せずに渡します。

```bash
curl -fsS http://127.0.0.1:8023/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Good afternoon, everyone.","voice":"kagawa_en","response_format":"wav"}' \
  -o /tmp/indextts-test.wav
```

`voice/symposium_opening_script.md` の7文は起動時に生成してcacheするため、`Start` 後は待たずに順次再生できます。各文末には `INDEXTTS_TTS_TRAILING_SILENCE_MS`（既定500 ms）の無音を付加します。cacheは文字列の完全一致で引くため、原稿は `infra/tokkio/tokkio_llm.py` の `SYMPOSIUM_OPENING_SCRIPT` と一字一句そろえます（`infra/tests/test_en_tts_symposium_script.py` で確認）。

## Tokkio symposium speech

`infra/tokkio/.env` で次を選択し、英語ASRを含むTokkio構成を再適用します。

```dotenv
TOKKIO_TTS_PROCESSOR=QwenTTSService
TOKKIO_IRODORI_TTS_ENABLED=false
TOKKIO_QWEN_TTS_ENABLED=true
TOKKIO_QWEN_TTS_BASE_URL=http://10.209.1.12:8023
TOKKIO_QWEN_TTS_HEALTH_URL=http://127.0.0.1:8023/healthz
TOKKIO_QWEN_TTS_SERVICE=ace-indextts-tts.service
```

`QwenTTSService` と `TOKKIO_QWEN_TTS_*` は既存の英語HTTP transportとの互換名です。実際のbackendはIndexTTS-2.5です。

```bash
infra/tokkio/manage_tokkio.sh reapply --env-file infra/tokkio/.env
```

英語モードでは自動挨拶を無効化します。Tokkio WebUIで `Start` と発話または入力すると固定原稿を7文に分けて通常TTSへ渡し、読み上げ中の `Stop` は即時中断します。`Please read the symposium opening.` も利用できます。

日本語対話へ戻す場合は `TOKKIO_TTS_PROCESSOR=IrodoriTTSService`、Irodori有効、英語TTS無効へ戻し、`reapply` で日本語ASRも戻します。

## Qwen comparison backend

Qwen3-TTS用の `qwen_tts_server.py` と `install_qwen_tts_service.sh` は比較用として残しています。portはElevenLabs gatewayの8022と重ならない8025です。Tokkio本番経路では使用しません。
