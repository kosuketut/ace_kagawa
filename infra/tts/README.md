# Irodori-TTS HTTP service (kagawa)

`irodori_server.py` は Tokkio の `IrodoriTTSService` から呼ばれる host-side TTS です。`/v1/audio/speech` で 16 kHz mono PCM / WAV を返し、`/healthz` で使用中の checkpoint と参照音声の有無を返します。

## モデル

既定は `Aratako/Irodori-TTS-v4.1-Small`（MIT）です。`IRODORI_TTS_HF_CHECKPOINT` で差し替えます。

| checkpoint | 用途 | 備考 |
| --- | --- | --- |
| `Aratako/Irodori-TTS-v4.1-Small` | 既定 | v4 系で読み精度が最も高い。RF 40 steps |
| `Aratako/Irodori-TTS-v4.1-Small-MF` | 低遅延 | MeanFlow 蒸留。4 steps で RF 40 steps に近い読み・話者類似度 |
| `Aratako/Irodori-TTS-v4-Large` | 比較用 | 3.29B。Gemma Terms of Use が適用される。読み精度は v4.1-Small より僅かに低い |
| `Aratako/Irodori-TTS-500M-v3` | 切り戻し | 以前の既定 |

2026-09-29 に v3 / v4.1-Small / v4.1-Small-MF / v4-Large を香川先生の参照音声で聞き比べ、v4.1-Small を採用しました（MF は約 2.5 倍速いものの、試聴では v4.1-Small の方が良好）。比較音声は `/data/ACE/irodori/benchmarks/20260929_v4_upgrade/index.html` にあります。

量子化版は `Aratako/Irodori-TTS-v4.1-Small-Quantized/int8-weight-only` のように `repo/subfolder` で指定できます。v4 系は `model.safetensors` と同梱 `tokenizer/` をまとめて取得するため、`HF_HOME=/data/ACE/irodori/hf-cache` 配下にキャッシュされます。

`IRODORI_TTS_NUM_STEPS` と `IRODORI_TTS_MAX_REF_SECONDS` は空なら checkpoint の既定値（RF 40 / MeanFlow 4 steps、推奨参照長）を使います。v3 に戻す場合は `IRODORI_TTS_NUM_STEPS=40` と `IRODORI_TTS_MAX_REF_SECONDS=30` も指定してください。

### 短文の尺上限

v4.1-Small / MF は香川先生の参照音声で「はい。」のような短文の尺を 3 秒前後と長く予測し、余った時間に意味のない発話を続けます（Whisper で確認）。予測尺を 0.5 倍程度まで縮めると正しく読むため、サーバーは推定モーラ数から尺の上限を決めて `max_seconds` に渡します。

```text
上限秒 = IRODORI_TTS_DURATION_CAP_BASE_SECONDS (0.7)
       + 推定モーラ数 / IRODORI_TTS_DURATION_CAP_MORA_PER_SECOND (6.0)   （下限 1.2 秒）
```

推定モーラ数はかな 1、漢字 1.7、英数字 1.5、文中の句読点 2 で数えます。中・長文では上限が予測尺より長いので効きません。`IRODORI_TTS_DURATION_CAP_MORA_PER_SECOND=0` で無効になります。

## Irodori-TTS 本体の更新

```bash
cd /home/kyano/workspace/ACE/ace_kagawa/Irodori-TTS
git pull --ff-only origin main
uv sync --extra cu128
```

v4 系のコードは `transformers>=5.12.1` を要求します。`uv sync` は `--extra cu128` を付けないと CUDA 版 torch が外れるので注意してください。

## 読み上げの正規化

モデルに渡す前に `normalize_speech_input()` が数字・日付・単位をひらがな読みに変換します。主な規則:

- 助数詞の音便: `14時`→じゅうよじ、`19時`→じゅうくじ、`4人`→よにん、`4年生`→よねん生、`1週間`→いっしゅうかん、`3ヶ月`→さんかげつ、`600点`→ろっぴゃくてん、`100個`→ひゃっこ
- 固有の読み: `1つ`→ひとつ、`20歳`→はたち、`1泊2日`→いっぱくふつか
- 日付: `4/1（火）`、`2026/4/1`、`4月1日`→ついたち。月を伴わない `1日` と `1日目`・`1日間` は「いちにち」
- 範囲: `2〜3人`→ふたりからさんにん
- 金額・単位: `¥5,000`→ごせんえん、`1000万円`→いっせん万円、`36.5℃`、`100m`、`5km`、`10kg`
- `5G`・`3D`・`RTX4090` のような英字に接する数字は変換しない

`M/D` 形式は日付として読むため、`1/3` のような分数は「いちがつみっか」になります。分数は `3分の1` と書かれていれば「さんぶんのいち」と読みます。

Tokkio 側（`infra/tokkio/tokkio_llm.py` の `SpeechSegmentBuffer`）は LLM のストリームを文単位で TTS に送ります。`1.5` や `250,000` の途中で切ると読みが壊れるため、数字の間の `.` と `,` は区切りとして扱いません。
