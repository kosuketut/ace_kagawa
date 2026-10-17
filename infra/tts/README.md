# Irodori-TTS HTTP service (kagawa)

`irodori_server.py` は Tokkio の `IrodoriTTSService` から呼ばれる host-side TTS です。`/v1/audio/speech` で 16 kHz mono PCM / WAV を返し、`/healthz` で使用中の checkpoint と参照音声の有無を返します。

## モデル

既定は `Aratako/Irodori-TTS-v4.1-Small`（MIT）です。`IRODORI_TTS_HF_CHECKPOINT` で差し替えます。 合成には既定で speech.cpp 版を使います（後述）。checkpoint の切り替えは PyTorch 版（`IRODORI_TTS_BACKEND=python`）での話です。

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

16 文字以下の文に掛ける `IRODORI_TTS_SHORT_DURATION_SCALE` は 2026-10-08 に 0.8 から 1.0 に変えました。短く中身の少ない文は上の上限で縮むので、0.8 は「東京工科大学の学長です。」のような詰まった短文だけを約 9.6 モーラ/秒まで速めていました（長文は 6〜8 モーラ/秒）。1.0 でも Whisper の聞き取りは同等でした（各 6 回）。

```text
上限秒 = IRODORI_TTS_DURATION_CAP_BASE_SECONDS (0.7)
       + 推定モーラ数 / IRODORI_TTS_DURATION_CAP_MORA_PER_SECOND (6.0)   （下限 1.2 秒）
```

推定モーラ数はかな 1、漢字 1.7、数字 1.5、英大文字は文字名の長さ（`W`・`X` 4、`H`・`R`・`Z` 3、その他 2）、英小文字 1、文中の句読点 2 で数えます。中・長文では上限が予測尺より長いので効きません。`IRODORI_TTS_DURATION_CAP_MORA_PER_SECOND=0` で無効になります。

## speech.cpp で合成する（既定、2026-10-16〜）

unit は `IRODORI_TTS_BACKEND=speechcpp` で、合成だけを [speech.cpp](https://github.com/nyosegawa/speech.cpp) v0.8.7 の常駐 worker（Vulkan、F16）に任せます。HTTP の応答形式、読みの正規化、短文キャッシュ、固定挨拶、後処理（16 kHz・トリム・フェード）は PyTorch 版と同じです。モデルは同じ v4.1-Small です。

### 12 steps（2026-10-17〜）

unit は `IRODORI_TTS_NUM_STEPS=12` で、checkpoint 既定の 40 steps より少ない steps で合成します。2026-10-17 に香川先生の声で 40・24・16・12・10・8 steps を比べ、試聴で 12 steps に決めました（`docs/audits/tts-alternatives-2026-10-17.md`）。

| steps | 1 文の生成（中央値、5 文） | 26 文での生成（中央値） | 本人らしさ（CAMPPlus、40 との差） | 言い誤り（26 文 × 5 回） |
| --- | ---: | ---: | ---: | ---: |
| 40 | 0.586 秒 | 0.548 秒 | — | 11/125 |
| 16 | 0.285 秒 | 0.264 秒 | +0.001 | 9/125 |
| **12** | **0.238 秒** | **0.213 秒** | +0.007 | 10/125 |
| 10 | 0.216 秒 | — | −0.015 | — |
| 8 | 0.191 秒 | — | −0.087 | — |

- 12 より下げない。10 steps から本人らしさが下がり始め、8 steps で大きく崩れる（読みは 8 steps でも保たれる）。
- 音声の長さは steps によらず同じ。速くなるのは同じ長さの音声を少ない計算で作るため。
- 言い誤りの大半は「いいえ。」「ええ。」で、40 steps でも同じように崩れる（下の「短文は尺を固定する」）。
- 固定挨拶の PCM はファイルなので、作り直さない限り 40 steps の音声のまま。短文キャッシュはメモリ上だけなので、起動時に 12 steps で作り直される。

### PyTorch 版との比較と設定（2026-10-16、40 steps の時点）

本番と同じ 33.04 秒の参照音声で比べた結果（`docs/audits/speech-cpp-tts-benchmark-2026-10-16.md` の 8 節）は次のとおりです。このサービス経由で同じ seed で作った音声は、比較・試聴に使った音声とバイト単位で同じでした（短文と通常文で確認）。

| | PyTorch 版 | speech.cpp 版 |
| --- | ---: | ---: |
| 生成完了（中央値） | 1.008 秒 | 0.578 秒 |
| 最初の音声まで | 1.008 秒 | 0.510 秒 |
| GPU メモリ | 6.39 GiB | 2.21 GiB |
| 本人らしさ（CAMPPlus cosine） | 0.721 | 0.692 |

| 環境変数 | 値（unit） | 意味 |
| --- | --- | --- |
| `IRODORI_TTS_SPEECHCPP_BIN` | `/data/ACE/speech-cpp/runtime/v0.8.7/speech` | 公式 Linux x64 Vulkan リリース。sha256 `86efdef9…` |
| `IRODORI_TTS_SPEECHCPP_MODEL` | `/data/ACE/speech-cpp/models/Irodori-TTS-859M-v4.1-F16.gguf` | sha256 `f334932b…` |
| `IRODORI_TTS_SPEECHCPP_VOICE` | `/data/ACE/irodori/speech-cpp/kagawa_voice_ref_48k_mono.voice.gguf` | 参照音声 33.04 秒から CPU で作った voice。sha256 `4b6c2cd4…` |
| `IRODORI_TTS_SPEECHCPP_DEVICE` | `Vulkan1` | Vulkan の GPU 名 |
| `IRODORI_TTS_SPEECHCPP_GPU_UUID` | `GPU-d4c17c05-…` | worker がこの GPU に載らなければ起動を止める |
| `IRODORI_TTS_SPEECHCPP_THREADS` | 8 | |
| `IRODORI_TTS_SPEECHCPP_TIMEOUT_SECONDS` | 120 | 応答がなければ worker を止め、次の要求で起動し直す |

注意点:

- **GPU の選び方が PyTorch 版と違う**。`CUDA_VISIBLE_DEVICES` は Vulkan に効かず、このマシンでは Vulkan の番号が逆（`Vulkan1` が nvidia-smi の GPU 0）。起動時に `nvidia-smi` で worker の GPU UUID を照合し、`/healthz` の `speechcpp_gpu_uuid` に出す。
- **voice ファイルは CPU で一度だけ作る**。GPU で作った voice では本人らしさが 0.692 から 0.675 に下がった。CPU では約 100 秒かかるので、サービスは作らない。無いときは `install_irodori_tts_service.sh` が作る。手で作るなら `speech voice <MODEL> <参照WAV> <VOICE> --device cpu`。参照音声を変えたら voice も作り直す。
- **短文は尺を固定する**。speech.cpp には `max_seconds` がないため、16 文字以下の文には上の尺上限を `seconds` として渡す。2026-10-16 の確認では、短い返事 16 種類すべてで speech.cpp の予測尺が上限より長く、上限なしでは多くの文で意味のない発話が付いた。上限 1.2 秒になる「いいえ。」「ええ。」は聞き取りが崩れることがある。
- **透かし（silentcipher）は付かない**。`/healthz` の `watermark` が `false` になる。LoRA にも対応していない（設定すると起動しない）。
- worker の stderr はサービスのログ（journald）に出る。worker が落ちたら、次の要求で起動し直す。

PyTorch 版に戻すには、unit の `IRODORI_TTS_BACKEND=python` にして入れ直します。

## サービスの登録・設定の反映

```bash
infra/tts/install_irodori_tts_service.sh            # unit を /etc に入れて daemon-reload
infra/tts/install_irodori_tts_service.sh --restart  # 稼働中のサービスにも反映（話し中の音声は途切れます）
```

`--restart` を付けない場合、稼働中のサービスは次の起動まで古い設定のままです。

## Irodori-TTS 本体の更新

```bash
cd /home/kyano/workspace/ACE/ace_kagawa/Irodori-TTS
git pull --ff-only origin main
uv sync --extra cu128
```

v4 系のコードは `transformers>=5.12.1` を要求します。`uv sync` は `--extra cu128` を付けないと CUDA 版 torch が外れるので注意してください。

## 読み上げの正規化

モデルに渡す前に `normalize_speech_input()` が数字・日付・単位をひらがな読みに変換します。主な規則:

- 助数詞の音便: `14時`→じゅうよじ、`19時`→じゅうくじ、`4人`→よにん、`4年生`→よねん生、`1週間`→いっしゅうかん、`3ヶ月`→さんかげつ、`600点`→ろっぴゃくてん、`100個`→ひゃっこ、`21世紀`→にじゅういっせいき、`10種類`→じゅっしゅるい、`1000本`→せんぼん
- `10%`→じゅっぱーせんと、`10km`→じゅっキロメートル のように、`じゅう` は無声音で始まる単位の前で促音化します
- 小数: `1.5倍`→いってんご倍、`8.5`→はってんご（`6.5`→ろくてんご）。`v4.1`→vよんてんいち
- 大きな数: `10兆円`→じゅっちょう円、`90京`→きゅうじゅうけい（`京都` などは対象外）
- 固有の読み: `1つ`→ひとつ、`20歳`→はたち、`1泊2日`→いっぱくふつか
- 時刻: `09:05`→くじごふん（先頭の 0 は読まない）、`10:00〜12:00まで`→じゅうじからじゅうにじまで（「まで」を重ねない）。`質問1：10月` のように見出し語（質問・ポイント・ステップ・問題など）の後のコロンは区切りとして読みます
- 序数: 数字の前の `第` は「だい」とかなで書きます（`第1月曜日`→だいいち月曜日。「第いち」と混ぜるとモデルが言い崩れるため。2026-10-08 の試聴で 1/3 → 3/3）
- 日付: `4/1（火）`、`2026/4/1`、`2026-07-19`、`4月1日`。`19日`/`29日`→じゅうくにち/にじゅうくにち。月の後の「ついたち」は `はちがつ ついたち` と空白を入れます（つが続くとモデルが言いよどむため。2026-10-08 の試聴で 8/12 → 6/6）。月を伴わない `1日` は「いちにち」、ただし `1日（土）` や `今月の1日` は「ついたち」
- 範囲・符号: `2〜3人`→ふたりからさんにん、`2,3人`→に、さんにん、`9時-17時`→くじからじゅうしちじ、`-5℃`→マイナスごど、`前年比 +5%`→前年比 プラスごぱーせんと、`16:9`→じゅうろくたいきゅう
- 住所・郵便番号: `〒192-0982`→ゆうびんばんごういちきゅうにのぜろきゅうはちに、`片倉町1404-1`→片倉町せんよんひゃくよんのいち、`西蒲田5-23-22`→西蒲田ごのにじゅうさんのにじゅうに、`1丁目`→いっちょうめ
- 単位: `¥5,000`→ごせんえん、`36.5℃`、`100m`、`10kg`、`400Gb/s`→よんひゃくギガビット毎秒、`kW`・`V`・`Hz`・`dB`・`fps`
- 記号: 全角の英数字・記号（`１０：３０`、`ＡＩ`）は半角に直してから読みます。`Q&A`→キューアンドエー、`説明&寮見学`→説明と寮見学、`→`→「、」、`※` は読まない。`よろしくね〜。` の波ダッシュは長音
- URL は読みません。`…をご覧ください` のように名詞として使われていれば「ウェブサイト」に置き換え、括弧内の URL は削除します。メールアドレスはそのまま渡します
- 振り仮名: `片倉町（かたくらまち）` は読みを 1 回だけ読みます
- `5G`・`3D`・`RTX4090` のような英字に接する整数は変換しない

`M/D` 形式は日付として読むため、`1/3` のような分数は「いちがつみっか」になります。分数は `3分の1` と書かれていれば「さんぶんのいち」と読みます。

Tokkio 側（`infra/tokkio/tokkio_llm.py` の `SpeechSegmentBuffer`）は LLM のストリームを文単位で TTS に送ります。`1.5` や `250,000` の途中で切ると読みが壊れるため、数字の間の `.` と `,` は区切りとして扱いません。符号・箇条書き・矢印の扱い（`-5` の `-` を箇条書きとして消さない、` + `→プラス、`→`→「、」）はサーバー側と同じ規則です。

## 音声の後処理

モデル出力（48 kHz）は応答前に次の順で処理します。

1. 帯域制限つきリサンプル（`scipy.signal.resample_poly`。scipy がない環境では窓付き sinc の FIR）で 16 kHz に変換します。以前の線形補間は 8〜24 kHz の成分を折り返し、摩擦音帯域（4〜8 kHz）に本来の音より 5〜12 dB 低いだけの雑音を足していました。
2. モデルが前後に付ける部屋ノイズの区間（中央値で前 0.23 秒・後 0.42 秒）を詰めます。発話の前は 60 ms、文末（`。！？`）の後は 300 ms、それ以外は 120 ms を残します。文中の間は削りません。Tokkio は文ごとに TTS を呼ぶので、これで話し始めが約 0.18 秒早くなり、文と文の間が 0.65 秒から約 0.36 秒（モデルが文中に入れる間 0.44 秒に近い値）になります。
3. 両端に 10 ms のフェードをかけ、切れ目のクリック音を消します。
4. -0.5 dB のゲインをかけてから四捨五入で 16 bit にします（以前は透かし処理でピークが 1.003 まで出て、ほぼ毎回クリップしていました）。

固定挨拶の PCM（`IRODORI_TTS_FIXED_OPEN_CAMPUS_GREETING_PCM`）も応答時に同じ詰め・フェード・ゲインを通します（ファイル自体は書き換えません）。

| 環境変数 | 既定 | 内容 |
| --- | --- | --- |
| `IRODORI_TTS_TRIM_ENABLED` | `true` | 前後のノイズ区間を詰める |
| `IRODORI_TTS_TRIM_LEAD_MS` | `60` | 発話前に残す長さ |
| `IRODORI_TTS_SENTENCE_TRAIL_MS` | `300` | 文末の後に残す長さ |
| `IRODORI_TTS_CLAUSE_TRAIL_MS` | `120` | 文末以外の後に残す長さ |
| `IRODORI_TTS_FADE_MS` | `10` | 両端のフェード |
| `IRODORI_TTS_OUTPUT_GAIN_DB` | `-0.5` | 出力ゲイン |

読み上げる文字がない入力（`！` だけ、URL だけなど）は、モデルを呼ばず空の音声を返します。空白だけの入力は 400 です。

## ElevenLabs に切り替える

`elevenlabs_server.py` は同じ `/v1/audio/speech` 契約で ElevenLabs を呼ぶ中継サービス（port 8022）です。Tokkio 側のコードは変えず、`infra/tokkio/.env` の向き先だけを切り替えます。

- Eleven v4 / v4 Turbo は、pipecat の `ElevenLabsTTSService` が使う TTS WebSocket（`stream-input` / `multi-stream-input`）に非対応で、HTTP 400 `unsupported_model` になります（2026-09-29 確認）。そのため中継サービスは HTTP ストリーミング（`/v1/text-to-speech/{voice_id}/stream`）を使います。
- v4 系では ElevenLabs の日本語正規化オプション（`apply_language_text_normalization`）が使えません。生の数字は誤読することがあるため（`703,000円` → 73万円）、中継サービスでも `normalize_speech_input()` を通します。
- 既定モデルは `eleven_v4_turbo`。実測で最初の音声まで 0.2〜0.4 秒、`eleven_v4` は 0.8〜1.5 秒でした。

```bash
# 1. infra/tokkio/.env に voice id を設定（API キーは TOKKIO_ELEVENLABS_API_KEY を共用）
TOKKIO_ELEVENLABS_VOICE_ID=<ElevenLabs の voice id>

# 2. 中継サービスを登録して起動
sudo install -m 0644 infra/tts/ace-elevenlabs-tts.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl start ace-elevenlabs-tts
curl -s http://127.0.0.1:8022/healthz   # voice id か API キーが未設定なら 503

# 3. infra/tokkio/.env で Tokkio の TTS 向き先を中継サービスに変更し、reapply
TOKKIO_IRODORI_TTS_BASE_URL=http://<TOKKIO_APP_HOST_IPV4_ADDR>:8022
TOKKIO_IRODORI_TTS_HEALTH_URL=http://127.0.0.1:8022/healthz
TOKKIO_IRODORI_TTS_SERVICE=ace-elevenlabs-tts.service
```

Irodori に戻すときは 3 の 3 行を元の値（port 8021、`ace-irodori-tts.service`）に戻して reapply します。
