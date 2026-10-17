# TTS 高速化の選択肢: 調査と実機比較（Irodori 以外も含む）

実施日: 2026-10-17 / 対象: `ace_kagawa`

すべて香川先生の参照音声からのゼロショットクローンで比べた。本番の `.env`、systemd unit、Tokkio 構成は変更していない。計測は両 GPU が空いている時間に行った（本番の Irodori・vLLM は開始時点ですでに停止しており、こちらでは起動も停止もしていない）。

## 結論

> **2026-10-17 追記: 試聴のうえ Small/12 に決定した**（26 文の追加検証は 2.8 節）。unit に `IRODORI_TTS_NUM_STEPS=12` を追加済み。`/etc` への導入は未実施。

**別モデルに替えるより、今の Irodori v4.1-Small の steps を 40 から 16 に減らすのがよい。** 同じ speech.cpp・同じ voice ファイルのまま設定 1 行で、1 文の生成完了が 0.586 秒 → 0.285 秒（約 2.1 倍速）になる。本人らしさ・読みの自動指標は 40 steps と差がなかった。

| 順位 | 構成 | 生成完了 p50 | 現行比 | 本人らしさ（全長参照 / 別録音） | 読み（CER / 崩れ） | 判断 |
| --- | --- | ---: | ---: | --- | --- | --- |
| — | Irodori Small/40（現行） | 0.586 秒 | 1.0 倍 | 0.692 / 0.622 | 0.034 / 0 回 | — |
| **1** | **Irodori Small/16** | **0.285 秒** | **2.1 倍** | 0.693 / 0.623（差なし） | 0.034 / 0 回 | **推奨。試聴で確認してから切り替え** |
| 2 | Irodori Small/12 | 0.238 秒 | 2.5 倍 | 0.699 / 0.630（差なし） | 0.034 / 0 回 | 26 文の追加検証（2.8 節）でも Small/40 と同等。試聴で問題なければ採用してよい。10 steps から本人らしさが落ち始めるので、これより下げない |
| 3 | Irodori MF/4 | 0.170 秒 | 3.4 倍 | 0.671 / 0.595（やや低い） | 0.054 / 2 回 | 最速。短文「少々お待ちください」を 10 回中 2 回言い誤る |
| 4 | Drama 3（speech.cpp） | 2.6 秒（最初の音声 0.14 秒） | 0.22 倍 | 0.718〜0.727 / 0.667〜0.678 | 0.071〜0.121 / 1〜2 回 | 今のサーバーでは遅い。途中から流す改修をしても、短文で無音・暴走・参照文の読み上げが出る |
| 5 | Qwen3-TTS 1.7B | 7.7〜8.9 秒 | 0.07 倍 | 0.709〜0.729 / 0.647〜0.664 | 0.039〜0.046 / 1 回 | 声と読みは互角だが、今の実装では実時間より遅い。speech.cpp では香川先生の声を使えない |
| × | Irodori Large/16（speech.cpp） | 0.576 秒 | 1.0 倍 | 0.611 / 0.562（低い） | 0.087 / 3 回 | 速くならず、本人らしさが低い。短文で雑音のような音声が 2 回、数字の言い誤りが 1 回 |
| × | Fun-CosyVoice3-0.5B | 3.6 秒 | 0.16 倍 | （参考にならない、後述） | 0.52〜0.68 / 46〜48 回 | 日本語の漢字の読みが崩れる。除外 |

- 「崩れ」は 50 回中、Whisper の CER が 0.2 を超えた回数。本人らしさは CAMPPlus 埋め込みの cosine（本人の別録音 4 秒片は平均 0.764、別人は 0.105）。
- **自動指標が同じでも、聞いて同じとは限らない。** steps を減らすと、話者が崩れる前にざらつきや濁りが出ることが多い。CAMPPlus も Whisper もそれは測らない。聞き比べページのブラインド比較（12 組）で決める。
- 体感で縮むのは**最初の 1 文だけ**。2 文目以降は 1 文目の再生中に作られる。LLM の最初の 1 文は 0.17〜0.56 秒（1.2 節）なので、Small/16 にすると応答開始が約 0.3 秒早くなる。
- 聞き比べページ: `/data/ACE/speech-cpp/20261017-compare/index.html`（ブラインド比較 12 組、全 18 構成 × 5 文 × seed 1〜3、本人の録音）。

### speech.cpp だけで比べた場合（ボイスクローン必須）

speech.cpp（v0.8.8 時点）で参照音声からクローンできるのは **Irodori（v4.1-Small、v4.1-Small-MF、v4-Large）と Drama 3 だけ**。Qwen3-TTS は既成の 9 話者専用で、クローンには対応していない。

| 構成 | 生成完了 p50（今のサーバー） | 最初の音声 p50（サーバー改修後） | 本人らしさ（全長 / 別録音） | CER | 崩れ・暴走 | GPU メモリ |
| --- | ---: | ---: | --- | ---: | --- | ---: |
| Irodori Small/40（現行） | 0.586 秒 | 0.517 秒 | 0.692 / 0.622 | 0.034 | 0 / 0 | 2.2 GiB |
| **Irodori Small/16** | **0.285 秒** | 0.217 秒 | 0.693 / 0.623 | 0.034 | 0 / 0 | 2.2 GiB |
| Irodori Small/12 | 0.238 秒 | 0.169 秒 | 0.699 / 0.630 | 0.034 | 0 / 0 | 2.2 GiB |
| Irodori MF/4 | 0.170 秒 | 0.068 秒 | 0.671 / 0.595 | 0.054 | 2 / 0 | 2.2 GiB |
| Irodori Large/16 | 0.576 秒 | 0.512 秒 | 0.611 / 0.562 | 0.087 | 3 / 0 | 7.1 GiB |
| Irodori Large/40 | 1.360 秒 | 1.302 秒 | 0.645 / 0.583 | 0.039 | 1 / 0 | 7.1 GiB |
| Drama 3（参照 33 秒） | 2.672 秒 | 0.147 秒 | 0.727 / 0.678 | 0.071 | 2 / 3 | 21.6 GiB |
| Drama 3（参照 10.3 秒） | 2.625 秒 | 0.139 秒 | 0.718 / 0.667 | 0.121 | 1 / 1 | 21.6 GiB |

- **今のサーバーでは「生成完了」がそのまま初音になる**（チャンクを集めてから返すため、1.1 節）。「最初の音声」は、サーバーを worker のチャンクを届いた順に流す作りに直した場合の値。
- 崩れ＝CER が 0.2 を超えた回数（50 回中）、暴走＝音声の長さが現行の 2 倍を超えた回数。
- **Drama 3 は途中から流せば最初の音声が 0.14 秒と速いが、勧めない**。文全体を作るのに 2.6 秒かかり（実時間の 2.6 倍の速さなので、流しながらなら途切れはしない）、短文で不安定だった。「少々お待ちください。」で 17 秒の無音に近い音声（seed 2）、数字の文で 23 秒の暴走（seed 10）、参照 10 秒版では参照音声の書き起こし「こんにちは。東京工科大学学長の香川豊です…」をそのまま読んだ回（seed 4）があった。中・長文 3 種は CER 0〜0.008 で正確だった。
- 改修後で比べても、Drama 3 は Irodori Small/16 より 0.08 秒早く話し始めるだけで、文を作り終えるのは 2.3 秒遅い。今のサーバーのままなら、話し始めが 2.3 秒遅い。
- 「最初の音声」の列は、Drama 3 にサーバー改修が効くかを見るためのもの。Irodori は文全体を一度に作ってから音声に戻すので、改修で縮むのは約 0.07 秒だけで、前後の無音トリムを最初のチャンクで行う改修も要る。Irodori の高速化の手段にはならない。
- Qwen3-TTS を speech.cpp で香川先生の声にするには、香川先生の声で微調整し、それを speech.cpp の形式に変換する必要がある。公式の微調整の出力は CustomVoice と同じ構造（`tts_model_type` を `custom_voice` に、話者の埋め込みを `codec_embedding` の 3000 行目に入れ、`speaker_encoder` を捨てる。`finetuning/sft_12hz.py:134-158`）なので、変換スクリプト（公式 CustomVoice 以外を assert で拒否する）を改造すれば載る見込みはある。未試行。学習に使える本人の音声は 33 秒（約 7 文）しかない。

## 1. 現状の遅延

### 1.1 TTS（本番）

- speech.cpp・v4.1-Small・40 steps・33.04 秒の参照で、1 文の生成完了は中央値 0.586 秒（今回）、0.578 秒（2026-10-16）。
- **本番の初音＝生成完了**。Tokkio は `stream=true` で受け取り、届いた順に再生する（`infra/tokkio/tokkio_irodori_tts.py:83-128`）。ところがサーバーは speech.cpp worker のチャンクを最後まで集めてから返す（`infra/tts/irodori_server.py:1491-1506`）。progressive 分割も無効（`/healthz` の `progressive_stream_enabled: false`）。
- worker の最初のチャンクは完了より約 0.07 秒早い（初音 0.517 秒 / 完了 0.586 秒）。サーバーで逐次中継しても縮むのはこの差だけ。Irodori は文全体を一度に作る方式なので、大きくは縮まない。

### 1.2 LLM の最初の 1 文

vLLM（`ace-vllm-llama31-swallow-8b`）へ直接、ストリーミングで 3 回測った。RAG・controller・ASR を通さない値なので、本番の値はこれより長い。

| 質問 | 最初のトークン | 最初の読点 | 最初の文末 |
| --- | ---: | ---: | ---: |
| 東京工科大学への行き方を教えてください。 | 0.029 秒 | 0.200 秒 | 0.426 秒 |
| 先生の研究について簡単に教えてください。 | 0.027 秒 | 0.052 秒 | 0.171 秒 |
| オープンキャンパスはいつですか？ | 0.027 秒 | 0.196 秒 | 0.558 秒 |

Tokkio の `SpeechSegmentBuffer` は、文末か、28 字を超えた後の区切りで TTS に送る（`infra/tokkio/tokkio_llm.py:397-451`）。

## 2. 実機比較（2026-10-17）

### 2.1 全構成の結果

5 文（短文・専門・アクセス・数字・長文）× seed 1〜10 の 50 回ずつ。ウォームアップは各文 2 回。入力はすべて本番の読み正規化（`normalize_speech_input()`）を通した文。音声は本番と同じ後処理（16 kHz、前後の無音トリム、フェード、−0.5 dB）を通してから評価した。

| 構成 | 生成完了 p50 | p90 | 初音 p50 | 本人らしさ（全長参照） | 現行との差 [95%CI] | 別録音 | 現行との差 [95%CI] | CER | 崩れ | 尺が現行の 2 倍超 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Irodori Small/40（現行） | 0.586 | 0.786 | 0.517 | 0.692 | — | 0.622 | — | 0.034 | 0/50 | 0/50 |
| Irodori Small/24 | 0.385 | 0.530 | 0.316 | 0.694 | +0.002 [−0.014, +0.018] | 0.623 | +0.001 [−0.015, +0.017] | 0.034 | 0/50 | 0/50 |
| Irodori Small/16 | 0.285 | 0.403 | 0.217 | 0.693 | +0.001 [−0.015, +0.018] | 0.623 | +0.001 [−0.015, +0.018] | 0.034 | 0/50 | 0/50 |
| Irodori Small/12 | 0.238 | 0.348 | 0.169 | 0.699 | +0.007 [−0.009, +0.023] | 0.630 | +0.008 [−0.008, +0.024] | 0.034 | 0/50 | 0/50 |
| Irodori Small/10 | 0.216 | 0.309 | 0.145 | 0.677 | −0.015 [−0.031, +0.001] | 0.608 | −0.013 [−0.030, +0.003] | 0.034 | 0/50 | 0/50 |
| Irodori Small/8 | 0.191 | 0.277 | 0.121 | 0.605 | −0.087 [−0.113, −0.062] | 0.524 | −0.097 [−0.125, −0.070] | 0.036 | 0/50 | 0/50 |
| Irodori MF/8 | 0.215 | 0.308 | 0.113 | 0.684 | −0.008 [−0.029, +0.011] | 0.596 | −0.025 [−0.044, −0.008] | 0.039 | 1/50 | 0/50 |
| Irodori MF/4 | 0.170 | 0.259 | 0.068 | 0.671 | −0.021 [−0.046, +0.001] | 0.595 | −0.026 [−0.047, −0.007] | 0.054 | 2/50 | 0/50 |
| Qwen3-TTS 1.7B（ICL） | 7.683 | 15.091 | 7.683 | 0.709 | +0.017 [−0.039, +0.067] | 0.664 | +0.042 [−0.012, +0.090] | 0.046 | 1/50 | 0/50 |
| Qwen3-TTS 1.7B（x-vector） | 8.915 | 15.644 | 8.915 | 0.729 | +0.037 [+0.011, +0.062] | 0.647 | +0.026 [+0.002, +0.049] | 0.039 | 1/50 | 0/50 |
| Qwen3-TTS 0.6B（ICL） | 8.508 | 16.432 | 8.508 | 0.735 | +0.043 [−0.019, +0.096] | 0.687 | +0.066 [+0.012, +0.110] | 0.206 | 10/50 | 6/50 |
| Qwen3-TTS 0.6B（x-vector） | 8.792 | 16.606 | 8.792 | 0.710 | +0.018 [−0.005, +0.040] | 0.597 | −0.024 [−0.046, −0.004] | 0.065 | 2/50 | 0/50 |
| Irodori Large/16（v0.8.8） | 0.576 | 0.876 | 0.512 | 0.611 | −0.081 [−0.125, −0.042] | 0.562 | −0.060 [−0.096, −0.027] | 0.087 | 3/50 | 0/50 |
| Irodori Large/40（v0.8.8） | 1.360 | 2.086 | 1.302 | 0.645 | −0.047 [−0.079, −0.020] | 0.583 | −0.038 [−0.066, −0.014] | 0.039 | 1/50 | 0/50 |
| Drama 3 Q4_K（参照 33 秒、v0.8.8） | 2.672 | 4.980 | 0.147 | 0.727 | +0.035 [−0.007, +0.070] | 0.678 | +0.057 [+0.019, +0.089] | 0.071 | 2/50 | 3/50 |
| Drama 3 Q4_K（参照 10.3 秒、v0.8.8） | 2.625 | 4.646 | 0.139 | 0.718 | +0.026 [+0.000, +0.051] | 0.667 | +0.046 [+0.023, +0.068] | 0.121 | 1/50 | 1/50 |
| CosyVoice3（参照 24.6 秒） | 3.560 | 6.790 | 3.544 | 0.829 | +0.137 [+0.118, +0.154] | 0.676 | +0.054 [+0.037, +0.072] | 0.517 | 46/50 | 0/50 |
| CosyVoice3（参照 10.3 秒） | 3.643 | 7.942 | 2.950 | 0.785 | +0.093 [+0.068, +0.117] | 0.657 | +0.036 [+0.013, +0.058] | 0.682 | 48/50 | 1/50 |

- 時間は秒。95%CI は平均の差のブートストラップ（10,000 回）。50 音声は 5 文 × seed 10 で独立ではないため、やや狭めに出る。
- CER は Whisper large-v3-turbo の聞き取りと原文の文字の差。時刻の表記差（10:00 と 10時）も数えるので、数字の文はどの構成も 0.17 前後になる。比べるのは構成間の差。
- 「尺が現行の 2 倍超」は、同じ文での現行の音声長（中央値）の 2 倍を超えた回数。文の後に意味のない発話が続いたものの目安。

### 2.2 Irodori の steps

- **16 steps までは指標が 40 steps と変わらない**。本人らしさ、別録音との近さ、CER、崩れの回数とも差がなく、95%CI はすべて 0 をまたぐ。速さは 24 steps で 1.5 倍、16 steps で 2.1 倍、12 steps で 2.5 倍。
- **10 steps から本人らしさが下がり始め、8 steps で大きく崩れる**（−0.087）。読み（CER）は 8 steps でも保たれ、声の近さだけが落ちる。12 steps は境目の 2 段手前なので、16 steps を既定に推す。
- 音声の長さは steps によらず同じだった（尺の予測は steps に依存しない）。速くなったのは、同じ長さの音声を少ない計算で作れたため。
- **MF は同じ速さの Small より劣る**。MF/8（0.215 秒）は Small/10〜12 とほぼ同じ速さで、別録音との近さが −0.025 低く、短文の言い誤りが 1 回あった。MF/4 は「少々お待ちください」を「少々お待たせください」「早々お待ちいただきまさい」と 10 回中 2 回言い誤った。10/16 の測定でも同じ seed（2 と 5）で同じ言い誤りが出ていた。
- 現行（Small/40）と MF/4 の再計測は、10/16 の値（0.578 秒 / 0.168 秒、本人らしさ 0.692 / 0.671、CER 0.034 / 0.054）を ±0.01 で再現した。

### 2.3 Irodori v4-Large と Drama 3（speech.cpp v0.8.8）

- **Large は速くも良くもならない**。16 steps で 0.576 秒（現行の Small/40 と同じ）、40 steps で 1.360 秒。本人らしさは Small/40 より 0.047〜0.081 低く、16 steps は CER も 0.087 に上がった。崩れ 3/50 の内訳は、短文で雑音のような大音量の音声が 2 回（RMS 0.23〜0.32。Whisper は「そうそう」「ご視聴ありがとうございました」と誤認識）、数字の文の言い誤り（「はちがつ ついたち」→「8月、ついついたち」）が 1 回。10/16 の 10 秒参照での測定（Large F16/40 1.317 秒、本人らしさ 0.661）と同じ傾向。重みは Gemma Terms of Use。
- **Drama 3**（Fish Audio、35B の MoE で動くのは 3B、Q4_K 20.8 GB）は全層が GPU に載った（CPU に回す旨のログなし、GPU メモリ 21.6 GiB）。voice ファイルは参照音声と書き起こしから CPU で作り、33 秒・10.3 秒とも約 30 秒だった。書き起こしは 2.4 節と同じ Whisper の結果を直したもの。
- Drama 3 の本人らしさは現行より高く出た（全長 +0.026〜+0.035、別録音 +0.046〜+0.057）が、2.4 節のとおりこの指標は収録条件も拾うので「同等以上」と読む。
- ライセンスは Fish Audio Community License 1.0。年間売上 50 万ドルまで商用利用可、他者向けの音声・ボイスクローンのサービスには別契約が要る（[speech.cpp の Drama 3 文書](https://github.com/nyosegawa/speech.cpp/blob/main/docs/models/drama3.md)）。大学のアバターがどちらに当たるかは確認していない。

### 2.4 Qwen3-TTS（Qwen/Qwen3-TTS-12Hz-1.7B-Base、0.6B-Base）

- **1.7B は声と読みで Irodori と互角**。CER 0.039〜0.046、崩れ 1/50。本人らしさは 0.709〜0.729 で現行より高い値が出た。ただし CosyVoice3 が崩れた日本語で 0.829 という本人の録音（0.764）以上の値を出したことから、この指標は声の同一性だけでなく収録条件の近さも拾っている。Qwen の高い値は「同等以上」と読み、「より本人らしい」とは読まない。
- **話す速さが 1〜2 割速い**。無音トリム後の長さは専門の文で 5.80〜6.25 秒（現行 6.97 秒）、長文で 10.79〜10.97 秒（同 12.30 秒）。自然に聞こえるかは試聴で確認する。
- **今の実装では遅すぎる**。公式パッケージ（qwen-tts 0.1.1、PyTorch、SDPA、bfloat16）で 1 文 7.7〜8.9 秒、RTFx 0.7（実時間より遅い）。公式パッケージは生成のストリーミングに対応していない（`non_streaming_mode` は入力側の模擬だけ）。flash-attn は入れていない。
- **速く動かす経路が、香川先生の声では確認できていない**。speech.cpp の Qwen3-TTS は既成の 9 話者専用で、クローンは非対応（[文書](https://github.com/nyosegawa/speech.cpp/blob/main/docs/models/qwen3-tts.md)）。公式の微調整で単一話者モデルを作り、それを speech.cpp の GGUF に変換できるかは未確認。vLLM-Omni 経由の初音も未計測。試すなら 2〜3 日の見込みで、結果は保証できない。
- **0.6B は使えない**。ICL では短文の後に意味のない発話が続き（50 回中 6 回で尺が 2 倍超、最長 28.7 秒）、CER 0.206。x-vector では CER 0.065。
- ICL は参照の書き起こしが必要。33 秒の参照を Whisper で書き起こし、明らかな誤り（劇群→激務）だけ直して使った（`ref_transcript.txt`）。

### 2.5 Fun-CosyVoice3-0.5B-2512

- **日本語の漢字の読みが崩れる**。「私の専門分野は材料強度学…」が「私の先輩は、文語、敬虚、敬虚…」、「八王子キャンパスへは」が「パワンズキャンパスへは」のように聞こえる。CER 0.52（参照 24.6 秒）、0.68（参照 10.3 秒）。
- 切り分けとして、既定の文前処理（`text_frontend=True`）と、全文をかなにした入力も各 15 回試した。前処理ありは CER 0.53（14/15 崩れ、出力は前処理なしと同じ）、かな入力は 0.28（8/15 崩れ）。かなでも実用水準に届かない。
- 本人らしさが 0.785〜0.829 と高く出たが、崩れた音声での値なので判断に使わない（2.4 節）。
- 公式 PyTorch 実装（fp32）で 1 文 3.6 秒。ストリーミングでも、短い文は最初のチャンクが文全体になり、初音は 2.9〜3.5 秒だった。
- 参照音声は 30 秒までなので、33 秒の参照を最後の文の切れ目（24.55 秒）と最初の大きな切れ目（10.29 秒）で切って使った。強化学習版の重み（`llm.rl.pt`）は試していない。

### 2.8 追加検証: Small/40・16・12 を 26 文で比較

5 文だけでは短文が 1 種類しかないため、本番の定型の短文 14 種（「はい。」「いいえ。」「こんにちは。ご用件をどうぞ。」など）と、数字・英単語・長文を含む文 12 種の計 26 文で、seed 1〜5 を作り直した（各構成 130 回、本番と同じ speech.cpp v0.8.7・尺上限あり）。

| 構成 | 生成完了 p50 | 短文の p50 | 長文（74 字）| 本人らしさ（全長参照） | 現行との差 [95%CI] | 言い誤り（CER>0.2） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Small/40（現行） | 0.548 秒 | 0.483 秒 | 0.836 秒 | 0.615 | — | 11/125 |
| Small/16 | 0.264 秒 | 0.218 秒 | 0.437 秒 | 0.618 | +0.003 [−0.026, +0.032] | 9/125 |
| Small/12 | 0.213 秒 | 0.165 秒 | 0.365 秒 | 0.615 | −0.000 [−0.029, +0.029] | 10/125 |

- **3 構成で差はなかった**。言い誤りの大半は「いいえ。」（3 構成とも 5/5）と「ええ。」（同 4/5）で、40 steps でも起きる既知の問題（尺上限が下限の 1.2 秒になる文。`infra/tts/README.md` に記載）。ほかは「はじめまして。香川豊です。」（40 steps の seed 3 だけ）、「八王子駅南口から…」（40・12 steps の seed 1）。
- 「受付は9:30から17:00までです。」は全回で CER 0.28 だが、Whisper が「9時30分」と書く表記の差なので数えていない（5 回 × 3 構成）。
- 本人らしさの絶対値が 2.1 節（0.69 前後）より低いのは、短い音声が多く、埋め込みが不安定になるため。構成間の差で比べる。
- 音声: `/data/ACE/speech-cpp/20261017-validate/`（聞き比べは比較ページの `validate.html`）。

### 2.6 条件と限界

- GPU は RTX 6000 Ada（48 GB）2 枚。Irodori は GPU 0（Vulkan1）、speech.cpp v0.8.7・F16・CPU 8 スレッド・短文の尺上限あり（本番と同じ）。voice ファイルは本番と同じもの（sha256 `4b6c2cd4…`）。
- Qwen3-TTS は GPU 0、CosyVoice3 は GPU 1。Qwen 1.7B（x-vector）と 0.6B（ICL）の計測中は、もう一方の GPU で CosyVoice3 が動いていた（CPU は共有）。Qwen 1.7B（ICL）は単独。PyTorch 版の時間は、どれも最適化前の参考値。
- Irodori の計測開始時の CPU の 1 分平均負荷は 2.3〜3.9（k8s 等の常駐分、Small/10 は未記録）。Large と Drama 3 の計測開始時は 4.4〜7.3（直前の voice 作成などの名残）。
- Small・MF は本番と同じ speech.cpp v0.8.7、Large と Drama 3 は v0.8.8（`/data/ACE/speech-cpp/runtime/v0.8.8`、sha256 一致を確認）。v0.8.8 の Irodori Small/40 は v0.8.7 と同じ seed でバイト単位まで同じ音声だった（3 文で確認）。Large と Drama 3 は汎用の worker 計測スクリプト（`20261017-speechcpp/bench_worker.py`）で測った。短文の尺上限は Irodori だけに渡した（本番と同じ）。
- 本人らしさ・読みは自動指標で、MOS や本人一致の確率ではない。最終判断は試聴で行う。

### 2.7 保存場所・再実行

保存先: `/data/ACE/speech-cpp/20261017-compare/`

- `index.html`: 聞き比べページ（ブラインド比較 12 組＋全構成）
- `summary-tables.md`、`summary.json`、`timing.json`: 集計
- `final/`: 生の音声と各要求の記録（`requests.jsonl`）。`post16k/`: 後処理後の音声
- `speaker-similarity-summary.json`、`asr.jsonl`、`asr-diag.jsonl`（CosyVoice3 の切り分け）
- スクリプト: `run_irodori.sh`（`SPECS="small:16 mf:4"` のように構成を指定）、`run_qwen.sh`、`run_cosyvoice.sh`、`bench_qwen.py`、`bench_cosyvoice.py`、`postprocess.py` → `speaker_similarity.py` → `asr_check.py` → `timing.py` → `summarize.py`
- Large・Drama 3 の生データ: `/data/ACE/speech-cpp/20261017-speechcpp/`（`final/`、`voices/`、`run_large.sh`、`run_drama.sh`、`bench_worker.py`）。`20261017-compare/final/` からシンボリックリンクで参照している
- 環境: Qwen3-TTS は `/data/ACE/qwen3-tts/.venv`、CosyVoice3 は `/data/ACE/cosyvoice3/`（`.venv`、`repo`＝FunAudioLLM/CosyVoice `074ca6d`、`models/Fun-CosyVoice3-0.5B`、`ref/`）

## 3. Small/16 または Small/12 への切り替え方（未実施）

unit（`infra/tts/ace-irodori-tts.service`）に 1 行足して入れ直す（12 にする場合は `12`）。

```ini
Environment=IRODORI_TTS_NUM_STEPS=16
```

```bash
infra/tts/install_irodori_tts_service.sh            # サービス停止中はこれだけ。次の manage_tokkio.sh start で反映される
infra/tts/install_irodori_tts_service.sh --restart  # 稼働中のサービスに反映する場合（話し中の音声は途切れる）
```

- 短文キャッシュはメモリ上だけ（`irodori_server.py:1602`）なので、再起動後は 16 steps の音声になる。
- 固定挨拶の PCM（`IRODORI_TTS_FIXED_OPEN_CAMPUS_GREETING_PCM`）はファイルなので、作り直さない限り 40 steps の音声のまま。
- 短文キャッシュの事前生成（`IRODORI_TTS_SHORT_CACHE_PREWARM_NUM_STEPS`）は空なら同じ steps を使う。

## 4. モデルを変えない工夫（すべて未計測のアイデア）

- **最初の区切りを短くする**: 1 文目だけ最初の読点で切って TTS に送る。短い音声ほど速い（Small/16 で短文 0.172 秒、長文 0.404 秒）。ローカルアバター経路には「最初の読点」で切る処理を移植済み（TASKS.md）。
- **サーバーで worker のチャンクを逐次中継する**: 最大 0.07 秒程度。前後の無音トリムを最初のチャンクで先に行う必要がある。

## 5. 机上調査: その他のローカルモデル

日本語対応・短い参照からのクローン・自前 GPU で動くことを条件に探した。数値は各プロジェクトの公表値で、香川先生の声では試していない。

| モデル | 日本語 | クローン | 速さの公表値 | ライセンス | 判断 |
| --- | --- | --- | --- | --- | --- |
| VoxCPM2 | 30 言語に含む。日本語の評価なし | あり | RTF 約 0.13（RTX 4090、Nano-vLLM） | Apache-2.0 | 未試験。日本語の品質が不明 |
| IndexTTS-2.5 | 公式に ja（かな入力） | あり | RTF 約 0.21（RTX 4090）、ストリーミングの記載なし | bilibili 独自ライセンス | 速さで Irodori に届かない |
| Chatterbox Multilingual | 言語一覧に ja | あり | 公開モデルの数値なし | MIT | 日本語の品質・速さとも不明 |
| GPT-SoVITS v2Pro〜v4 | あり | 5 秒参照、1 分で微調整 | RTF 0.014（RTX 4090） | MIT | 自己回帰なので文長しだい。日本語の微調整は README の Todo で未完 |

除外したもの:

- Fish Audio S2 Pro: 非商用ライセンス（Fish Audio Research License）。約 5B で重い。
- MioTTS-0.1B（Aratako）: 日本語の短文 CER が平均 50.6（最良 13.4）とばらつきが大きい。重みは Falcon-LLM License。
- Irodori v4-Large: 遅い（speech.cpp 文書で RTX 2080 1.25 秒）。Gemma Terms of Use。
- Irodori の v4.1 より新しい公開版: Aratako の Hugging Face ページでは見つからなかった（最新 10 件を確認）。
- Chatterbox-Turbo（英語のみ）。Zonos2、Higgs Audio v2.5 は日本語の速さ・品質を一次資料で確認できなかった。

## 6. 机上調査: 香川先生専用の小さなモデルを学習する

Style-Bert-VITS2（JP-Extra）のような VITS 系は、文全体を一度に作るが非常に軽い。

| エンジン | 学習データ | 速さ | ライセンス | 注意 |
| --- | --- | --- | --- | --- |
| Style-Bert-VITS2 JP-Extra | 最低量の公式記載なし。FAQ では「数分」で動いた報告。論文では 1 話者 10〜15 分 | 公表値なし。RTX 6000 Ada で 1 文 0.06〜0.15 秒と推定（未確認） | コードは AGPL-3.0。事前学習モデルの条件は未確認 | 読みとアクセントは pyopenjtalk。既存の数字正規化はそのまま使える |
| AivisSpeech Engine | SBV2 と同じ | 単一ユーザー向けで、GPU サーバー用途は想定外と README に記載 | LGPL-3.0、モデルごとに別ライセンス | 中身は SBV2 |
| GPT-SoVITS（微調整） | README では 1 分 | RTF 0.014（RTX 4090） | MIT | 日本語の微調整は未完（README の Todo） |

- **手持ちの約 63 秒では足りない**。Irodori で作った音声を学習に使う方法もあるが、Irodori の癖や読み誤りがそのまま学習される。合成音声を大量に足すと話者の近さが下がるという報告もある（ZeSTA、[arXiv 2603.04219](https://arxiv.org/abs/2603.04219)、要旨のみ確認）。
- 現実的には**本人に 10〜15 分の読み上げを録音してもらう**のが前提になる。そのうえで 25〜40 人時の見積もり（調査エージェントの推定、根拠なし）。
- Irodori v4.1-Small のモデルカードは MIT で、出力を学習に使うことは禁じていない。ただし同意のない声のクローンは禁止している。香川先生の書面の同意を取っておく。

## 7. 机上調査: クラウド API

| サービス | 日本語 | 香川先生の声 | 初音の公表値 | 条件 |
| --- | --- | --- | --- | --- |
| **ElevenLabs** eleven_v4_turbo / Flash v2.5 | あり | Instant Voice Clone は Starter（月 $6、30,000 クレジット）から。Professional は Creator（月 $22、121,000 クレジット）から | v4_turbo: 日本から 0.2〜0.4 秒（2026-09-29 実測、既成の声）。Flash v2.5 は推論 約 75 ms（通信を除く） | 中継サービス（port 8022）は作成済み。v4 系は WebSocket 非対応。データ所在地の指定は Enterprise のみ、日本リージョンなし |
| Cartesia Sonic-3.6 | あり（44 言語） | Instant Clone は Pro（月 $5）から、約 10 秒 | 「90 ms 未満」（宣伝ページ、条件不明） | WebSocket あり。pipecat 対応。クローンした日本語の品質は未確認 |
| NVIDIA Magpie TTS Zeroshot（NIM 26.07、自前 GPU） | 26.07 で 12 言語に拡大、ja を含む | 3〜10 秒の参照 | 公表値なし | NVIDIA の利用申請が必要。16 GB 以上の GPU。Tokkio/pipecat に標準の接続部品あり |
| Azure Personal Voice | あり | 5〜90 秒＋本人の同意文の録音 | 公表値なし | 利用申請制。Japan East は Personal Voice 非対応 |
| Google Chirp 3 HD Instant Custom Voice | あり（日本語の同意文あり） | 10 秒以内の参照＋同意文 | 公表値なし | 営業経由の許可リスト制 |

- ElevenLabs v4_turbo の 0.2〜0.4 秒は、Small/16（0.285 秒）とほぼ同じ。速さだけのために外部サービスへ切り替える理由は小さくなった。
- 外部サービスに香川先生の声と会話の文章を送ることになる。大学としての可否は別途判断が必要。
- 30,000 クレジットは日本語で約 3 万字。1 回答 150 字なら約 200 回答で尽きる（1 文字 1 クレジットとして概算、v4 系の消費率は未確認）。
- Hume Octave は 2026-11-13 に API 提供終了予定のため除外。

## 8. 確認範囲と限界

- Web の一次資料は調査用のサブエージェントが取得し、ページの要約として受け取った。ライセンス本文や論文は全文を読んでいない。ElevenLabs の料金と speech.cpp・Qwen3-TTS 微調整の文書は自分でも取得して確認した。
- 実機で試したのは Irodori（speech.cpp）、Drama 3（speech.cpp）、Qwen3-TTS、CosyVoice3 だけ。VoxCPM2、IndexTTS-2.5、Chatterbox、GPT-SoVITS、クラウド API は試していない。
- 未確認: Qwen3-TTS の高速な実行経路（vLLM-Omni、微調整＋speech.cpp）、Drama 3 の Q8_0 以上の重み、CosyVoice3 の強化学習版・vLLM 版、Style-Bert-VITS2 の実測速度、Magpie Zeroshot の速さ、F5-TTS の日本語版・Kotoba-Speech・rinna など未調査の日本語系モデル。
- ace_oyabu は PyTorch 版 Irodori・20 steps（`ace_oyabu/infra/tts/ace-oyabu-irodori-tts.service:20`）で speech.cpp は入っていない。同じ考え方（steps を減らす）は使えるが、声の設定が別なので編集していない。

## 一次資料

- Irodori MF モデルカード: https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small-MF
- Irodori v4.1-Small モデルカード: https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small
- speech.cpp: https://github.com/nyosegawa/speech.cpp 、Qwen3-TTS の文書 https://github.com/nyosegawa/speech.cpp/blob/main/docs/models/qwen3-tts.md
- Qwen3-TTS: https://github.com/QwenLM/Qwen3-TTS 、論文 https://arxiv.org/abs/2601.15621 、微調整 https://github.com/QwenLM/Qwen3-TTS/blob/main/finetuning/README.md
- CosyVoice: https://github.com/FunAudioLLM/CosyVoice 、モデル https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512
- VoxCPM2: https://huggingface.co/openbmb/VoxCPM2
- IndexTTS: https://github.com/index-tts/index-tts
- Chatterbox: https://github.com/resemble-ai/chatterbox
- Fish Audio S2: https://huggingface.co/fishaudio/s2-pro
- MioTTS: https://huggingface.co/Aratako/MioTTS-0.1B
- Style-Bert-VITS2: https://github.com/litagin02/Style-Bert-VITS2 、論文 https://arxiv.org/html/2505.17320v1
- AivisSpeech Engine: https://github.com/Aivis-Project/AivisSpeech-Engine
- GPT-SoVITS: https://github.com/RVC-Boss/GPT-SoVITS
- ZeSTA: https://arxiv.org/abs/2603.04219
- ElevenLabs: https://elevenlabs.io/pricing 、https://elevenlabs.io/docs/overview/models
- Cartesia: https://cartesia.ai/pricing 、https://docs.cartesia.ai/build-with-cartesia/tts-models/latest
- NVIDIA Magpie TTS: https://docs.nvidia.com/nim/speech/26.07.0/reference/support-matrix/tts.html 、https://docs.nvidia.com/nim/speech/26.07.0/tts/voice-cloning.html
- Azure Personal Voice: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/personal-voice-overview
- Google Chirp 3 Instant Custom Voice: https://docs.cloud.google.com/text-to-speech/docs/chirp3-instant-custom-voice
- Hume: https://dev.hume.ai/docs/text-to-speech-tts/overview
