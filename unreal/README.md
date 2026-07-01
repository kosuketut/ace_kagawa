# Unreal Integration

このディレクトリには UE 5.6 C++ プロジェクトへ組み込むための最小雛形を置いています。`ACEAvatarSandbox/` に `.uproject` とゲームモジュールの骨格を追加してあり、`Plugins/ACEConversation` を同梱しています。

## Required Engine Side Setup

1. UE 5.6 の C++ プロジェクトを作成する
2. MetaHuman を追加する
3. NVIDIA ACE Unreal Plugin を導入する
4. `Face_AnimBP` に `Apply ACE Face Animations` ノードを入れる
5. pose asset に `mh_arkit_mapping_pose_A2F` を指定する
6. `Mouth Close` の二重適用を避けるため、既定の口閉じカーブ干渉を外す
7. `RemoteA2F` を有効にし、NVCF 側の Audio2Face-3D へ接続する

既存プロジェクトを使わずこの repo の雛形から始める場合は、`unreal/ACEAvatarSandbox/ACEAvatarSandbox.uproject` をベースにしてください。

## ConversationBridgeComponent

`Plugins/ACEConversation` の `UConversationBridgeComponent` は以下を担うための足場です。

- orchestrator への WebSocket 接続
- `session.start` / `mic.end` / binary `mic.chunk` の送信
- `asr.partial` / `asr.final` / `llm.delta` / `state` / `error` の受信
- `tts.start` / binary `tts.chunk` / `tts.end` の受信

プロジェクト側では以下を接続してください。

- マイク入力: `PushMicChunk()`
- 発話終端: `SendMicEnd()`
- 音声再生: `OnTtsStarted` / `OnTtsAudioChunk` / `OnTtsEnded`
- ACE A2F: `OnTtsAudioChunk` を受けて `FACERuntimeModule::Get().AnimateFromAudioSamples()`

## Audio Playback Component

`UACEAudioPlaybackComponent` を追加してあります。`ConversationBridgeComponent` の delegate を以下のようにつなぐと、Blueprint だけでも受信 PCM をスピーカーへ流せます。

1. `OnTtsStarted` -> `StartPlayback`
2. `OnTtsAudioChunk` -> `PushPcm16`
3. `OnTtsEnded` -> `EndPlayback`

そのうえで同じ `OnTtsAudioChunk` を ACE A2F 呼び出しにも流せば、音声再生と口形生成で同じ PCM を共有できます。

## Project Skeleton

`ACEAvatarSandbox` モジュールには `AACEAvatarCharacter` を追加しています。これは次の最小配線を持ちます。

1. `UConversationBridgeComponent`
2. `UACEAudioPlaybackComponent`
3. `OnTtsStarted/OnTtsAudioChunk/OnTtsEnded` の結線

MetaHuman を配置したら、この Character を親にするか、同等の配線を既存 Character に移してください。

## Audio2Face Hook Points

受信 PCM は次の順で扱うと実運用しやすいです。

1. `tts.start` を受けたら新しい発話ターンを開始する
2. binary `tts.chunk` ごとに
   - `USoundWaveProcedural` に `QueueAudio()`
   - `FACERuntimeModule::Get().AnimateFromAudioSamples()` に同じ PCM を渡す
3. `tts.end` で `EndAudioSamples()`
4. 切断や中断時は `CancelAnimationGeneration()`

## Engine Config

`Config/DefaultEngine.ini.example` を参考に、Derived Data Cache を `/home2/ko66/ace-sandbox/ue-ddc` へ逃がしてください。

## ACE Runtime Hook

NVIDIA 公式ドキュメントでは、ランタイム PCM を A2F に入れる場合は次を使います。

- `ACERuntime` モジュールを追加する
- `#include "ACERuntimeModule.h"` を使う
- `FACERuntimeModule::Get().AnimateFromAudioSamples(...)`
- `FACERuntimeModule::Get().EndAudioSamples(...)`
- 切断時は `FACERuntimeModule::Get().CancelAnimationGeneration(...)`

この repo では NVIDIA ACE plugin 自体は同梱していないため、`AACEAvatarCharacter` 側には呼び出し位置だけコメントで示しています。実際に plugin を導入した段階でその部分を有効化してください。
