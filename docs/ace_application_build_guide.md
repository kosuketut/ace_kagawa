# ACE アプリケーション構築ガイド

作成日: 2026-05-16  
作業ディレクトリ: `/home/kyano/workspace/ACE/ace_kagawa`

## 1. このドキュメントの目的

このドキュメントは、NVIDIA ACE 系の技術を使って、対話型デジタルヒューマン / アバターアプリケーションをゼロから構築するための実践的なガイドです。単なるコマンド一覧ではなく、次の内容を他の人に共有できる粒度で説明します。

- ACE アプリケーションの全体像
- NVIDIA ACE / Tokkio / ACE Agent / Audio2Face-3D / Animation Graph / Unreal Renderer / Riva / Speech NIM / LLM NIM などの関係
- Kubernetes、Docker、NVIDIA GPU Operator、NVIDIA Container Toolkit がどの層を担当するか
- このリポジトリ `/home/kyano/workspace/ACE/ace_kagawa` に既に存在する構成、スクリプト、ドキュメントとの対応
- ローカル検証、Tokkio 5.0 デプロイ、Unreal 直結サンドボックス、Speech NIM compose、TTS CLI の扱い
- コピペ可能な確認・構築・デプロイ・トラブルシュート用コマンド
- スクリーンショットや図を後から追加すべき場所

この文書では、情報の性質を次の 3 種類に分けます。

| 区分 | 意味 |
| --- | --- |
| NVIDIA 公式 docs で確認できた事実 | NVIDIA 公式ドキュメントで確認した仕様、構成、前提条件、URL |
| このリポジトリから読み取れる事実 | `/home/kyano/workspace/ACE/ace_kagawa` に存在する README、docs、scripts、services、unreal、tts から確認した内容 |
| 一般的な構成としての推奨 | 公式情報と既存 repo を踏まえた設計・運用上の推奨。環境に合わせて調整が必要 |

## 2. 想定読者

このドキュメントは次の読者を想定しています。

- NVIDIA ACE を使ったデジタルヒューマンアプリケーションをこれから構築する開発者
- Docker や Kubernetes は聞いたことがあるが、GPU アプリケーションでどう使うかはまだ曖昧な人
- NVIDIA Riva / NIM / Audio2Face / Tokkio の名前は知っているが、どれがどの役割か整理したい人
- `/home/kyano/workspace/ACE/ace_kagawa` の既存 repo を引き継ぎ、Tokkio や Unreal サンドボックスを再構築したい人
- GPU ワークステーション上で、ローカル検証から Kubernetes デプロイまで一通り確認したい人

前提知識として、Linux の基本操作、`bash`、`git`、Python 仮想環境、Docker の基礎があると読みやすいです。Kubernetes については初学者でも追えるように、用語から説明します。

## 3. ACE アプリケーションとは何か

ここでの ACE アプリケーションとは、ユーザーの音声やテキスト入力を受け取り、AI が内容を理解し、回答を生成し、その回答を音声とアバターの表情・身振り・映像として返すアプリケーションを指します。

典型的な流れは次のとおりです。

1. ユーザーがブラウザ、Unreal アプリ、または別のクライアントから音声を入力する
2. 音声認識 ASR が音声をテキストに変換する
3. LLM または RAG がユーザー発話に対する応答テキストを生成する
4. TTS が応答テキストを音声に変換する
5. Audio2Face-3D が音声から顔のアニメーションを生成する
6. Animation Graph が姿勢、ジェスチャ、表情、位置などを統合する
7. Unreal Renderer などのレンダラがアバター映像を生成する
8. WebRTC、WebSocket、gRPC、HTTP などでクライアントに返す

NVIDIA 公式 docs で確認できた事実:

- NVIDIA Tokkio 5.0 は、インタラクティブなアバター体験を作るための NVIDIA AI Blueprint / reference implementation として説明されている。公式 overview は LLM または RAG アプリケーションとデジタルヒューマンインターフェースの統合を示すものと説明している。  
  URL: https://docs.nvidia.com/ace/tokkio/5.0/overview/overview.html
- Tokkio 5.0 の architecture では、Web UI、WebRTC、VST、SDR、ACE Controller、TTS、Audio2Face-3D、Animation Graph、Unreal Renderer などが連携するワークフローが説明されている。  
  URL: https://docs.nvidia.com/ace/tokkio/5.0/overview/architecture.html
- Audio2Face-3D は NVIDIA ACE の中核コンポーネントで、音声と emotion 入力から avatar animation を生成する NIM として説明されている。  
  URL: https://docs.nvidia.com/nim/digital-human/a2f-3d/latest/index.html

このリポジトリから読み取れる事実:

- ルート README は、この repo を `ACE Linux RTX Sandbox` として説明している。
- 現在の主経路は `Tokkio 5.0` を使った `browser-first / Kubernetes / multi-service` 構成である。
- 既存の `Audio2Face + MetaHuman + ASR/LLM/TTS` 直結サンドボックスは Unreal 中心の研究用サブパスとして残されている。
- 大きなキャッシュ、ログ、生成音声、Docker bind mount、UE 生成物は repo 外の `/home2/ko66/ace-sandbox` に置く方針である。Tokkio 作業ディレクトリは `infra/tokkio/workspace` に置くが、Git 管理外にする。

一般的な構成としての推奨:

- 初めて NVIDIA ACE 全体像を動かす場合は、まず Tokkio 5.0 reference workflow を主経路にする。
- 自作プロトコルや Unreal 直結の研究用サンドボックスは、Tokkio の構成を理解した後で、要件に合わせて置き換える。
- 本番に近い構成では Kubernetes を前提にし、単一コンテナや Docker Compose はコンポーネント単体の検証に限定する。

## 4. NVIDIA ACE の全体像

NVIDIA ACE は、デジタルヒューマン、アバター、音声対話、アニメーション、レンダリング、ストリーミングを組み合わせるための NVIDIA の技術群です。単一のライブラリではなく、複数のサービス、NIM、マイクロサービス、reference workflow の集合として理解すると整理しやすいです。

### 4.1 Tokkio

NVIDIA 公式 docs で確認できた事実:

- Tokkio 5.0 はインタラクティブなデジタルヒューマンを構築するための reference workflow / AI Blueprint である。
- Tokkio 5.0 quickstart は bare-metal machine に basic digital human avatar をセットアップし、ブラウザから対話できる状態にする手順を提供する。
- quickstart の前提は `Controller Instance` と `Application Instance` の 2 台構成である。Application Instance は Ubuntu 22.04、少なくとも 2x L4 または 2x A10 GPU、700 GB 以上のストレージが必要とされている。
- Tokkio 5.0 architecture は distributed, event-driven architecture として説明され、NVIDIA SDR が複数 GPU に負荷を分配する。
- Tokkio 5.0 の高レベル workflow では、Tokkio Web UI が coturn 経由で VST と WebRTC 接続し、SDR が ACE Controller / Animation Graph / Unreal Renderer へ stream を割り当てる。

主要 URL:

- https://docs.nvidia.com/ace/tokkio/5.0/overview/overview.html
- https://docs.nvidia.com/ace/tokkio/5.0/overview/architecture.html
- https://docs.nvidia.com/ace/tokkio/5.0/quickstart-guide.html
- https://docs.nvidia.com/ace/tokkio/5.0/reference-workflow/tokkio-ue.html

このリポジトリから読み取れる事実:

- `docs/architecture/tokkio-reference-stack.md` は Tokkio 5.0 を主系にした構成メモである。
- `infra/tokkio` には NVIDIA 公式 `NVIDIA/ACE` repo の one-click deployment を扱いやすくするための補助スクリプトがある。
- `docs/operations/tokkio-rebuild-runbook.md` には、このプロジェクトで検証済みの single-workstation rebuild 手順が記録されている。

### 4.2 ACE Agent

NVIDIA 公式 docs で確認できた事実:

- ACE Agent の latest docs は、Getting Started、Architecture、Deployment、Tutorials、Sample Bots、Configuration Guide、User Guide、API Guide、Best Practices、Reference を持つ。
- ACE Agent docs には Docker Environment、Kubernetes Environment、Python Environment、Sample Clients が含まれる。
- ACE Agent docs には Colang 2.0、LangChain、Low Latency Speech-To-Speech RAG Bot、RAG Bot、LLM Bot などのチュートリアルやサンプルがある。

主要 URL:

- https://docs.nvidia.com/ace/ace-agent/latest/index.html

一般的な構成としての推奨:

- 会話制御や bot 構成を NVIDIA ACE の方式に寄せるなら ACE Agent docs を確認する。
- この repo の `services/orchestrator` は独自 FastAPI オーケストレータなので、ACE Agent と同じものではない。Tokkio / ACE Agent 公式構成と比較するときは、どちらが本線かを明確にする。

### 4.3 Audio2Face-3D

NVIDIA 公式 docs で確認できた事実:

- Audio2Face-3D docs は Getting Started、Architecture、Interacting with Audio2Face-3D、Deployment、Support Matrix、Security、Troubleshooting などを含む。
- Audio2Face-3D Microservice には bidirectional gRPC、health check、configuration fetch、Unreal Engine interaction、Kubernetes deployment、container deployment が用意されている。
- Audio2Face-3D docs の latest は 2026-03-18 時点で更新されている。
- Getting Started では pregenerated TRT engine を使う方式と、GPU 上で TRT engine を生成する方式が説明されている。
- 公式 docs は、A10G、A30、L4、L40S、RTX 4090、RTX 5080、RTX 5090、RTX 6000 Ada、RTX PRO 6000 Blackwell、B200 などの pre-generated engine 対応 GPU を列挙している。対応状況は変わるため、実機では必ず最新 support matrix と `nim_list_model_profiles` 相当の確認を行う。

主要 URL:

- https://docs.nvidia.com/ace/audio2face-3d-microservice/latest/index.html
- https://docs.nvidia.com/ace/audio2face-3d-microservice/latest/text/getting-started/getting-started.html
- https://docs.nvidia.com/nim/digital-human/a2f-3d/latest/index.html

このリポジトリから読み取れる事実:

- Unreal 直結サンドボックスでは `RemoteA2F` を前提にし、ローカル Linux で Audio2Face-3D を直接 Unreal に同梱するのではなく、NVCF またはサービス側 A2F へ接続する想定が書かれている。
- `unreal/README.md` には `FACERuntimeModule::Get().AnimateFromAudioSamples(...)`、`EndAudioSamples(...)`、`CancelAnimationGeneration(...)` などの hook point が記載されている。

### 4.4 Animation Graph

NVIDIA 公式 docs で確認できた事実:

- Animation Graph Microservice docs は、Animation Graph を node-based system として説明し、animation state machines と blend trees の作成を支援するものとしている。
- Animation Graph は Omniverse の runtime framework for skeletal animation blending, playback, and control と説明されている。
- Default Animation Graph は `posture_state`、`gesture_state`、`facial_gesture_state`、`position_state` などの変数を提供する。

主要 URL:

- https://docs.nvidia.com/ace/animation-graph-microservice/1.0/index.html
- https://docs.nvidia.com/ace/animation-graph-microservice/latest/default-animation-graph.html
- https://docs.nvidia.com/ace/animation-graph-microservice/latest/customization.html

### 4.5 Unreal Renderer Microservice

NVIDIA 公式 docs で確認できた事実:

- Unreal Renderer Microservice は Unreal Engine real-time renderer を wrap する microservice である。
- avatar の current pose と audio は gRPC endpoint 経由で提供され、rendered frame と audio は WebRTC で stream される。
- renderer は Animation Graph microservice から animation data を受け取る典型構成になっている。
- MetaHuman Creator による avatar customization や、手動 Unreal project 作成のパスが説明されている。

主要 URL:

- https://docs.nvidia.com/ace/unreal-renderer-microservice/latest/index.html
- https://docs.nvidia.com/ace/tokkio/5.0/reference-workflow/tokkio-ue.html

このリポジトリから読み取れる事実:

- `unreal/ACEAvatarSandbox` は UE 5.6 C++ project skeleton である。
- `unreal/Plugins/ACEConversation` には WebSocket 接続、JSON control frame、binary PCM frame、TTS audio playback の足場がある。
- この repo には NVIDIA ACE Unreal Plugin 自体は同梱されていない。導入後に hook point を有効化する想定である。

### 4.6 Riva / Speech NIM / TTS NIM

NVIDIA 公式 docs で確認できた事実:

- NVIDIA Riva は multimodal conversational systems を構築するための SDK と説明されている。
- Riva は ASR、TTS、NLP などの speech services を提供し、GPU-accelerated inference pipeline として説明されている。
- NVIDIA Speech NIM microservices は GPU-accelerated Docker containers で、speech AI capabilities をアプリケーションの building blocks として提供する。
- Speech NIM は CUDA、TensorRT、Triton を含む NVIDIA inference stack と unified API を single container にパッケージし、gRPC と HTTP interfaces で deploy / scale / interact できる。
- Speech NIM docs は ASR NIM、NMT NIM、TTS NIM、gRPC Protobuf API、Realtime WebSocket API、Performance Benchmarks、Observability を含む。

主要 URL:

- https://docs.nvidia.com/riva/index.html
- https://docs.nvidia.com/nim/speech/latest/index.html

このリポジトリから読み取れる事実:

- `infra/compose/docker-compose.yml` は ASR NIM と TTS NIM を `GPU1` に固定して起動する最小構成である。
- `.env.example` では ASR image として `nvcr.io/nim/nvidia/nemotron-3.5-asr-streaming-0.6b:latest`、TTS image として `nvcr.io/nim/nvidia/magpie-tts-multilingual:latest` が例示されている。
- `services/orchestrator` は `nvidia-riva-client>=2.19.0` を依存に持つ。

### 4.7 NVIDIA NIM / LLM NIM

NVIDIA 公式 docs で確認できた事実:

- NVIDIA NIM microservices は NVIDIA AI Enterprise の一部として、foundation models のデプロイを加速するための microservices と説明されている。
- NIM for Large Language Models は、LLM を NVIDIA inference microservices として運用する production-ready な方法として説明されている。
- NIM LLM latest docs は 2026-05-13 時点で更新されており、NIM Day 0、NIM Turbo、NIM Certified などの offering を説明している。
- NIM LLM 2.0 系では、vLLM を core inference engine とする構成や、Kubernetes Deployment、Helm、NIM Operator Deployment などが docs に含まれる。

主要 URL:

- https://docs.nvidia.com/nim/
- https://docs.nvidia.com/nim/large-language-models/latest/about-nim-llm/overview.html

このリポジトリから読み取れる事実:

- `services/orchestrator` は LLM に NVIDIA NIM API を呼ぶ設計で、`.env` の `ACE_NIM_API_KEY`、`ACE_NIM_LLM_MODEL`、Speech NIM 接続先を調整する想定である。
- `docs/operations/tokkio-rebuild-runbook.md` には、Tokkio 5.0 構成で `ace-controller` が `NvidiaLLMService` を使う一方、TTS は ElevenLabs に依存するケースが記録されている。

## 5. 使用される主要技術の一覧

| 技術 | 役割 | この repo での関係 |
| --- | --- | --- |
| NVIDIA Tokkio 5.0 | browser-first な digital human reference workflow | 主経路。`infra/tokkio` と `docs/architecture/tokkio-reference-stack.md` が補助 |
| ACE Controller | ASR、LLM/RAG、TTS、avatar control、UI WebSocket などの制御 | Tokkio 内の公式構成。repo の `services/orchestrator` は独自代替 |
| ACE Agent | bot / conversational agent 構成、Colang、LangChain、RAG など | 公式 docs を参照する対象。repo に直接同梱されていない |
| Audio2Face-3D | 音声から顔アニメーションを生成 | Tokkio 構成、Unreal 直結サンドボックスの重要要素 |
| Animation Graph | 姿勢・ジェスチャ・表情などの animation data 統合 | Tokkio animation pipeline の構成要素 |
| Unreal Renderer Microservice | Unreal Engine で avatar をレンダリングし WebRTC stream | Tokkio 主経路および `unreal/` の参考対象 |
| Riva | ASR/TTS など speech AI | Tokkio speech path、Speech NIM、`services/orchestrator` 依存 |
| Speech NIM | ASR/TTS/NMT をコンテナ化した speech inference microservices | `infra/compose` に ASR/TTS 起動雛形 |
| LLM NIM | LLM inference microservice | `services/orchestrator` の LLM 接続先候補 |
| Docker | コンテナ実行 | NIM、orchestrator、補助サービスの実行基盤 |
| NVIDIA Container Toolkit | Docker/containerd から GPU を使えるようにする | Docker と Kubernetes の GPU 利用前提 |
| Kubernetes / k8s | 複数 microservice の配置、起動、疎通、スケール、復旧 | Tokkio の主な実行基盤 |
| Helm | Kubernetes アプリケーションの package manager | GPU Operator、NIM、Tokkio chart の文脈で登場 |
| NVIDIA GPU Operator | Kubernetes GPU node に必要な driver/toolkit/device plugin/monitoring を自動管理 | 本番や複数 node で重要 |
| WebRTC | 低遅延音声・映像 streaming | Tokkio UI と VST / Unreal renderer の間 |
| WebSocket | signaling、transcript、状態通知、独自 orchestrator protocol | Tokkio signaling と `services/orchestrator` |
| gRPC | microservice 間 API、Riva、A2F、renderer 連携 | NVIDIA ACE / NIM の多くで利用 |
| FastAPI | 独自 orchestrator の HTTP/WebSocket server | `services/orchestrator` |
| Unreal Engine 5.6 | MetaHuman / avatar rendering | `unreal/ACEAvatarSandbox` |
| Sarashina TTS | ローカル voice cloning / TTS 実験 | `tts/` 配下。NVIDIA ACE 本線とは別系統 |

## 6. アーキテクチャ概要

### 6.1 推奨する全体像

一般的な構成としての推奨:

```text
Browser / Client
  |
  | WebRTC / WebSocket / HTTP
  v
Tokkio UI / Ingress / VST / coturn
  |
  | stream lifecycle event
  v
SDR
  |
  +--> ACE Controller
  |      +--> ASR: Riva / Speech NIM
  |      +--> LLM/RAG: NVIDIA NIM / OpenAI / custom RAG
  |      +--> TTS: Speech NIM / ElevenLabs / other TTS
  |
  +--> Audio2Face-3D
  |
  +--> Animation Graph
  |
  +--> Unreal Renderer Microservice
          |
          | WebRTC video/audio stream
          v
      Browser / Client
```

NVIDIA 公式 docs で確認できた事実:

- Tokkio 5.0 architecture は、Web UI が coturn 経由で VST と WebRTC 接続し、VST が Redis message を出し、SDR が stream を ACE Controller / Animation Graph / Unreal Renderer に割り当てる流れを説明している。
- ACE Controller は live audio を処理し、knowledge base に接続し、TTS service を呼び、streaming audio response を生成する。
- streaming audio response は Audio2Face-3D に送られ、Audio2Face-3D が facial animation を生成する。
- ACE Controller は animation data と audio response を Animation Graph microservice に送り、Animation Graph が final animation を生成して renderer に送る。
- renderer は 3D avatar に animation を適用し、映像を Web UI に stream する。

### 6.2 この repo の実装対応

このリポジトリから読み取れる事実:

| パス | 役割 |
| --- | --- |
| `README.md` | repo 全体の入口。Tokkio 主経路と Unreal サブパスを説明 |
| `docs/architecture/tokkio-reference-stack.md` | Tokkio 5.0 主経路の構成メモ |
| `docs/architecture/ace-sandbox.md` | Unreal 直結研究サンドボックスの architecture |
| `docs/operations/tokkio-rebuild-runbook.md` | single-workstation Tokkio rebuild runbook |
| `docs/operations/tokkio-webui-startup-runbook.md` | Tokkio Web UI 起動確認系の runbook |
| `docs/operations/tokkio-japanese-customization.md` | Tokkio 日本語向けカスタマイズ手順 |
| `infra/tokkio` | Tokkio one-click deployment の補助 scripts |
| `infra/compose` | ASR/TTS Speech NIM の Docker Compose 雛形 |
| `services/orchestrator` | FastAPI + WebSocket の独自 half-duplex orchestrator |
| `unreal` | UE 5.6 / MetaHuman / ACEConversation plugin の skeleton |
| `tts` | Sarashina TTS を使うローカル voice cloning CLI |

一般的な構成としての推奨:

- `Tokkio` を本線にする場合、`services/orchestrator` と `unreal/Plugins/ACEConversation` は直接使わず、理解・比較・研究用として扱う。
- Tokkio を使わず独自 Unreal アプリを作る場合、`infra/compose` で ASR/TTS NIM を立て、`services/orchestrator` と Unreal plugin skeleton をつなぐ。
- `tts/` の Sarashina CLI は NVIDIA ACE 公式構成ではないため、本番 ACE アプリ構築ガイドでは「別系統の音声生成実験」として扱う。

### 6.3 スクリーンショット: 全体アーキテクチャ図

![ACE アプリケーション全体アーキテクチャ図](images/ace_application_architecture_overview.png)

画像メモ:

- 内容: Browser、Tokkio UI、VST、SDR、ACE Controller、ASR、LLM/RAG、TTS、Audio2Face-3D、Animation Graph、Unreal Renderer、Kubernetes、GPU node の関係図
- 到達方法: このドキュメントの `6.1 推奨する全体像` をもとに diagrams.net、Figma、PowerPoint、Mermaid などで図を作成する
- 必要なコマンド: なし。Mermaid で作る場合は Markdown viewer または mermaid-cli を使う
- Web サイトの場合の URL: NVIDIA Tokkio architecture docs https://docs.nvidia.com/ace/tokkio/5.0/overview/architecture.html を参考にする
- 撮影する場面: 構成要素とデータフローが 1 枚で見える状態
- Markdown プレースホルダ: `![ACE アプリケーション全体アーキテクチャ図](images/ace_application_architecture_overview.png)`

## 7. 最小構成と本番構成の違い

| 観点 | 最小構成 | 本番構成 |
| --- | --- | --- |
| 目的 | 機能確認、理解、単体検証 | 安定運用、スケール、監視、セキュリティ |
| 実行基盤 | 単一 GPU workstation、Docker Compose、単一 node Kubernetes | 複数 GPU node、Kubernetes、Helm、GPU Operator |
| ASR/TTS | Speech NIM compose または外部 API | Riva / Speech NIM / cloud API を SLA と監視付きで運用 |
| LLM | NVIDIA hosted API、単一 LLM NIM、OpenAI API | LLM NIM、RAG、fallback、rate limit、認証管理 |
| Avatar | Tokkio default avatar、MetaHuman skeleton | カスタム MetaHuman、renderer microservice、asset pipeline |
| ストリーミング | localhost または単一 host の WebRTC | coturn、TLS、Ingress、証明書、ネットワーク設計 |
| Secret | `.env` に手動設定 | Kubernetes Secret、外部 secret manager、RBAC、暗号化 |
| 監視 | `kubectl get pods`、logs、curl | Prometheus、Grafana、DCGM、alert、tracing |
| データ保持 | ローカル filesystem | PersistentVolume、object storage、backup |
| セキュリティ | 開発者ローカル前提 | TLS、network policy、RBAC、image scanning、audit |

NVIDIA 公式 docs で確認できた事実:

- Tokkio quickstart は基本的な avatar を bare-metal machine にセットアップするための導入手順であり、全 customization を詳述するものではないと明記している。
- NVIDIA GPU Operator は Kubernetes 上で GPU を使うために必要な driver、device plugin、container toolkit、GFD、DCGM monitoring などを自動管理する。

一般的な構成としての推奨:

- 最小構成ではまず `docker run --gpus all`、Speech NIM compose、Tokkio one-click deployment を順に確認する。
- 本番構成では GPU Operator、Kubernetes Secret、Ingress TLS、監視、ログ保全、GPU resource request、node selector / affinity を設計する。

## 8. ローカル開発環境の準備

### 8.1 作業ディレクトリと repo 確認

実行ディレクトリ: `/home/kyano/workspace/ACE/ace_kagawa`

```bash
pwd
find . -maxdepth 3 -type f \
  \( -name 'README.md' -o -name 'pyproject.toml' -o -name '*.md' -o -name '.env.example' \) \
  | sort
```

期待する主なファイル:

```text
README.md
docs/architecture/ace-sandbox.md
docs/architecture/tokkio-reference-stack.md
docs/operations/tokkio-rebuild-runbook.md
infra/tokkio/README.md
infra/tokkio/.env.example
infra/compose/README.md
infra/compose/.env.example
services/orchestrator/README.md
services/orchestrator/pyproject.toml
tts/README.md
tts/pyproject.toml
unreal/README.md
```

### 8.2 基本ツール確認

実行ディレクトリ: `/home/kyano/workspace/ACE/ace_kagawa`

```bash
uname -a
lsb_release -a || cat /etc/os-release
git --version
python3 --version
docker --version
kubectl version --client
helm version
```

注意:

- `kubectl version --client` は client だけの確認です。cluster への接続確認は後述の `kubectl cluster-info` や `kubectl get nodes` を使います。
- `helm version` が失敗する場合、Tokkio や GPU Operator の Helm chart 操作前に Helm を導入してください。

### 8.3 Python / orchestrator 開発環境

このリポジトリから読み取れる事実:

- `services/orchestrator` は FastAPI + asyncio ベースの half-duplex 会話オーケストレータである。
- 依存には `fastapi`、`uvicorn[standard]`、`httpx`、`pydantic`、`webrtcvad-wheels`、`nvidia-riva-client`、`websockets` が含まれる。
- mock mode として `ACE_MOCK_ASR=true`、`ACE_MOCK_LLM=true`、`ACE_MOCK_TTS=true` が用意されている。

実行ディレクトリ: `/home/kyano/workspace/ACE/ace_kagawa/services/orchestrator`

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
python3 tools/init_storage.py
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

別 terminal で確認:

```bash
curl -s http://127.0.0.1:8080/healthz
curl -s http://127.0.0.1:8080/status
python3 tools/demo_client.py --url ws://127.0.0.1:8080/ws/session --mock-audio
```

注意:

- この orchestrator は Tokkio 公式の ACE Controller ではありません。
- Tokkio 主経路を使う場合は、まず `infra/tokkio` と公式 `NVIDIA/ACE` repo 側の workflow を優先してください。

### 8.4 Sarashina TTS CLI 開発環境

このリポジトリから読み取れる事実:

- `tts/` は `sbintuitions/sarashina2.2-tts` を使ったローカル voice cloning workflow である。
- `uv run ace-tts download-reference`、`synthesize`、`clone-from-youtube` の CLI がある。
- 参照音声には話者・権利者の許諾が必要である。

実行ディレクトリ: `/home/kyano/workspace/ACE/ace_kagawa/tts`

```bash
uv sync
uv run ace-tts --help
```

参照音声から音声生成する例:

```bash
uv run ace-tts synthesize \
  --reference-audio data/reference_clip.wav \
  --reference-text 'ここに参照音声の文字起こしを入れてください。' \
  --text '生成したい文章をここに入れてください。' \
  --output output/cloned.wav
```

注意:

- Sarashina TTS は NVIDIA ACE 公式構成ではありません。
- ACE アプリケーションの本線に組み込む場合は、ライセンス、商用利用可否、watermark、品質、レイテンシ、TTS API 互換性を別途確認してください。

## 9. GPU / NVIDIA Driver / CUDA / Container Toolkit の確認

### 9.1 GPU と driver の確認

実行ディレクトリ: 任意

```bash
nvidia-smi
nvidia-smi --query-gpu=index,name,driver_version,cuda_version,memory.total --format=csv
```

確認ポイント:

- GPU が認識されているか
- driver version が表示されるか
- CUDA version が表示されるか
- Tokkio / Audio2Face-3D / Speech NIM / LLM NIM の必要 GPU メモリを満たすか

注意:

- `nvidia-smi` が host で成功しても、Docker や Kubernetes の Pod から GPU が使えるとは限りません。
- Docker / containerd / Kubernetes から GPU を使うには NVIDIA Container Toolkit、device plugin、GPU Operator などの設定が必要です。

### 9.2 Docker から GPU が見えるか確認

NVIDIA 公式 docs で確認できた事実:

- NVIDIA Container Toolkit は GPU-accelerated containers を build/run するための libraries と utilities の集合である。
- Toolkit には `nvidia-container-runtime`、`nvidia-ctk`、`nvidia-cdi-hook`、`nvidia-container-runtime-hook`、`nvidia-container-cli`、`libnvidia-container1` などが含まれる。

主要 URL:

- https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html

実行ディレクトリ: 任意

```bash
docker info | sed -n '/Runtimes/,+5p'
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

注意:

- `docker run --rm --gpus all ...` は Docker Hub / NVIDIA container image の pull が発生する場合があります。
- 社内 proxy、air-gapped environment、NGC login、image mirror の制約がある場合は先に確認してください。
- `could not select device driver "" with capabilities: [[gpu]]` が出る場合、Docker runtime から NVIDIA GPU が使える状態ではありません。

### 9.3 NVIDIA Container Toolkit 設定確認

実行ディレクトリ: 任意

```bash
which nvidia-ctk
nvidia-ctk --version
nvidia-container-runtime --version || true
nvidia-container-cli --version || true
```

Docker runtime 設定確認:

```bash
docker info | grep -i -E 'runtimes|nvidia|default runtime' -A3
```

危険度のある操作:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

注意:

- 上の `sudo nvidia-ctk runtime configure` と `systemctl restart docker` は Docker daemon 設定を変更し、既存コンテナに影響します。
- 共有サーバーでは必ず管理者と合意してから実行してください。
- Tokkio や NIM の稼働中に Docker を restart するとサービス停止につながる可能性があります。

### 9.4 CUDA toolkit と runtime の違い

一般的な構成としての推奨:

- host に CUDA toolkit が入っていなくても、NIM や CUDA container には必要 runtime が含まれる場合がある。
- 重要なのは host の NVIDIA driver が container 内の CUDA runtime と互換であり、container runtime が GPU device を渡せること。
- アプリケーションが custom CUDA build を必要とする場合のみ、host 側 CUDA toolkit の導入や version pinning を検討する。

確認コマンド:

```bash
nvcc --version || true
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 bash -lc 'nvidia-smi && ldconfig -p | grep -E "libcuda|libnvidia-ml" | head'
```

## 10. Docker の役割

Docker は、NIM、ASR/TTS、LLM、Audio2Face-3D、補助 API などを、依存関係ごと container image として実行するための基盤です。

NVIDIA 公式 docs で確認できた事実:

- NVIDIA Speech NIM microservices は GPU-accelerated Docker containers として提供され、gRPC / HTTP interface で利用する。
- NVIDIA Container Toolkit は GPU-accelerated containers を実行するために必要な runtime/tool 群を提供する。

このリポジトリから読み取れる事実:

- `infra/compose/docker-compose.yml` は ASR NIM と TTS NIM を起動する。
- compose では `runtime: nvidia`、`NVIDIA_VISIBLE_DEVICES: "1"`、`NIM_CACHE_PATH`、`NGC_API_KEY`、port mapping、healthcheck が使われている。
- ASR は host の `50051/9000`、TTS は `50052/9001` を使う前提である。

実行ディレクトリ: `/home/kyano/workspace/ACE/ace_kagawa/infra/compose`

```bash
cp .env.example .env
$EDITOR .env
docker compose --env-file .env -f docker-compose.yml up -d
docker compose --env-file .env -f docker-compose.yml ps
```

Speech NIM stack の確認:

```bash
python3 check_nim_stack.py --asr-http-url http://127.0.0.1:9000 --tts-http-url http://127.0.0.1:9001
python3 check_nim_stack.py --tts-grpc 127.0.0.1:50052 --tts-text "こんにちは、音声合成の確認です。"
```

ASR 実測:

```bash
python3 check_nim_stack.py --asr-grpc 127.0.0.1:50051 --asr-wav /path/to/input-16khz-mono-pcm16.wav
```

注意:

- `.env` の `NGC_API_KEY` は secret です。git commit しないでください。
- NIM image の pull には NGC 権限や license acceptance が必要な場合があります。
- `NVIDIA_VISIBLE_DEVICES: "1"` は GPU index 1 を使う設定です。GPU が 1 枚しかない環境では動きません。環境に合わせて変更してください。

## 11. Kubernetes の役割

### 11.1 「クバネティス？」とは何か

「クバネティス？」と呼ばれているものは Kubernetes のことです。略称として `k8s` と書かれることもあります。`Kubernetes` の `K` と `s` の間に 8 文字あるため `k8s` です。

Kubernetes は、複数の containerized application をまとめて配置、起動、停止、再起動、更新、スケール、ネットワーク公開、設定注入、secret 管理するための orchestration platform です。ACE アプリケーションでは、ASR、LLM、TTS、Audio2Face-3D、Animation Graph、renderer、UI、ingress、monitoring など多くの service が同時に動くため、Kubernetes が重要になります。

### 11.2 ACE アプリケーションで Kubernetes が持つ役割

一般的な構成としての推奨:

- 複数の ACE microservice を Pod として起動する
- 各 Pod に必要な GPU、CPU、memory を割り当てる
- ASR/TTS/A2F/renderer などを Service 名で相互接続する
- Ingress や NodePort / LoadBalancer で外部から UI/API にアクセスする
- ConfigMap で非秘密設定を配布する
- Secret で API key や NGC credential を管理する
- Helm chart で複雑な YAML 群を package 化して install / upgrade / rollback する
- NVIDIA GPU Operator や device plugin により `nvidia.com/gpu` resource を Pod に割り当てる
- `kubectl logs`、`kubectl describe`、`kubectl get events` で障害調査する

### 11.3 主要 Kubernetes リソース

NVIDIA 公式 docs ではなく Kubernetes 公式 docs で確認した事実:

- Deployment は、通常 state を持たない application workload を実行する Pod 群を管理し、desired state に向けて Pod / ReplicaSet を declarative に更新する。  
  URL: https://kubernetes.io/docs/concepts/workloads/controllers/deployment/
- Service は、cluster 内の 1 つ以上の Pod で実行される network application を単一 endpoint として公開する。  
  URL: https://kubernetes.io/docs/concepts/services-networking/service/
- Ingress は、cluster 外から Service への HTTP / HTTPS route を定義する API object である。Ingress を機能させるには Ingress controller が必要である。  
  URL: https://kubernetes.io/docs/concepts/services-networking/ingress/
- Namespace は、単一 cluster 内で resource group を分離する mechanism であり、production cluster では `default` namespace を避けることが推奨されている。  
  URL: https://kubernetes.io/docs/concepts/overview/working-with-objects/namespaces/
- ConfigMap は non-confidential data を key-value pair として保存する API object であり、Pod は環境変数、command-line argument、設定ファイル volume として利用できる。機密情報には Secret を使うべきである。  
  URL: https://kubernetes.io/docs/concepts/configuration/configmap/
- Secret は password、token、key などの sensitive data を保存する object である。ただし Kubernetes Secrets は default では API server の backing store である etcd に暗号化されず保存されるため、RBAC や encryption at rest の検討が必要である。  
  URL: https://kubernetes.io/docs/concepts/configuration/secret/
- Kubernetes は GPU を device plugin framework により扱い、plugin 導入後に `nvidia.com/gpu` のような schedulable resource として公開できる。GPU は `limits` section に指定する。  
  URL: https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/

### 11.4 ACE と Kubernetes の関係を初学者向けに説明

ACE アプリケーションは、1 つの巨大なプログラムではなく、複数の専門サービスが協調する構成です。

例えば、会話アバターを動かすには次が必要です。

- ブラウザ UI
- 音声・映像 stream の入口
- 音声認識
- LLM / RAG
- 音声合成
- 顔アニメーション生成
- 全身アニメーション制御
- Unreal renderer
- ログ、監視、設定、secret、GPU 管理

これらを手作業で 1 つずつ `docker run` することもできますが、サービス数が増えるほど、起動順、ネットワーク、再起動、GPU 割り当て、設定変更、バージョン更新が難しくなります。Kubernetes は、この複数サービス構成を「宣言した状態に近づけ続ける」ための基盤です。

ACE / Tokkio では、Kubernetes は次のような見え方になります。

```text
Kubernetes cluster
  Namespace: app
    Deployment / StatefulSet:
      - ace-controller
      - a2f
      - animation-graph
      - unreal-renderer
      - riva-speech
      - tokkio-ui
      - tokkio-ingress
      - VST
      - Redis / Triton / SDR sidecars
    Service:
      - internal service discovery
      - UI/API/renderer endpoint
    Secret:
      - NGC key
      - NVIDIA API key
      - TTS provider key
    ConfigMap:
      - service config
      - workflow config
```

このリポジトリから読み取れる事実:

- Tokkio 検証では `app` namespace の Pod 状態を `kubectl get pods -n app` で確認する。
- `docs/operations/tokkio-rebuild-runbook.md` では、`a2f-a2f-deployment`、`ace-controller`、`ia-animation-graph-microservice`、`ia-unreal-renderer-microservice`、`riva-speech`、`tokkio-ingress`、`tokkio-ui` などの Pod が確認対象になっている。

## 12. NVIDIA GPU Operator の役割

NVIDIA 公式 docs で確認できた事実:

- NVIDIA GPU Operator は Kubernetes operator framework を使い、GPU を Kubernetes で利用するために必要な NVIDIA software components の管理を自動化する。
- GPU Operator が扱う component には、NVIDIA driver、Kubernetes device plugin for GPUs、NVIDIA Container Toolkit、GPU Feature Discovery、DCGM based monitoring などが含まれる。
- Kubernetes は device plugin framework で GPU などの特殊 hardware resource にアクセスするが、driver、container runtime、library などの構成は複雑であるため、GPU Operator がそれらを自動化する。

主要 URL:

- https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/index.html
- https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/getting-started.html
- https://docs.nvidia.com/datacenter/cloud-native/index.html

GPU Operator の位置づけ:

```text
Kubernetes
  |
  +-- NVIDIA GPU Operator
        |
        +-- NVIDIA Driver
        +-- NVIDIA Container Toolkit
        +-- NVIDIA Kubernetes Device Plugin
        +-- GPU Feature Discovery
        +-- DCGM / DCGM Exporter
        +-- MIG Manager など
```

導入例:

実行ディレクトリ: 任意

```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update

helm install --wait gpu-operator \
  -n gpu-operator --create-namespace \
  nvidia/gpu-operator
```

host に NVIDIA driver を事前導入済みの場合の例:

```bash
helm install --wait gpu-operator \
  -n gpu-operator --create-namespace \
  nvidia/gpu-operator \
  --set driver.enabled=false
```

host に NVIDIA driver と NVIDIA Container Toolkit を事前導入済みの場合の例:

```bash
helm install --wait gpu-operator \
  -n gpu-operator --create-namespace \
  nvidia/gpu-operator \
  --set driver.enabled=false \
  --set toolkit.enabled=false
```

注意:

- GPU Operator の導入は cluster に大きな変更を加えます。
- 既に driver / toolkit / device plugin が手動導入されている cluster では、二重管理を避けるため chart options を慎重に設定してください。
- 本番 cluster では platform support、Kubernetes version、container runtime、OS version、security policy、privileged namespace の扱いを事前に確認してください。

確認コマンド:

```bash
kubectl get pods -n gpu-operator
kubectl get nodes -o wide
kubectl describe node <gpu-node-name> | grep -A8 -i "nvidia.com/gpu"
kubectl get node <gpu-node-name> -o jsonpath='{.status.allocatable.nvidia\.com/gpu}{"\n"}'
```

GPU を要求する Pod の最小例:

```bash
cat <<'YAML' > /tmp/gpu-test-pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-test
spec:
  restartPolicy: Never
  containers:
    - name: cuda
      image: nvidia/cuda:12.4.1-base-ubuntu22.04
      command: ["nvidia-smi"]
      resources:
        limits:
          nvidia.com/gpu: 1
YAML

kubectl apply -f /tmp/gpu-test-pod.yaml
kubectl logs pod/gpu-test
kubectl delete -f /tmp/gpu-test-pod.yaml
```

注意:

- 上記は `/tmp/gpu-test-pod.yaml` に一時ファイルを作成します。
- image pull が必要です。
- GPU が 1 枚しかない場合、既存 workload が GPU を使っていると Pending になる可能性があります。

## 13. ACE 関連サービスの構成

### 13.1 Tokkio 主経路

このリポジトリから読み取れる事実:

- `infra/tokkio` は Tokkio 5.0 を controller 側から起動する補助物である。
- Tokkio 本体はこの repo には含めず、NVIDIA 公式 `NVIDIA/ACE` repo を利用する。
- `prepare_tokkio_workspace.py` は `.env` を読んで `/home/kyano/workspace/ACE/ace_kagawa/infra/tokkio/workspace` 配下を初期化し、one-click script 用 `my-config.env` を生成する。
- `deploy_tokkio.sh` は `envbuild.sh` の `init-config`、`install`、`info`、`uninstall` を叩くラッパである。
- `manage_tokkio.sh` は install 済み Tokkio 環境の `start`、`stop`、`status`、`restart`、`reapply`、`restart-controller`、`logs` をまとめる運用ラッパである。
- `check_tokkio_endpoints.py` は UI/API/Grafana と `kubectl get pods -n app` の簡易確認を行う。
- `check_tokkio_ngc_access.py` は Tokkio chart repo と `Audio2Face-3D` image pull 権限を事前確認する。

主な directory:

```text
/home/kyano/workspace/ACE/ace_kagawa/infra/tokkio
/home/kyano/workspace/ACE/ace_kagawa/infra/tokkio/workspace
/home/kyano/workspace/ACE/ace_kagawa/infra/tokkio/workspace/controller
/home/kyano/workspace/ACE/ace_kagawa/infra/tokkio/workspace/NVIDIA-ACE
```

### 13.2 Unreal 直結サンドボックス

このリポジトリから読み取れる事実:

- `docs/architecture/ace-sandbox.md` は Tokkio ではなく Unreal 直結の研究サンドボックス用である。
- 対象は Linux RTX 単一 workstation 上の real-time half-duplex 会話アバターである。
- `GPU0` を Unreal / MetaHuman 表示用、`GPU1` を Speech NIM 用に固定する構想が書かれている。
- 会話状態は `LISTENING -> THINKING -> SPEAKING -> LISTENING` の 4 状態に限定している。
- `services/orchestrator` は WebSocket session、WebRTC VAD、ASR、NVIDIA NIM API への LLM streaming、sentence chunking 後の TTS streaming、JSONL logging を担う。
- `infra/compose` は ASR NIM / TTS NIM を `GPU1` に固定する。
- `unreal/Plugins/ACEConversation` は Unreal 側 WebSocket / PCM 受け渡しの skeleton である。

### 13.3 TTS / voice cloning CLI

このリポジトリから読み取れる事実:

- `tts` は `ace-tts` という `uv` project である。
- `pyproject.toml` は `sarashina2-2-tts` を GitHub source として指定している。
- `yt-dlp` と `ffmpeg` を使って許諾済み YouTube audio を取得し、参照音声から音声合成する workflow である。

一般的な構成としての推奨:

- NVIDIA ACE 本線の TTS としてはまず Riva / Speech NIM / Tokkio 既定 TTS を検討する。
- Sarashina TTS は日本語品質や voice cloning 実験の補助として扱い、商用・本番利用ではライセンス確認を必須にする。

## 14. 音声認識、音声合成、LLM、アバター描画、アニメーション、ストリーミングの流れ

### 14.1 Tokkio 5.0 の流れ

NVIDIA 公式 docs で確認できた事実に基づく流れ:

1. ユーザーが Tokkio Web UI を開く
2. Web UI が coturn server 経由で VST と WebRTC 接続する
3. WebRTC signaling には WebSocket connection を使う
4. VST が新しい client connection と streamID を Redis message として通知する
5. SDR が stream を利用可能な ACE Controller、Animation Graph、Unreal Renderer の Pod に route する
6. renderer が avatar video を Web UI に stream し始める
7. ACE Controller が VST から live audio を受け取る
8. ACE Controller が ASR で音声を text にする
9. ACE Controller が knowledge base / LLM / RAG に接続して response を生成する
10. ACE Controller が TTS service を呼び、streaming audio response を生成する
11. audio response が Audio2Face-3D に送られ、facial animation が生成される
12. ACE Controller が animation data、audio response、gesture trigger を Animation Graph に送る
13. Animation Graph が final animation を生成し renderer に送る
14. renderer が 3D avatar に animation を適用し、映像を Web UI に stream する
15. Web UI は ACE Controller との WebSocket で transcript、tables、images などの追加情報も受け取る

### 14.2 独自 Unreal 直結サンドボックスの流れ

このリポジトリから読み取れる事実に基づく流れ:

1. Unreal client が `services/orchestrator` に WebSocket 接続する
2. client は `session.start` JSON frame を送る
3. client は `16kHz mono PCM16` の mic binary frame を 20ms ごとに送る
4. VAD または明示 `mic.end` で発話終端を検出する
5. orchestrator が ASR partial / final transcript を返す
6. orchestrator が NVIDIA NIM API へ LLM prompt を送り streaming delta を返す
7. sentence chunking 後に TTS streaming を行う
8. TTS PCM を Unreal へ binary frame として返す
9. Unreal 側 `ACEAudioPlaybackComponent` が受信 PCM を再生する
10. 同じ PCM を ACE A2F runtime hook に渡して口形・顔アニメーションを生成する

## 15. ゼロから構築する手順

この章は、Tokkio 5.0 を主経路とした ACE アプリケーション構築手順です。既存 repo の補助スクリプトを活用しつつ、公式 `NVIDIA/ACE` repo を使います。

### 15.1 事前に必要なもの

NVIDIA 公式 docs で確認できた事実:

- Tokkio 5.0 quickstart は Controller Instance と Application Instance の 2 台構成を前提にしている。
- Controller Instance は Ubuntu 22.04、SSH key pair、passwordless sudo が必要である。
- Application Instance は Ubuntu 22.04、2x L4 または 2x A10 以上、700 GB 以上の storage、初回実行前に既存 Kubernetes が動いていないこと、passwordless sudo が必要である。

このリポジトリから読み取れる事実:

- この repo の runbook では single workstation 構成として controller と app を同一 machine で扱った記録がある。
- 検証済み workspace root は `/home/kyano/workspace/ACE/ace_kagawa/infra/tokkio/workspace` である。
- Tokkio branch/profile は `5.0.0-ga` / `tokkio-1stream` が使われている。

必要な account / secret:

- NVIDIA API key
- NGC CLI API key
- Audio2Face-3D NIM image / chart へのアクセス権限
- Tokkio chart / image へのアクセス権限
- TTS provider key。Tokkio 5.0 quickstart default では ElevenLabs を使うケースがある
- OpenAI key は `OpenAILLMService` を使う場合のみ必要。NVIDIA LLM 経路なら placeholder で validation を通す repo helper がある

### 15.2 repository-side config を作る

実行ディレクトリ: `/home/kyano/workspace/ACE/ace_kagawa`

```bash
cp infra/tokkio/.env.example infra/tokkio/.env
$EDITOR infra/tokkio/.env
```

最低限確認する項目:

```dotenv
TOKKIO_ACE_BRANCH=5.0.0-ga
TOKKIO_PROFILE=tokkio-1stream

TOKKIO_WORKSPACE_DIR=/home/kyano/workspace/ACE/ace_kagawa/infra/tokkio/workspace
TOKKIO_ACE_REPO_DIR=/home/kyano/workspace/ACE/ace_kagawa/infra/tokkio/workspace/NVIDIA-ACE
TOKKIO_CONTROLLER_DIR=/home/kyano/workspace/ACE/ace_kagawa/infra/tokkio/workspace/controller

TOKKIO_APP_HOST_IPV4_ADDR=<application-host-ip>
TOKKIO_APP_HOST_SSH_USER=<ssh-user>
TOKKIO_COTURN_HOST_IPV4_ADDR=<coturn-host-ip>
TOKKIO_COTURN_HOST_SSH_USER=<ssh-user>

TOKKIO_NVIDIA_API_KEY=<your-nvidia-api-key>
TOKKIO_NGC_CLI_API_KEY=<your-ngc-api-key>
TOKKIO_OPENAI_API_KEY=<optional-or-empty>
TOKKIO_ELEVENLABS_API_KEY=<your-elevenlabs-api-key-if-used>
```

注意:

- `.env` は secret を含むため commit しないでください。
- `TOKKIO_APP_HOST_IPV4_ADDR` はブラウザや controller から到達可能な IP にします。
- single workstation 構成では app/coturn/controller が同一 host になる場合がありますが、公式 quickstart は 2 台構成を前提にしています。

### 15.3 workspace artifacts を生成する

実行ディレクトリ: `/home/kyano/workspace/ACE/ace_kagawa`

```bash
python3 infra/tokkio/prepare_tokkio_workspace.py --env-file infra/tokkio/.env
```

期待される生成物:

```text
/home/kyano/workspace/ACE/ace_kagawa/infra/tokkio/workspace/controller/generated/my-config.env
/home/kyano/workspace/ACE/ace_kagawa/infra/tokkio/workspace/controller/ace-app-config.yml
```

確認:

```bash
ls -lah /home/kyano/workspace/ACE/ace_kagawa/infra/tokkio/workspace/controller
ls -lah /home/kyano/workspace/ACE/ace_kagawa/infra/tokkio/workspace/controller/generated
```

### 15.4 NVIDIA 公式 ACE repo を clone する

実行ディレクトリ: `/home/kyano/workspace/ACE/ace_kagawa`

```bash
git clone https://github.com/NVIDIA/ACE.git /home/kyano/workspace/ACE/ace_kagawa/infra/tokkio/workspace/NVIDIA-ACE
```

既に存在する場合:

```bash
git -C /home/kyano/workspace/ACE/ace_kagawa/infra/tokkio/workspace/NVIDIA-ACE status
git -C /home/kyano/workspace/ACE/ace_kagawa/infra/tokkio/workspace/NVIDIA-ACE branch --show-current
git -C /home/kyano/workspace/ACE/ace_kagawa/infra/tokkio/workspace/NVIDIA-ACE log -1 --oneline
```

注意:

- 公式 repo 側を直接編集する場合は、どの変更が local customization か分かるように patch / branch / diff を残してください。
- この repo の `prepare_tokkio_workspace.py` は `NVIDIA-ACE` clone が存在すれば日本語向け `llm-rag` patch を自動適用する仕組みを持っています。

### 15.5 controller config を初期化する

実行ディレクトリ: `/home/kyano/workspace/ACE/ace_kagawa`

```bash
./infra/tokkio/deploy_tokkio.sh init-config --env-file infra/tokkio/.env
```

生成・確認対象:

```text
/home/kyano/workspace/ACE/ace_kagawa/infra/tokkio/workspace/controller/ace-app-config.yml
```

編集:

```bash
$EDITOR /home/kyano/workspace/ACE/ace_kagawa/infra/tokkio/workspace/controller/ace-app-config.yml
```

確認する項目:

- host address
- SSH user
- SSH key path
- selected Tokkio profile
- avatar / renderer / workflow settings
- LLM / TTS / API key の参照
- single stream か multi stream か

### 15.6 NGC と Audio2Face-3D access を事前確認する

実行ディレクトリ: `/home/kyano/workspace/ACE/ace_kagawa`

```bash
python3 infra/tokkio/check_tokkio_ngc_access.py \
  --env-file /home/kyano/workspace/ACE/ace_kagawa/infra/tokkio/workspace/controller/generated/my-config.env
```

期待する結果:

- Tokkio chart repository にアクセスできる
- `nvcr.io/nim/nvidia/audio2face-3d` の pull 権限が確認できる

注意:

- `412 Precondition Failed` や `Please accept license on the browser` が出る場合は、NGC/NVIDIA のブラウザ画面で Audio2Face-3D NIM の利用規約承認が未完了です。
- この確認は install 前に行うと、長い deployment の途中で失敗するリスクを減らせます。

### 15.7 Tokkio を install する

実行ディレクトリ: `/home/kyano/workspace/ACE/ace_kagawa`

```bash
./infra/tokkio/deploy_tokkio.sh install --env-file infra/tokkio/.env
```

注意:

- この操作は Kubernetes、container runtime、image pull、Helm chart、Tokkio services などに大きな変更を加えます。
- 共有 workstation では、既存 workload や Docker / Kubernetes の状態に影響します。
- install log に出る `ui_endpoint`、`api_endpoint`、Grafana endpoint を控えてください。

## 16. デプロイ手順

### 16.1 Tokkio 初回デプロイ

実行ディレクトリ: `/home/kyano/workspace/ACE/ace_kagawa`

```bash
cp infra/tokkio/.env.example infra/tokkio/.env
$EDITOR infra/tokkio/.env

python3 infra/tokkio/prepare_tokkio_workspace.py --env-file infra/tokkio/.env

git clone https://github.com/NVIDIA/ACE.git /home/kyano/workspace/ACE/ace_kagawa/infra/tokkio/workspace/NVIDIA-ACE

./infra/tokkio/deploy_tokkio.sh init-config --env-file infra/tokkio/.env
$EDITOR /home/kyano/workspace/ACE/ace_kagawa/infra/tokkio/workspace/controller/ace-app-config.yml

python3 infra/tokkio/check_tokkio_ngc_access.py \
  --env-file /home/kyano/workspace/ACE/ace_kagawa/infra/tokkio/workspace/controller/generated/my-config.env

./infra/tokkio/deploy_tokkio.sh install --env-file infra/tokkio/.env
```

### 16.2 Day-2 operations

このリポジトリから読み取れる事実:

- `manage_tokkio.sh start` は `containerd`、`kubelet`、`nginx`、`coturn` を起動してから Pod と endpoint を確認する。
- `stop` は非破壊停止で、app namespace の Deployment / StatefulSet を `replicas=0` へ落として GPU workload を静止する。
- replica 数は `/home/kyano/workspace/ACE/ace_kagawa/infra/tokkio/workspace/controller/generated/app-workload-replicas.tsv` に保存され、次の `start` で復元される。

実行ディレクトリ: `/home/kyano/workspace/ACE/ace_kagawa/infra/tokkio`

```bash
./manage_tokkio.sh start --env-file .env
./manage_tokkio.sh status --env-file .env
./manage_tokkio.sh stop --env-file .env
```

再適用:

```bash
./manage_tokkio.sh reapply --env-file .env
```

controller restart:

```bash
./manage_tokkio.sh restart-controller --env-file .env
./manage_tokkio.sh logs controller --env-file .env
```

### 16.3 Kubernetes 状態確認

実行ディレクトリ: 任意

```bash
kubectl cluster-info
kubectl get nodes -o wide
kubectl get namespaces
kubectl get pods -n app -o wide
kubectl get svc -n app
kubectl get ingress -A
kubectl get events -n app --sort-by=.lastTimestamp | tail -n 50
```

GPU resource 確認:

```bash
kubectl describe nodes | grep -A8 -B2 'nvidia.com/gpu'
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable.nvidia\\.com/gpu
```

Pod 詳細:

```bash
kubectl describe pod -n app <pod-name>
kubectl logs -n app <pod-name> --tail=200
```

StatefulSet / Deployment:

```bash
kubectl get deploy,statefulset -n app
kubectl rollout status deployment -n app <deployment-name>
```

### 16.4 Endpoint 確認

このリポジトリから読み取れる事実:

- `check_tokkio_endpoints.py` は UI/API/Grafana と `kubectl get pods -n app` の簡易確認を行う。
- 過去の single workstation 検証では、install output が API を `https` として示しても、実際の確認では `http://<app-ip>:30888` が必要な場合があった。

実行ディレクトリ: `/home/kyano/workspace/ACE/ace_kagawa`

```bash
python3 infra/tokkio/check_tokkio_endpoints.py \
  --insecure \
  --kubectl \
  --ui-url https://<app-ip>:30111 \
  --api-url http://<app-ip>:30888 \
  --grafana-url http://<app-ip>:32300
```

ブラウザで確認:

```text
https://<app-ip>:30111
http://<app-ip>:32300
http://<app-ip>:30180
```

注意:

- self-signed certificate の場合、ブラウザで警告が出ることがあります。
- `http` / `https` の mismatch は false negative の原因になります。install output と実際の応答を両方確認してください。

## 17. 動作確認手順

### 17.1 Tokkio UI で会話できるか

確認手順:

1. `kubectl get pods -n app` で全 Pod が `Running` または ready になっていることを確認する
2. `check_tokkio_endpoints.py` で UI/API/Grafana を確認する
3. ブラウザで Tokkio UI を開く
4. camera / microphone permission を許可する
5. avatar が表示されることを確認する
6. マイクで話す
7. transcript が更新されるか確認する
8. avatar が音声で応答するか確認する
9. Grafana や Pod logs で error が出ていないか確認する

コマンド:

```bash
kubectl get pods -n app -o wide
python3 infra/tokkio/check_tokkio_endpoints.py \
  --insecure \
  --kubectl \
  --ui-url https://<app-ip>:30111 \
  --api-url http://<app-ip>:30888 \
  --grafana-url http://<app-ip>:32300
kubectl logs -n app ace-controller-ace-controller-deployment-0 --tail=200
```

### 17.2 Audio2Face-3D が正常か

確認ポイント:

- A2F Pod が Running か
- A2F image pull が license error で止まっていないか
- A2F service が readiness / health check に通っているか
- avatar の口形が audio と同期しているか

コマンド:

```bash
kubectl get pods -n app | grep -i -E 'a2f|audio'
kubectl describe pod -n app <a2f-pod-name>
kubectl logs -n app <a2f-pod-name> --tail=200
```

### 17.3 ASR / TTS / LLM の切り分け

ASR 確認:

```bash
kubectl logs -n app ace-controller-ace-controller-deployment-0 --tail=300 | grep -i -E 'asr|transcript|riva|speech'
```

LLM 確認:

```bash
kubectl logs -n app ace-controller-ace-controller-deployment-0 --tail=300 | grep -i -E 'llm|nvidia|nim|openai|rag'
```

TTS 確認:

```bash
kubectl logs -n app ace-controller-ace-controller-deployment-0 --tail=300 | grep -i -E 'tts|eleven|speech|audio'
```

このリポジトリから読み取れる事実:

- 過去の Tokkio 5.0 single-workstation runbook では、ASR が動いていても response audio が返らない場合、`ace-controller` 側の TTS、特に ElevenLabs key を確認する必要があると記録されている。
- `config.yaml` が `NvidiaLLMService` を使っていても、TTS が ElevenLabs に依存するケースがある。

### 17.4 独自 orchestrator の mock 検証

実行ディレクトリ: `/home/kyano/workspace/ACE/ace_kagawa/services/orchestrator`

```bash
source .venv/bin/activate
python3 tools/init_storage.py
ACE_MOCK_ASR=true ACE_MOCK_LLM=true ACE_MOCK_TTS=true \
  uvicorn app.main:app --host 0.0.0.0 --port 8080
```

別 terminal:

```bash
curl -s http://127.0.0.1:8080/status
python3 tools/demo_client.py --url ws://127.0.0.1:8080/ws/session --mock-audio
```

テスト:

```bash
python3 -m unittest discover -s tests
```

### 17.5 Speech NIM compose 検証

実行ディレクトリ: `/home/kyano/workspace/ACE/ace_kagawa/infra/compose`

```bash
docker compose --env-file .env -f docker-compose.yml ps
python3 check_nim_stack.py --asr-http-url http://127.0.0.1:9000 --tts-http-url http://127.0.0.1:9001
python3 check_nim_stack.py --tts-grpc 127.0.0.1:50052 --tts-text "こんにちは、音声合成の確認です。"
```

## 18. スクリーンショットや画像を入れるべき箇所

### 18.1 NVIDIA ACE / Tokkio 公式ドキュメント

![NVIDIA Tokkio 5.0 公式 Overview](images/nvidia_tokkio_5_overview.png)

画像メモ:

- 内容: NVIDIA Tokkio 5.0 公式 Overview ページ
- 到達方法: ブラウザで https://docs.nvidia.com/ace/tokkio/5.0/overview/overview.html を開く
- 必要なコマンド: なし
- Web サイト URL: https://docs.nvidia.com/ace/tokkio/5.0/overview/overview.html
- 撮影する場面: 左側ナビゲーションと Overview 本文が見える状態
- Markdown プレースホルダ: `![NVIDIA Tokkio 5.0 公式 Overview](images/nvidia_tokkio_5_overview.png)`

### 18.2 Tokkio Architecture

![NVIDIA Tokkio 5.0 Architecture ページ](images/nvidia_tokkio_5_architecture.png)

画像メモ:

- 内容: Tokkio 5.0 architecture docs の workflow / pipeline 説明
- 到達方法: ブラウザで https://docs.nvidia.com/ace/tokkio/5.0/overview/architecture.html を開く
- 必要なコマンド: なし
- Web サイト URL: https://docs.nvidia.com/ace/tokkio/5.0/overview/architecture.html
- 撮影する場面: Workflow の番号付き説明、または architecture diagram が見える状態
- Markdown プレースホルダ: `![NVIDIA Tokkio 5.0 Architecture ページ](images/nvidia_tokkio_5_architecture.png)`

### 18.3 Audio2Face-3D docs

![Audio2Face-3D 公式ドキュメント](images/nvidia_audio2face_3d_docs.png)

画像メモ:

- 内容: Audio2Face-3D docs のトップまたは architecture / deployment navigation
- 到達方法: ブラウザで https://docs.nvidia.com/ace/audio2face-3d-microservice/latest/index.html を開く
- 必要なコマンド: なし
- Web サイト URL: https://docs.nvidia.com/ace/audio2face-3d-microservice/latest/index.html
- 撮影する場面: Getting Started、Architecture、Deployment、Troubleshooting が見える左 nav
- Markdown プレースホルダ: `![Audio2Face-3D 公式ドキュメント](images/nvidia_audio2face_3d_docs.png)`

### 18.4 NVIDIA GPU Operator docs

![NVIDIA GPU Operator 公式ドキュメント](images/nvidia_gpu_operator_docs.png)

画像メモ:

- 内容: NVIDIA GPU Operator docs の About the Operator ページ
- 到達方法: ブラウザで https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/index.html を開く
- 必要なコマンド: なし
- Web サイト URL: https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/index.html
- 撮影する場面: GPU Operator が driver、device plugin、container toolkit、GFD、DCGM を管理する説明が見える状態
- Markdown プレースホルダ: `![NVIDIA GPU Operator 公式ドキュメント](images/nvidia_gpu_operator_docs.png)`

### 18.5 Host GPU 確認

![nvidia-smi の実行結果](images/nvidia_smi_result.png)

画像メモ:

- 内容: host terminal で `nvidia-smi` を実行し、GPU、driver、CUDA version、memory が見える画面
- 到達方法: terminal を開く
- 必要なコマンド:

```bash
nvidia-smi
```

- Web サイト URL: なし
- 撮影する場面: `nvidia-smi` の表全体が見える状態
- Markdown プレースホルダ: `![nvidia-smi の実行結果](images/nvidia_smi_result.png)`

### 18.6 Docker GPU 確認

![Docker から GPU を使った nvidia-smi](images/docker_gpu_nvidia_smi.png)

画像メモ:

- 内容: Docker container 内で `nvidia-smi` が成功した画面
- 到達方法: terminal で Docker GPU 確認コマンドを実行する
- 必要なコマンド:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

- Web サイト URL: なし
- 撮影する場面: container 内 `nvidia-smi` の結果が表示され、エラーがない状態
- Markdown プレースホルダ: `![Docker から GPU を使った nvidia-smi](images/docker_gpu_nvidia_smi.png)`

### 18.7 Kubernetes Pod 状態

![Tokkio app namespace の Pod 一覧](images/tokkio_kubectl_get_pods_app.png)

画像メモ:

- 内容: `kubectl get pods -n app -o wide` の実行結果
- 到達方法: Tokkio install 後、terminal で Kubernetes 状態確認を実行する
- 必要なコマンド:

```bash
kubectl get pods -n app -o wide
```

- Web サイト URL: なし
- 撮影する場面: `ace-controller`、`a2f`、`animation-graph`、`unreal-renderer`、`riva-speech`、`tokkio-ui` などが Running になっている状態
- Markdown プレースホルダ: `![Tokkio app namespace の Pod 一覧](images/tokkio_kubectl_get_pods_app.png)`

### 18.8 Tokkio Web UI

![Tokkio Web UI とアバター表示](images/tokkio_web_ui_avatar.png)

画像メモ:

- 内容: ブラウザで Tokkio Web UI を開き、avatar が表示されている画面
- 到達方法: Tokkio install 後、install output または `.env` の UI URL をブラウザで開く
- 必要なコマンド:

```bash
python3 infra/tokkio/check_tokkio_endpoints.py \
  --insecure \
  --kubectl \
  --ui-url https://<app-ip>:30111 \
  --api-url http://<app-ip>:30888 \
  --grafana-url http://<app-ip>:32300
```

- Web サイト URL: `https://<app-ip>:30111`
- 撮影する場面: avatar が表示され、マイク許可後に会話可能な状態
- Markdown プレースホルダ: `![Tokkio Web UI とアバター表示](images/tokkio_web_ui_avatar.png)`

### 18.9 Grafana / observability

![Tokkio Grafana ダッシュボード](images/tokkio_grafana_dashboard.png)

画像メモ:

- 内容: Tokkio / Kubernetes / GPU / service metrics の Grafana 画面
- 到達方法: Tokkio install 後、Grafana endpoint をブラウザで開く
- 必要なコマンド:

```bash
python3 infra/tokkio/check_tokkio_endpoints.py \
  --insecure \
  --grafana-url http://<app-ip>:32300
```

- Web サイト URL: `http://<app-ip>:32300`
- 撮影する場面: Pod や service metrics が表示されている状態
- Markdown プレースホルダ: `![Tokkio Grafana ダッシュボード](images/tokkio_grafana_dashboard.png)`

### 18.10 Unreal サンドボックス

![Unreal ACEAvatarSandbox プロジェクト](images/unreal_ace_avatar_sandbox.png)

画像メモ:

- 内容: Unreal Engine 5.6 で `ACEAvatarSandbox.uproject` を開いた画面
- 到達方法: Unreal Engine 5.6 で `/home/kyano/workspace/ACE/ace_kagawa/unreal/ACEAvatarSandbox/ACEAvatarSandbox.uproject` を開く
- 必要なコマンド:

```bash
ls -lah /home/kyano/workspace/ACE/ace_kagawa/unreal/ACEAvatarSandbox/ACEAvatarSandbox.uproject
```

- Web サイト URL: なし
- 撮影する場面: project が開き、ACEConversation plugin または Character skeleton が確認できる状態
- Markdown プレースホルダ: `![Unreal ACEAvatarSandbox プロジェクト](images/unreal_ace_avatar_sandbox.png)`

### 18.11 Orchestrator status

![orchestrator status endpoint](images/orchestrator_status_endpoint.png)

画像メモ:

- 内容: `services/orchestrator` の `/status` response
- 到達方法: orchestrator を起動し、curl で `/status` を確認する
- 必要なコマンド:

```bash
cd /home/kyano/workspace/ACE/ace_kagawa/services/orchestrator
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8080
curl -s http://127.0.0.1:8080/status
```

- Web サイト URL: `http://127.0.0.1:8080/status`
- 撮影する場面: ASR / TTS / LLM の接続状態や mock 設定が JSON で表示されている状態
- Markdown プレースホルダ: `![orchestrator status endpoint](images/orchestrator_status_endpoint.png)`

## 19. トラブルシューティング

### 19.1 host では `nvidia-smi` が動くが Docker では GPU が見えない

症状:

```text
could not select device driver "" with capabilities: [[gpu]]
```

確認:

```bash
nvidia-smi
docker info | grep -i -E 'runtimes|nvidia|default runtime' -A3
which nvidia-ctk
nvidia-ctk --version
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

推奨対応:

- NVIDIA Container Toolkit が入っているか確認する
- Docker runtime に NVIDIA runtime が登録されているか確認する
- 共有 host では管理者と合意してから `sudo nvidia-ctk runtime configure --runtime=docker` と Docker restart を検討する

### 19.2 Kubernetes Pod が Pending のまま

確認:

```bash
kubectl get pods -n app
kubectl describe pod -n app <pod-name>
kubectl get events -n app --sort-by=.lastTimestamp | tail -n 50
kubectl describe nodes | grep -A8 -B2 'nvidia.com/gpu'
```

よくある原因:

- `nvidia.com/gpu` が node に allocatable として出ていない
- GPU が不足している
- node selector / affinity が合っていない
- image pull secret がない
- taint / toleration が合っていない
- PersistentVolume / storage class が用意されていない

### 19.3 Audio2Face-3D image pull / license で失敗する

確認:

```bash
python3 infra/tokkio/check_tokkio_ngc_access.py \
  --env-file /home/kyano/workspace/ACE/ace_kagawa/infra/tokkio/workspace/controller/generated/my-config.env

kubectl get pods -n app | grep -i a2f
kubectl describe pod -n app <a2f-pod-name>
```

推奨対応:

- NGC API key が正しいか確認する
- Audio2Face-3D NIM の license を NVIDIA / NGC browser 上で承認する
- private registry / proxy / firewall の制約を確認する

### 19.4 UI は開くが avatar が話さない

切り分け:

```bash
kubectl logs -n app ace-controller-ace-controller-deployment-0 --tail=300
kubectl logs -n app <riva-speech-pod-name> --tail=200
kubectl logs -n app <a2f-pod-name> --tail=200
```

確認観点:

- ASR transcript は出ているか
- LLM response は生成されているか
- TTS provider key が正しいか
- ElevenLabs / Speech NIM / other TTS の API error がないか
- Audio2Face-3D に audio response が届いているか
- renderer 側で animation data / audio stream が受け取れているか

このリポジトリから読み取れる事実:

- 過去の runbook では、ASR が動いていても TTS が ElevenLabs key で失敗すると spoken response が返らないケースが記録されている。

### 19.5 API endpoint が https と http で食い違う

症状:

- install output は `https://<app-ip>:30888` を示す
- 実際には `http://<app-ip>:30888` で応答する

確認:

```bash
curl -k -I https://<app-ip>:30888 || true
curl -I http://<app-ip>:30888 || true
python3 infra/tokkio/check_tokkio_endpoints.py \
  --insecure \
  --api-url http://<app-ip>:30888
```

推奨対応:

- `check_tokkio_endpoints.py` に実際に応答する protocol を指定する
- Ingress / nginx / service の TLS termination 設定を確認する
- false negative を避けるため UI/API/Grafana それぞれで `http` / `https` を検証する

### 19.6 orchestrator の mock は動くが実 ASR/TTS が動かない

確認:

```bash
cd /home/kyano/workspace/ACE/ace_kagawa/services/orchestrator
source .venv/bin/activate
curl -s http://127.0.0.1:8080/status
python3 tools/demo_client.py --url ws://127.0.0.1:8080/ws/session --mock-audio
```

Speech NIM 側:

```bash
cd /home/kyano/workspace/ACE/ace_kagawa/infra/compose
docker compose --env-file .env -f docker-compose.yml ps
python3 check_nim_stack.py --asr-http-url http://127.0.0.1:9000 --tts-http-url http://127.0.0.1:9001
```

確認観点:

- `services/orchestrator/.env` の ASR/TTS endpoint が compose の port と一致しているか
- `ACE_MOCK_ASR`、`ACE_MOCK_LLM`、`ACE_MOCK_TTS` が意図せず true / false になっていないか
- NGC key が compose `.env` に正しく入っているか
- GPU index が compose と実機で合っているか

## 20. セキュリティと認証情報の扱い

### 20.1 secret を commit しない

対象:

- `infra/tokkio/.env`
- `infra/compose/.env`
- `services/orchestrator/.env`
- NGC API key
- NVIDIA API key
- ElevenLabs API key
- OpenAI API key
- SSH private key
- TLS private key

確認:

```bash
git status --short
git diff -- . ':!*.lock'
git ls-files | grep -E '(^|/)\.env$|id_rsa|id_ed25519|secret|token|key'
```

注意:

- `.env.example` には placeholder のみ入れる。
- 実 secret を含む `.env` は `.gitignore` に入れる。
- 共有するドキュメントには `<YOUR_KEY>` のような placeholder を使う。

### 20.2 Kubernetes Secret の注意点

Kubernetes 公式 docs で確認した事実:

- Secret は password、token、key などの sensitive data を保持する object である。
- Secret は default では etcd に暗号化されず保存される。
- Secret への広い RBAC 権限は、namespace 内の secret 全体を読めてしまうリスクがある。

一般的な構成としての推奨:

- production では Kubernetes secret encryption at rest を有効化する。
- RBAC は最小権限にする。
- secret を environment variable として渡すか volume mount するかをサービスごとに決める。
- external secret manager の利用を検討する。
- API key を rotate できる手順を runbook 化する。

Secret 作成例:

```bash
kubectl create namespace ace-app
kubectl create secret generic ace-api-keys \
  --namespace ace-app \
  --from-literal=ngc_api_key='<YOUR_NGC_API_KEY>' \
  --from-literal=nvidia_api_key='<YOUR_NVIDIA_API_KEY>' \
  --dry-run=client -o yaml > /tmp/ace-api-keys.secret.yaml
```

注意:

- `/tmp/ace-api-keys.secret.yaml` には secret が base64 で含まれます。安全ではありません。
- 実運用では file に残さず `kubectl create secret ...` を直接適用するか、secret manager を使ってください。

### 20.3 Network / TLS

一般的な構成としての推奨:

- Web UI、API、Grafana、ACE Configurator を外部公開する場合は、TLS と認証を必ず設計する。
- Grafana や configurator を public network にそのまま出さない。
- Ingress controller、cert-manager、WAF、VPN、SSH tunnel の利用を検討する。
- `coturn` は NAT traversal に必要だが、公開範囲と credential 管理に注意する。

## 21. コストと必要リソース

### 21.1 GPU / storage

NVIDIA 公式 docs で確認できた事実:

- Tokkio 5.0 quickstart の Application Instance は、少なくとも 2x L4 または 2x A10 GPU、700 GB 以上の storage を必要とする。
- Audio2Face-3D は GPU profile によって pre-generated TRT engine を利用できる場合と、local build が必要な場合がある。
- Speech NIM / LLM NIM / A2F / renderer は GPU memory と storage cache を消費する。

このリポジトリから読み取れる事実:

- 大きな cache、logs、audio、UE DDC、Docker、Tokkio state は `/home2/ko66/ace-sandbox` 配下に逃がす方針である。
- `infra/compose` の ASR/TTS NIM cache は `/home2/ko66/ace-sandbox/nim-cache/asr` と `/home2/ko66/ace-sandbox/nim-cache/tts` に置く。

一般的な構成としての推奨:

- 開発検証でも 700 GB 以上の空き容量を見込む。
- image pull、model cache、TRT engine、logs、UE DDC は repo 配下に置かない。
- GPU 1 枚構成で Tokkio full stack を動かすのは厳しい可能性が高い。最小検証は service 単体に分ける。
- LLM NIM を self-host する場合は、model size に応じて VRAM と multi-GPU 設計を別途見積もる。

### 21.2 外部 API cost

費用が発生しうるもの:

- NVIDIA hosted API / NIM API 利用
- NGC / NVIDIA AI Enterprise entitlement
- ElevenLabs TTS
- OpenAI API
- cloud GPU instance
- managed Kubernetes
- storage / egress / TURN traffic

一般的な構成としての推奨:

- 開発環境では API usage limit を設定する。
- TTS / LLM は log に token count、request count、latency、error を残す。
- 大量テスト前に cost guardrail を入れる。

## 22. 参考リンク

### 22.1 NVIDIA 公式

- NVIDIA Tokkio 5.0 Overview: https://docs.nvidia.com/ace/tokkio/5.0/overview/overview.html
- NVIDIA Tokkio 5.0 Architecture: https://docs.nvidia.com/ace/tokkio/5.0/overview/architecture.html
- NVIDIA Tokkio 5.0 Quickstart Guide: https://docs.nvidia.com/ace/tokkio/5.0/quickstart-guide.html
- NVIDIA Tokkio 5.0 Unreal Renderer Microservice Introduction: https://docs.nvidia.com/ace/tokkio/5.0/reference-workflow/tokkio-ue.html
- NVIDIA ACE Agent latest docs: https://docs.nvidia.com/ace/ace-agent/latest/index.html
- NVIDIA Audio2Face-3D Microservice latest docs: https://docs.nvidia.com/ace/audio2face-3d-microservice/latest/index.html
- NVIDIA Audio2Face-3D NIM docs: https://docs.nvidia.com/nim/digital-human/a2f-3d/latest/index.html
- NVIDIA Animation Graph Microservice docs: https://docs.nvidia.com/ace/animation-graph-microservice/1.0/index.html
- NVIDIA Unreal Renderer Microservice docs: https://docs.nvidia.com/ace/unreal-renderer-microservice/latest/index.html
- NVIDIA Riva docs: https://docs.nvidia.com/riva/index.html
- NVIDIA Speech NIM Microservices docs: https://docs.nvidia.com/nim/speech/latest/index.html
- NVIDIA NIM docs: https://docs.nvidia.com/nim/
- NVIDIA NIM for LLMs latest overview: https://docs.nvidia.com/nim/large-language-models/latest/about-nim-llm/overview.html
- NVIDIA Cloud Native Technologies: https://docs.nvidia.com/datacenter/cloud-native/index.html
- NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html
- NVIDIA GPU Operator: https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/index.html
- NVIDIA GPU Operator install guide: https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/getting-started.html

### 22.2 Kubernetes / Helm 公式

- Kubernetes Deployments: https://kubernetes.io/docs/concepts/workloads/controllers/deployment/
- Kubernetes Services: https://kubernetes.io/docs/concepts/services-networking/service/
- Kubernetes Ingress: https://kubernetes.io/docs/concepts/services-networking/ingress/
- Kubernetes Namespaces: https://kubernetes.io/docs/concepts/overview/working-with-objects/namespaces/
- Kubernetes ConfigMaps: https://kubernetes.io/docs/concepts/configuration/configmap/
- Kubernetes Secrets: https://kubernetes.io/docs/concepts/configuration/secret/
- Kubernetes Device Plugins: https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/
- Kubernetes Schedule GPUs: https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/
- Helm docs: https://helm.sh/docs/

### 22.3 この repo 内

- `/home/kyano/workspace/ACE/ace_kagawa/README.md`
- `/home/kyano/workspace/ACE/ace_kagawa/docs/architecture/tokkio-reference-stack.md`
- `/home/kyano/workspace/ACE/ace_kagawa/docs/architecture/ace-sandbox.md`
- `/home/kyano/workspace/ACE/ace_kagawa/docs/operations/tokkio-rebuild-runbook.md`
- `/home/kyano/workspace/ACE/ace_kagawa/docs/operations/tokkio-webui-startup-runbook.md`
- `/home/kyano/workspace/ACE/ace_kagawa/docs/operations/tokkio-japanese-customization.md`
- `/home/kyano/workspace/ACE/ace_kagawa/infra/tokkio/README.md`
- `/home/kyano/workspace/ACE/ace_kagawa/infra/compose/README.md`
- `/home/kyano/workspace/ACE/ace_kagawa/services/orchestrator/README.md`
- `/home/kyano/workspace/ACE/ace_kagawa/unreal/README.md`
- `/home/kyano/workspace/ACE/ace_kagawa/tts/README.md`

## 23. 用語集

| 用語 | 説明 |
| --- | --- |
| ACE | NVIDIA Avatar Cloud Engine / digital human 関連技術群として扱われる NVIDIA のサービス・マイクロサービス群 |
| Tokkio | NVIDIA ACE 技術を使った interactive avatar の reference workflow / AI Blueprint |
| ACE Agent | Bot / conversational agent の構築、Colang、LangChain、RAG などを扱う ACE 関連コンポーネント |
| ACE Controller | Tokkio 内で ASR、LLM/RAG、TTS、avatar control、UI 連携などを orchestrate する中核 service |
| Audio2Face-3D / A2F | 音声から avatar の facial animation を生成する NVIDIA NIM / microservice |
| Animation Graph | 姿勢、gesture、facial gesture、position などを組み合わせる animation framework / microservice |
| Unreal Renderer Microservice | Unreal Engine で avatar scene を render し、WebRTC で stream する microservice |
| Riva | NVIDIA の speech AI SDK。ASR、TTS、NLP などを提供 |
| NIM | NVIDIA Inference Microservice。AI model を optimized container / API として提供する仕組み |
| Speech NIM | ASR、TTS、NMT など speech AI の NIM |
| LLM NIM | Large Language Model を運用するための NIM |
| ASR | Automatic Speech Recognition。音声認識 |
| TTS | Text-To-Speech。音声合成 |
| LLM | Large Language Model |
| RAG | Retrieval-Augmented Generation。検索・知識ベースを使って LLM 応答を補強する方式 |
| VST | Video Storage Toolkit。Tokkio で stream lifecycle や media handling に関わる |
| SDR | Stream Distribution and Routing。Tokkio で stream を microservice / GPU に分配する仕組み |
| WebRTC | 低遅延の audio/video communication protocol |
| WebSocket | 双方向通信。signaling、status、transcript、独自 protocol に使われる |
| gRPC | 高性能 RPC framework。NIM / Riva / A2F / renderer などで使われる |
| Kubernetes / k8s | container orchestration platform。「クバネティス」と呼ばれるもの |
| Pod | Kubernetes で container を動かす最小単位 |
| Deployment | stateless application workload の Pod 群を管理する Kubernetes resource |
| StatefulSet | stateful workload を管理する Kubernetes resource |
| Service | Pod 群を安定した network endpoint として公開する Kubernetes resource |
| Ingress | cluster 外から HTTP/HTTPS で Service に routing する Kubernetes resource |
| Namespace | Kubernetes resource group を分離する仕組み |
| ConfigMap | non-confidential configuration を key-value で保持する Kubernetes resource |
| Secret | password、token、key など sensitive data を保持する Kubernetes resource |
| Helm | Kubernetes application package manager |
| GPU Operator | Kubernetes で NVIDIA GPU を利用するための driver/toolkit/device plugin/monitoring などを自動管理する operator |
| Device Plugin | Kubernetes node の特殊 device を kubelet に登録し、`nvidia.com/gpu` などとして公開する仕組み |
| NVIDIA Container Toolkit | Docker/containerd などから NVIDIA GPU を使うための runtime/tool 群 |
| DCGM | NVIDIA Data Center GPU Manager。GPU metrics / monitoring に使われる |
| GFD | GPU Feature Discovery。GPU node labels を付けるための component |
| NGC | NVIDIA GPU Cloud / NVIDIA catalog。container image や model を取得する |
| coturn | TURN server。WebRTC の NAT traversal に使う |
| MetaHuman | Unreal Engine / Epic の high-fidelity digital human asset system |
| Pixel Streaming | Unreal Engine の render result を WebRTC などで stream する技術 |

## 24. 未確認事項と次に確認すべきこと

このドキュメント作成時点で、以下は環境依存または未確認です。

- 現在の実機で Tokkio 5.0 full stack が再デプロイできるか
- 現在の NGC / NVIDIA API key が Tokkio chart、Audio2Face-3D NIM、Speech NIM image にアクセスできるか
- 現在の GPU driver、container runtime、Kubernetes version が最新 NVIDIA docs の support matrix と一致しているか
- Tokkio 5.0 の既定 TTS provider を ElevenLabs から Speech NIM に置き換える場合の具体設定
- NVIDIA ACE Agent をこの repo に導入する場合の directory design
- 本番用 Ingress TLS、認証、RBAC、secret encryption at rest、network policy の具体 manifest
- custom MetaHuman を Tokkio Unreal Renderer Microservice に組み込む具体 asset pipeline
- `/home2/ko66/ace-sandbox` の容量、backup、cleanup policy
- `tts/` の Sarashina workflow を ACE 本線に接続する場合のライセンス・商用利用・watermark・latency 条件

## 25. このドキュメントの更新ルール

一般的な構成としての推奨:

- NVIDIA docs は更新が速いため、Tokkio、Audio2Face-3D、Speech NIM、LLM NIM、GPU Operator、Container Toolkit は作業前に公式 URL を再確認する。
- この repo の helper script や runbook が更新された場合、`このリポジトリから読み取れる事実` の節を更新する。
- スクリーンショットを追加したら、`images/` 配下に保存し、撮影時のコマンド・URL・状態をこの文書に残す。
- secret、実 IP、private endpoint、API key を画像や Markdown に含めない。
