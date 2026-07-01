# Tokkio WebUI 起動運用手順書

## 1. 目的

この文書は、**すでに一度デプロイ済みの Tokkio 5.0 環境**を、**すべて停止した状態から WebUI 表示まで立ち上げる**ための運用手順書です。

対象は以下です。

- OS 再起動後に Tokkio を起動する
- `containerd` / `kubelet` / `nginx` / `coturn` が止まった状態から復旧する
- Kubernetes Pod が揃わないときに確認箇所を切る
- 必要な場合のみ Tokkio app/controller を再適用する

対象外:

- ホスト再プロビジョニング
- 初回のフル再構築
- Tokkio 未導入状態からの新規構築

初回構築や完全再構築が必要な場合は、[tokkio-rebuild-runbook.md](/home/kyano/workspace/ACE/ace_kagawa/docs/operations/tokkio-rebuild-runbook.md:1) を参照してください。

## 2. 対象環境

- 作業ディレクトリ: `/home/kyano/workspace/ACE/ace_kagawa`
- Tokkio 以外の大きなデータ・生成物・キャッシュ: `/home2/ko66`
- Tokkio workspace: `/home/kyano/workspace/ACE/ace_kagawa/infra/tokkio/workspace`
- 公式 ACE repo: `/home/kyano/workspace/ACE/ace_kagawa/infra/tokkio/workspace/NVIDIA-ACE`
- branch: `5.0.0-ga`
- profile: `tokkio-1stream`
- 単一ワークステーション構成
- APP host IP: `10.209.1.12`
- SSH user: `kyano`

既知の到達先:

- UI: `https://10.209.1.12:30111`
- API: install 表示は `https://10.209.1.12:30888` だが、実運用確認は `http://10.209.1.12:30888`
- Grafana: `http://10.209.1.12:32300`
- ACE Configurator: `http://10.209.1.12:30180`

前提:

- `passwordless sudo` が使える
- `kyano@10.209.1.12` への self-SSH が使える
- `infra/tokkio/.env` と generated env が残っている
- 既存の Kubernetes クラスタが壊れていない

## 3. 関連ファイル

- `infra/tokkio/.env`
- `infra/tokkio/prepare_tokkio_workspace.py`
- `infra/tokkio/check_tokkio_ngc_access.py`
- `infra/tokkio/check_tokkio_endpoints.py`
- `infra/tokkio/deploy_tokkio.sh`
- `infra/tokkio/manage_tokkio.sh`
- `/home/kyano/workspace/ACE/ace_kagawa/infra/tokkio/workspace/controller/generated/my-config.env`
- `/home/kyano/workspace/ACE/ace_kagawa/infra/tokkio/workspace/controller/ace-app-config.yml`

## 4. 起動の全体像

この環境では、WebUI 表示までの起動順は次の理解でよいです。

1. ホスト側の基盤サービスを起動する
   - `containerd`
   - `kubelet`
   - `nginx`
   - `coturn`
2. Kubernetes 上の `app` namespace の Pod が揃うのを待つ
3. UI / API / Grafana を確認する
4. ブラウザで UI を開く
5. 音声応答まで確認する

重要:

- `nginx` は Tokkio UI 配信用
- `coturn` は WebRTC 用
- Tokkio 本体は Kubernetes Pod 群として動く

## 5. 最短起動手順

まずはこれを実行します。

```bash
./infra/tokkio/manage_tokkio.sh start --env-file infra/tokkio/.env
```

このコマンドは次をまとめて行います。

- `prepare_tokkio_workspace.py` で generated env を同期
- `sudo systemctl start containerd kubelet nginx coturn`
- `kubectl get pods -n app`
- `check_tokkio_endpoints.py --insecure --kubectl`
- UI / API / Grafana の確認 URL 表示

日常運用でよく使う補助コマンド:

```bash
./infra/tokkio/manage_tokkio.sh status --env-file infra/tokkio/.env
./infra/tokkio/manage_tokkio.sh stop --env-file infra/tokkio/.env
./infra/tokkio/manage_tokkio.sh restart --env-file infra/tokkio/.env
```

補足:

- `start` は既存インストール済み環境の復帰用で、`deploy_tokkio.sh install` は自動実行しません
- `stop` は非破壊停止のみで、Kubernetes リソースや永続データは消しません
- `stop` は host 側サービス停止の前に `app` namespace の `Deployment` / `StatefulSet` を `replicas=0` に落とし、GPU を使う Tokkio workload を静止します
- `start` は退避済み replica 数があれば `/home/kyano/workspace/ACE/ace_kagawa/infra/tokkio/workspace/controller/generated/app-workload-replicas.tsv` から復元します
- このホストが Tokkio 専用ワークステーションに近い前提で `containerd` と `kubelet` も停止します

### 5.1 手動でホスト側サービスを起動する場合

```bash
sudo systemctl start containerd
sudo systemctl start kubelet
sudo systemctl start nginx
sudo systemctl start coturn
```

状態確認:

```bash
systemctl is-active containerd kubelet nginx coturn
```

期待値:

- すべて `active`

### 5.2 Kubernetes Pod 状態を確認

```bash
kubectl get pods -n app
```

目安:

- `a2f`
- `ace-controller`
- `riva-speech`
- `tokkio-ui`
- `tokkio-ingress`
- `vms`
- `triton0`

が `Running` になっていること

必要なら `./infra/tokkio/manage_tokkio.sh status --env-file infra/tokkio/.env` でも同じ確認をまとめて行えます。

### 5.3 endpoint を確認

```bash
python3 infra/tokkio/check_tokkio_endpoints.py \
  --insecure \
  --kubectl \
  --ui-url https://10.209.1.12:30111 \
  --api-url http://10.209.1.12:30888 \
  --grafana-url http://10.209.1.12:32300
```

期待値:

- UI が `200`
- Grafana が `200`
- `kubectl_pods` が `ok: true`

補足:

- API は `404` でも、`http://10.209.1.12:30888` に応答していれば「死んでいる」とは限りません
- この環境では `https://10.209.1.12:30888` をそのまま信じないこと

### 5.4 ブラウザで確認

- `https://10.209.1.12:30111` を開く
- MetaHuman が見えることを確認する
- マイク入力が認識されることを確認する

## 6. 状態別の対処

### 6.1 OS 再起動直後

まずは次を実行します。

```bash
./infra/tokkio/manage_tokkio.sh start --env-file infra/tokkio/.env
```

手動で追う場合は以下だけを行います。

```bash
sudo systemctl start containerd
sudo systemctl start kubelet
sudo systemctl start nginx
sudo systemctl start coturn
kubectl get pods -n app
```

Pod が揃えば、そのまま endpoint 確認へ進みます。

### 6.2 `kubectl get pods -n app` が通らない

まずホスト基盤を確認します。

```bash
systemctl is-active containerd kubelet
```

必要ならログを確認します。

```bash
sudo journalctl -u containerd -n 100 --no-pager
sudo journalctl -u kubelet -n 100 --no-pager
```

`manage_tokkio.sh status` でも、`kubectl` が通らない場合に `containerd` / `kubelet` を先に見るヒントを出します。

### 6.3 一部の Pod だけ起動しない

対象 Pod を特定して確認します。

```bash
kubectl describe pod -n app <pod>
```

最低限見るべき Pod:

- `a2f-a2f-deployment-*`
- `ace-controller-ace-controller-deployment-0`
- `riva-speech-*`
- `tokkio-ui-*`
- `tokkio-ingress-*`

### 6.4 起動だけで戻らない

この場合は再適用に進みます。最短では次を使います。

```bash
./infra/tokkio/manage_tokkio.sh reapply --env-file infra/tokkio/.env
```

手動で実行する場合は以下です。

```bash
python3 infra/tokkio/prepare_tokkio_workspace.py --env-file infra/tokkio/.env
./infra/tokkio/deploy_tokkio.sh install --env-file infra/tokkio/.env
```

## 7. 必要時の再デプロイ手順

以下の条件なら再デプロイを使います。

- Pod が欠けたまま戻らない
- secret を更新した
- Tokkio app/controller の設定反映が必要
- 起動はしているが controller の状態が古い

### 7.1 generated env を作り直す

```bash
python3 infra/tokkio/prepare_tokkio_workspace.py --env-file infra/tokkio/.env
```

### 7.2 必要なら NGC / A2F 権限を確認

```bash
python3 infra/tokkio/check_tokkio_ngc_access.py \
  --env-file /home/kyano/workspace/ACE/ace_kagawa/infra/tokkio/workspace/controller/generated/my-config.env
```

### 7.3 Tokkio app を再適用

運用上は次を使うと `prepare` と `install` と `status` を一続きで実行できます。

```bash
./infra/tokkio/manage_tokkio.sh reapply --env-file infra/tokkio/.env
```

手動で分けて実行する場合は次です。

```bash
./infra/tokkio/deploy_tokkio.sh install --env-file infra/tokkio/.env
```

### 7.4 controller に secret を再読込させる

特に ElevenLabs key を更新した時は、`ace-controller` Pod の再作成が必要です。

最短手順:

```bash
./infra/tokkio/manage_tokkio.sh restart-controller --env-file infra/tokkio/.env
./infra/tokkio/manage_tokkio.sh logs controller --env-file infra/tokkio/.env
```

手動手順:

```bash
kubectl delete pod -n app ace-controller-ace-controller-deployment-0
```

その後:

```bash
kubectl get pods -n app
kubectl logs -n app ace-controller-ace-controller-deployment-0 --tail=200
```

必要に応じて次も使えます。

```bash
./infra/tokkio/manage_tokkio.sh logs riva --env-file infra/tokkio/.env
./infra/tokkio/manage_tokkio.sh logs a2f --env-file infra/tokkio/.env
```

## 8. WebUI 起動成功の判断基準

最低限の成功条件:

- `https://10.209.1.12:30111` が開く
- MetaHuman が表示される
- `kubectl get pods -n app` で主要 Pod が `Running`
- `a2f` が `Running`
- `riva-speech` が `Running`
- `ace-controller` が `Running`
- API は `http://10.209.1.12:30888` として到達確認できる

追加確認:

- 音声認識が通る
- `ace-controller` ログに `Invalid API key` が出ていない
- 音声応答が返る

## 9. 既知のハマりどころ

- Helm と `helm-diff` plugin の互換性問題があったため、ホスト Helm は新しめが必要
- NGC chart repo access が通っても、`Audio2Face-3D` image pull 権限は別に確認が必要
- API endpoint は `https` と表示されても実際は `http` の場合がある
- ElevenLabs secret を更新しても、`ace-controller` Pod を再起動しない限り古い環境変数を握り続ける
- Tokkio 5.0 参照構成はローカル `Riva TTS` に自動では切り替わらず、既定の TTS 実装は ElevenLabs 依存
- `Audio2Face-3D` の初回 pull は非常に大きく、約 `22.4 GB`、約 `5 分` かかった

## 10. 症状別の復旧手順

### 10.1 A2F が起動しない

典型症状:

- `a2f-a2f-deployment-*` が `Init:ImagePullBackOff`
- `kubectl describe` で `412 Precondition Failed`

確認:

```bash
kubectl describe pod -n app <pod>

python3 infra/tokkio/check_tokkio_ngc_access.py \
  --env-file /home/kyano/workspace/ACE/ace_kagawa/infra/tokkio/workspace/controller/generated/my-config.env
```

対処:

1. ブラウザで `Audio2Face-3D` NIM の license を承認する
2. helper の A2F image scope が通ることを確認する
3. Pod を引き直す

```bash
kubectl delete pod -n app <a2f-pod>
```

### 10.2 WebUI は出るが返答音声が返らない

典型症状:

- UI は開く
- MetaHuman は見える
- ASR は通る
- 返答音声が返らない

主因として確認済みのもの:

- `ace-controller` が TTS を ElevenLabs にハードコードしており、ElevenLabs API key が invalid

確認:

```bash
kubectl logs -n app ace-controller-ace-controller-deployment-0 --tail=200
```

見るべき文字列:

- `ElevenLabsTTSServiceWithEndOfSpeech`
- `Invalid API key`

対処:

1. `/home/kyano/workspace/ACE/ace_kagawa/infra/tokkio/.env` の `TOKKIO_ELEVENLABS_API_KEY` を更新
2. generated env を再生成
3. Tokkio app を再適用
4. `ace-controller` Pod を再起動

```bash
python3 infra/tokkio/prepare_tokkio_workspace.py --env-file infra/tokkio/.env
./infra/tokkio/deploy_tokkio.sh install --env-file infra/tokkio/.env
kubectl delete pod -n app ace-controller-ace-controller-deployment-0
```

### 10.3 secret を更新したのに挙動が変わらない

原因:

- `ace-controller` は secret を起動時に読む

対処:

```bash
kubectl delete pod -n app ace-controller-ace-controller-deployment-0
```

必要なら確認:

```bash
kubectl logs -n app ace-controller-ace-controller-deployment-0 --tail=200
```

### 10.4 API の protocol mismatch

症状:

- install 出力では `https://10.209.1.12:30888`
- 実際には `https` で失敗する

確認:

```bash
python3 infra/tokkio/check_tokkio_endpoints.py \
  --insecure \
  --kubectl \
  --ui-url https://10.209.1.12:30111 \
  --api-url https://10.209.1.12:30888 \
  --grafana-url http://10.209.1.12:32300
```

対処:

- このホストでは API を `http://10.209.1.12:30888` として扱う

## 11. 現時点の運用メモ

確認済み:

- UI は起動した
- MetaHuman は見えた
- A2F は license 承認後に復旧した
- ASR は `riva-speech` で通った
- ElevenLabs key の正しい反映先は `infra/tokkio/.env`
- `ace-controller` は secret 更新後に Pod 再作成が必要だった
- 新しい `ace-controller` Pod では `ELEVENLABS_API_KEY` が PID 1 の環境に載っていることを確認した

未確認:

- 最終的に返答音声が正常に返ったかは、ブラウザ側で再テストが必要

## Quick Start

```bash
./infra/tokkio/manage_tokkio.sh start --env-file infra/tokkio/.env
```

## 起動だけで戻らない時に使うコマンド

```bash
./infra/tokkio/manage_tokkio.sh reapply --env-file infra/tokkio/.env
python3 infra/tokkio/check_tokkio_ngc_access.py --env-file /home/kyano/workspace/ACE/ace_kagawa/infra/tokkio/workspace/controller/generated/my-config.env
./infra/tokkio/manage_tokkio.sh restart-controller --env-file infra/tokkio/.env
```

## First 5 Debug Commands

```bash
kubectl get pods -n app
kubectl describe pod -n app <pod>
kubectl logs -n app ace-controller-ace-controller-deployment-0 --tail=200
python3 infra/tokkio/check_tokkio_ngc_access.py --env-file /home/kyano/workspace/ACE/ace_kagawa/infra/tokkio/workspace/controller/generated/my-config.env
python3 infra/tokkio/check_tokkio_endpoints.py --insecure --kubectl --ui-url https://10.209.1.12:30111 --api-url http://10.209.1.12:30888 --grafana-url http://10.209.1.12:32300
```
