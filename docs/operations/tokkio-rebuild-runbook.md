# Tokkio 5.0 Single-Workstation Rebuild Runbook

## 1. Overview

This runbook documents how to rebuild the verified Tokkio 5.0 single-workstation setup on the Linux RTX host used during this project.

The goal is operational repeatability, not a historical log. Use this document to:

- prepare the workspace and controller config
- deploy the official Tokkio 5.0 one-click workflow
- verify the browser/UI/A2F/ASR path
- diagnose the two main failure classes encountered here:
  - `Audio2Face-3D` image pull/license failures
  - `ElevenLabs` TTS failures in `ace-controller`

This document does not contain any live secrets. Replace placeholders such as `<YOUR_KEY>` or `<REDACTED>` with your own values.

## 2. Target Environment

- Working repository: `/home/kyano/workspace/ACE`
- Large non-Tokkio data, generated artifacts, caches, and logs: `/home2/ko66`
- Tokkio workspace root: `/home/kyano/workspace/ACE/infra/tokkio/workspace`
- Official ACE repo clone: `/home/kyano/workspace/ACE/infra/tokkio/workspace/NVIDIA-ACE`
- Generated controller env: `/home/kyano/workspace/ACE/infra/tokkio/workspace/controller/generated/my-config.env`
- Tokkio branch/profile used here:
  - branch: `5.0.0-ga`
  - profile: `tokkio-1stream`
- Host topology: single workstation, controller and app on the same machine
- Verified host identity used during deployment:
  - `APP_HOST_IPV4_ADDR=10.209.1.12`
  - `APP_HOST_SSH_USER=kyano`
  - single-node deployment also reused the same machine for Coturn

Assumptions:

- Linux RTX workstation with working NVIDIA driver and Kubernetes-capable Tokkio prerequisites
- passwordless `sudo`
- self-SSH works for `kyano@10.209.1.12`

## 3. Directory Layout

Repository-side files:

- `/home/kyano/workspace/ACE/infra/tokkio/.env`
- `/home/kyano/workspace/ACE/infra/tokkio/.env.example`
- `/home/kyano/workspace/ACE/infra/tokkio/prepare_tokkio_workspace.py`
- `/home/kyano/workspace/ACE/infra/tokkio/check_tokkio_ngc_access.py`
- `/home/kyano/workspace/ACE/infra/tokkio/check_tokkio_endpoints.py`
- `/home/kyano/workspace/ACE/infra/tokkio/deploy_tokkio.sh`

Runtime-side files and directories:

- `/home/kyano/workspace/ACE/infra/tokkio/workspace/`
- `/home/kyano/workspace/ACE/infra/tokkio/workspace/controller/`
- `/home/kyano/workspace/ACE/infra/tokkio/workspace/controller/generated/my-config.env`
- `/home/kyano/workspace/ACE/infra/tokkio/workspace/controller/ace-app-config.yml`
- `/home/kyano/workspace/ACE/infra/tokkio/workspace/NVIDIA-ACE/`
- `/home/kyano/workspace/ACE/infra/tokkio/workspace/NVIDIA-ACE/workflows/tokkio/5.0.0-ga/scripts/one-click/baremetal/`

Operational rule:

- keep non-Tokkio heavy assets, model pulls, logs, and caches under `/home2/ko66`
- keep the Tokkio controller workspace and NVIDIA/ACE clone under `/home/kyano/workspace/ACE/infra/tokkio/workspace`
- keep only lightweight source/config in the repo under `/home/kyano/workspace/ACE`

## 4. Required Secrets and Accounts

Required accounts:

- NVIDIA API / NGC account with access to Tokkio charts and images
- NGC browser access capable of accepting the `Audio2Face-3D` NIM license
- ElevenLabs account and valid API key for TTS
- optional OpenAI account only if switching to `OpenAILLMService`

Required values for `infra/tokkio/.env`:

- `TOKKIO_NVIDIA_API_KEY=<YOUR_NVIDIA_API_KEY>`
- `TOKKIO_NGC_CLI_API_KEY=<YOUR_NGC_API_KEY>`
- `TOKKIO_ELEVENLABS_API_KEY=<YOUR_ELEVENLABS_API_KEY>`
- `TOKKIO_OPENAI_API_KEY=<YOUR_OPENAI_KEY_OR_LEAVE_EMPTY_IF_UNUSED>`
- `TOKKIO_APP_HOST_IPV4_ADDR=10.209.1.12`
- `TOKKIO_APP_HOST_SSH_USER=kyano`
- `TOKKIO_COTURN_HOST_IPV4_ADDR=10.209.1.12`
- `TOKKIO_COTURN_HOST_SSH_USER=kyano`

Important notes:

- The correct reflection point for the ElevenLabs key is `/home/kyano/workspace/ACE/infra/tokkio/.env` via `TOKKIO_ELEVENLABS_API_KEY`.
- `prepare_tokkio_workspace.py` generates `/home/kyano/workspace/ACE/infra/tokkio/workspace/controller/generated/my-config.env`.
- This repo’s helper fills an OpenAI placeholder when `TOKKIO_OPENAI_API_KEY` is empty, to satisfy Tokkio 5.0 secret validation while still using `NvidiaLLMService`.

## 5. Verified Working Architecture

Verified working pieces on this host:

- Tokkio 5.0 one-click deployment using official `NVIDIA/ACE`
- browser UI reachable
- MetaHuman visible in UI
- `Audio2Face-3D` reachable after license acceptance and initial image pull
- `riva-speech` ASR path working
- `ace-controller` using `NvidiaLLMService` for LLM

Verified service endpoints:

- UI: `https://10.209.1.12:30111`
- API reported by install: `https://10.209.1.12:30888`
- API used for practical verification: `http://10.209.1.12:30888`
- Grafana: `http://10.209.1.12:32300`
- ACE Configurator: `http://10.209.1.12:30180`

Important implementation fact:

- In this Tokkio 5.0 reference implementation, `LLM` and `TTS` are not coupled.
- `config.yaml` used `NvidiaLLMService`, but `ace-controller` `bot.py` still instantiated `ElevenLabsTTSServiceWithEndOfSpeech`.
- Result: even with local `riva-speech` working for ASR, response audio still depended on a valid ElevenLabs API key.

Inference:

- If the avatar hears the user and ASR updates appear, but no spoken response returns, inspect `ace-controller` TTS behavior before debugging ASR.

## 6. Step-by-Step Rebuild Procedure

### 6.1 Prepare repository-side config

From `/home/kyano/workspace/ACE`:

```bash
cp infra/tokkio/.env.example infra/tokkio/.env
```

Edit `infra/tokkio/.env` and set at minimum:

```dotenv
TOKKIO_ACE_BRANCH=5.0.0-ga
TOKKIO_PROFILE=tokkio-1stream

TOKKIO_WORKSPACE_DIR=/home/kyano/workspace/ACE/infra/tokkio/workspace
TOKKIO_ACE_REPO_DIR=/home/kyano/workspace/ACE/infra/tokkio/workspace/NVIDIA-ACE
TOKKIO_CONTROLLER_DIR=/home/kyano/workspace/ACE/infra/tokkio/workspace/controller

TOKKIO_ENV_FILE_NAME=my-config.env
TOKKIO_CONFIG_FILE_NAME=ace-app-config.yml

TOKKIO_APP_HOST_IPV4_ADDR=10.209.1.12
TOKKIO_APP_HOST_SSH_USER=kyano
TOKKIO_COTURN_HOST_IPV4_ADDR=10.209.1.12
TOKKIO_COTURN_HOST_SSH_USER=kyano

TOKKIO_NVIDIA_API_KEY=<YOUR_NVIDIA_API_KEY>
TOKKIO_NGC_CLI_API_KEY=<YOUR_NGC_API_KEY>
TOKKIO_OPENAI_API_KEY=<OPTIONAL_OR_EMPTY>
TOKKIO_ELEVENLABS_API_KEY=<YOUR_ELEVENLABS_API_KEY>
```

### 6.2 Generate workspace artifacts

```bash
python3 infra/tokkio/prepare_tokkio_workspace.py --env-file infra/tokkio/.env
```

Expected outputs:

- `/home/kyano/workspace/ACE/infra/tokkio/workspace/controller/generated/my-config.env`
- `/home/kyano/workspace/ACE/infra/tokkio/workspace/controller/ace-app-config.yml`

### 6.3 Clone the official repo

```bash
git clone https://github.com/NVIDIA/ACE.git /home/kyano/workspace/ACE/infra/tokkio/workspace/NVIDIA-ACE
```

### 6.4 Initialize controller config from the official workflow

```bash
./infra/tokkio/deploy_tokkio.sh init-config --env-file infra/tokkio/.env
```

Review and adjust:

- `/home/kyano/workspace/ACE/infra/tokkio/workspace/controller/ace-app-config.yml`

Minimum items to confirm:

- single-node host addresses
- deployment naming
- SSH key paths
- selected Tokkio profile

### 6.5 Preflight NGC and A2F access

Run the helper before install:

```bash
python3 infra/tokkio/check_tokkio_ngc_access.py \
  --env-file /home/kyano/workspace/ACE/infra/tokkio/workspace/controller/generated/my-config.env
```

Required result:

- Helm repo access must succeed
- `repository:nim/nvidia/audio2face-3d:pull` must also succeed

If the A2F check returns `412 Precondition Failed` with a browser-license message, stop and accept the `Audio2Face-3D` NIM license in the browser first.

### 6.6 Install Tokkio

```bash
./infra/tokkio/deploy_tokkio.sh install --env-file infra/tokkio/.env
```

During this project, the effective one-click workflow root was:

- `/home/kyano/workspace/ACE/infra/tokkio/workspace/NVIDIA-ACE/workflows/tokkio/5.0.0-ga/scripts/one-click/baremetal`

### 6.7 Watch Kubernetes state

```bash
kubectl get pods -n app
```

If a pod is unhealthy:

```bash
kubectl describe pod -n app <pod>
```

### 6.8 Check endpoints

Use the helper:

```bash
python3 infra/tokkio/check_tokkio_endpoints.py \
  --insecure \
  --kubectl \
  --ui-url https://10.209.1.12:30111 \
  --api-url http://10.209.1.12:30888 \
  --grafana-url http://10.209.1.12:32300
```

Important:

- install output reported API as `https://10.209.1.12:30888`
- in practice, the working verification path needed to treat it as `http://10.209.1.12:30888`

### 6.9 Validate controller logs

```bash
kubectl logs -n app ace-controller-ace-controller-deployment-0
```

This is the primary log source for TTS failures and controller-side pipeline issues.

### 6.10 If ElevenLabs key changes after deploy

First regenerate and reinstall:

```bash
python3 infra/tokkio/prepare_tokkio_workspace.py --env-file infra/tokkio/.env
./infra/tokkio/deploy_tokkio.sh install --env-file infra/tokkio/.env
```

Then force the controller to reread the secret:

```bash
kubectl delete pod -n app ace-controller-ace-controller-deployment-0
```

This was required in practice because `ace-controller` reads secret-backed env vars at process start.

## 7. Post-Deploy Verification

Minimum verification sequence:

1. Confirm pods are healthy:

```bash
kubectl get pods -n app
```

2. Confirm browser endpoints:

- UI: `https://10.209.1.12:30111`
- Grafana: `http://10.209.1.12:32300`
- API: verify as `http://10.209.1.12:30888`

3. Confirm A2F came up:

- `a2f-a2f-deployment-*` should reach `Running`
- the initial pull of `nvcr.io/nim/nvidia/audio2face-3d:1.3.16` was about `22.4 GB`
- first successful pull took about `5 minutes`

4. Confirm UI behavior:

- Tokkio UI loads
- MetaHuman renders
- browser microphone input is recognized

5. Confirm speech pipeline behavior:

- `riva-speech` logs show ASR calls
- `ace-controller` logs do not show `Invalid API key`

Useful commands:

```bash
kubectl logs -n app ace-controller-ace-controller-deployment-0 --tail=200
kubectl logs -n app riva-speech-6d9d5b7d4f-njtvx --tail=200
```

Final browser-side validation still required:

- speak to the avatar
- confirm spoken response is returned, not only ASR transcription

## 8. Known Pitfalls

- Helm / `helm-diff` plugin compatibility problem occurred on this host. Keep host Helm on a recent release. A newer Helm resolved the plugin issue here.
- NGC chart repo access passing is not enough. `Audio2Face-3D` image pull entitlement must be checked separately.
- API endpoint can be shown as `https` by install output while the working endpoint behaves as `http`.
- Updating the ElevenLabs secret is not enough by itself. `ace-controller` keeps old environment variables until the Pod is restarted.
- Tokkio 5.0 reference configuration does not automatically switch to local `Riva TTS`. The default TTS implementation is still ElevenLabs-dependent in `bot.py`.
- `Audio2Face-3D` first pull is large and slow. Do not treat several minutes of `Pulling` as immediate failure.
- `ace-controller` LLM success does not imply TTS success. `NvidiaLLMService` can be healthy while TTS still fails on ElevenLabs.
- `check_tokkio_endpoints.py` is the preferred local helper because it detects the API protocol mismatch pattern observed here.

## 9. Recovery Procedures

### 9.1 A2F license not accepted

Symptoms:

- `a2f-a2f-deployment-*` stuck in `Init:ImagePullBackOff`
- `kubectl describe pod -n app <a2f-pod>` shows pull failure for:
  - `nvcr.io/nim/nvidia/audio2face-3d:1.3.16`
- error contains `412 Precondition Failed`

How to confirm:

```bash
kubectl describe pod -n app <pod>

python3 infra/tokkio/check_tokkio_ngc_access.py \
  --env-file /home/kyano/workspace/ACE/infra/tokkio/workspace/controller/generated/my-config.env
```

Expected failing sign:

- A2F image scope check fails while ordinary Helm repo access succeeds

Remediation:

1. Open the NVIDIA/NGC browser page for `Audio2Face-3D` NIM.
2. Accept the license there.
3. Re-run the NGC helper until the A2F image scope returns success.
4. Recreate the A2F pod:

```bash
kubectl delete pod -n app <a2f-pod>
```

### 9.2 ElevenLabs invalid API key

Symptoms:

- browser hears user input
- ASR works
- no spoken reply returns
- `ace-controller` logs show:
  - `ElevenLabsTTSServiceWithEndOfSpeech`
  - `Invalid API key`

How to confirm:

```bash
kubectl logs -n app ace-controller-ace-controller-deployment-0 --tail=200
```

Remediation:

1. Update the correct source of truth:

- `/home/kyano/workspace/ACE/infra/tokkio/.env`
- set `TOKKIO_ELEVENLABS_API_KEY=<YOUR_VALID_KEY>`

2. Regenerate controller env and reinstall:

```bash
python3 infra/tokkio/prepare_tokkio_workspace.py --env-file infra/tokkio/.env
./infra/tokkio/deploy_tokkio.sh install --env-file infra/tokkio/.env
```

3. Recreate `ace-controller` so it reloads secret-backed env vars:

```bash
kubectl delete pod -n app ace-controller-ace-controller-deployment-0
```

4. Confirm the new Pod has `ELEVENLABS_API_KEY` in PID 1 env if needed.

### 9.3 Force Pod restart to reload secret-backed env

Use this when a Kubernetes secret is updated but the app still behaves as if the old value is active.

```bash
kubectl delete pod -n app ace-controller-ace-controller-deployment-0
kubectl get pods -n app
```

Rationale:

- `ace-controller` sources secret files during process startup
- a running Pod does not automatically rebuild its process environment after secret rotation

### 9.4 API protocol mismatch

Symptoms:

- install output says `https://10.209.1.12:30888`
- endpoint checks fail with TLS/protocol errors
- plain `http://10.209.1.12:30888` responds instead

How to confirm:

```bash
python3 infra/tokkio/check_tokkio_endpoints.py \
  --insecure \
  --kubectl \
  --ui-url https://10.209.1.12:30111 \
  --api-url https://10.209.1.12:30888 \
  --grafana-url http://10.209.1.12:32300
```

Look for:

- protocol mismatch indication
- successful fallback to `http://...:30888`

Remediation:

- treat the API as `http://10.209.1.12:30888` for practical verification on this host

## 10. Current Status / Open Items

Verified state reached during this effort:

- Tokkio 5.0 one-click deployment completed
- UI reachable at `https://10.209.1.12:30111`
- MetaHuman visible
- A2F recovered after NGC browser license acceptance
- ASR active through `riva-speech`
- ElevenLabs secret updated in the cluster
- `ace-controller` Pod recreated after secret update
- new `ace-controller` PID 1 environment included `ELEVENLABS_API_KEY`
- the stale `Invalid API key` state from the old Pod was cleared

Open item:

- final browser re-test was still required to confirm that response audio was now returned successfully end-to-end

Inference:

- if no new `Invalid API key` messages appear after the refreshed `ace-controller` Pod starts, the next likely validation point is browser-side replay of a new utterance rather than another infrastructure reinstall

## Quick Rebuild Checklist

- Populate `infra/tokkio/.env` with host addresses and secrets.
- Run `python3 infra/tokkio/prepare_tokkio_workspace.py --env-file infra/tokkio/.env`.
- Clone `NVIDIA/ACE` into `/home/kyano/workspace/ACE/infra/tokkio/workspace/NVIDIA-ACE`.
- Run `./infra/tokkio/deploy_tokkio.sh init-config --env-file infra/tokkio/.env`.
- Review `/home/kyano/workspace/ACE/infra/tokkio/workspace/controller/ace-app-config.yml`.
- Run `python3 infra/tokkio/check_tokkio_ngc_access.py --env-file /home/kyano/workspace/ACE/infra/tokkio/workspace/controller/generated/my-config.env`.
- Accept the `Audio2Face-3D` NIM license in browser if the helper reports A2F entitlement failure.
- Run `./infra/tokkio/deploy_tokkio.sh install --env-file infra/tokkio/.env`.
- Run `python3 infra/tokkio/check_tokkio_endpoints.py --insecure --kubectl --ui-url https://10.209.1.12:30111 --api-url http://10.209.1.12:30888 --grafana-url http://10.209.1.12:32300`.
- Run `kubectl get pods -n app` and inspect any unhealthy pod with `kubectl describe pod -n app <pod>`.
- If TTS fails, update `TOKKIO_ELEVENLABS_API_KEY`, rerun `prepare_tokkio_workspace.py` and `deploy_tokkio.sh install`, then restart `ace-controller`.

## First 5 Debug Commands

```bash
kubectl get pods -n app
kubectl describe pod -n app <pod>
kubectl logs -n app ace-controller-ace-controller-deployment-0 --tail=200
python3 infra/tokkio/check_tokkio_ngc_access.py --env-file /home/kyano/workspace/ACE/infra/tokkio/workspace/controller/generated/my-config.env
python3 infra/tokkio/check_tokkio_endpoints.py --insecure --kubectl --ui-url https://10.209.1.12:30111 --api-url http://10.209.1.12:30888 --grafana-url http://10.209.1.12:32300
```
