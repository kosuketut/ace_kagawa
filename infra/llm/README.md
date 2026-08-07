# Osaka-Swallow TensorRT-LLM

This directory contains the optional local LLM path for Tokkio realtime conversation. The current Tokkio default uses hosted NVIDIA NIM with `nvidia/nemotron-3-ultra-550b-a55b`; use this local TensorRT-LLM path only when you intentionally want to serve Osaka-Swallow on the workstation.

The model path is:

1. Download `tokyotech-llm/Qwen3-Swallow-32B-SFT-v0.2`.
2. Download private adapter `Koko0606/Osaka-Swallow-32B-LoRA-v6`.
3. Merge the LoRA into the base model.
4. Serve the merged model with `trtllm-serve` through an OpenAI-compatible API.

Large artifacts must live under `/data/ACE`, not in the repository.

## Prepare Storage

```bash
mkdir -p /data/ACE/{data,hf-cache,models,logs}
ln -s /data/ACE/data ./data
```

## Merge

Run inside an environment with `torch`, `transformers`, `accelerate`, `peft`, and `safetensors`.

```bash
export HF_TOKEN=<your-huggingface-token>
python3 infra/llm/merge_osaka_swallow_lora.py
```

The merged model is written to:

```text
/data/ACE/models/osaka-swallow-32b-lora-v6-merged
```

## Serve

Use a TensorRT-LLM container compatible with the installed driver. The Qwen3 TensorRT-LLM guide currently lists CUDA driver 575+ as the baseline for that path, so confirm the selected image against this host before long runs.

The LLM is intentionally managed separately from `infra/tokkio/manage_tokkio.sh`. Start it before Tokkio when the controller should call the local OpenAI-compatible endpoint, and stop it separately when GPU memory should be released.

Put runtime settings in `/data/ACE/secrets/llm.env` so the same values are used for start, stop, logs, and checks:

```bash
mkdir -p /data/ACE/secrets
$EDITOR /data/ACE/secrets/llm.env
```

Example values:

```dotenv
HF_TOKEN=<your-huggingface-token>
TRTLLM_IMAGE=nvcr.io/nvidia/tensorrt-llm/release:latest
TRTLLM_CONTAINER_NAME=ace-trtllm-osaka-swallow-manual
NVIDIA_VISIBLE_DEVICES=0,1
TRTLLM_PORT=8010
TRTLLM_TP_SIZE=2
TRTLLM_SERVED_MODEL_NAME=osaka-swallow-32b-lora-v6-merged
```

Run the start and health-check commands from the repo root:

```bash
docker compose --env-file /data/ACE/secrets/llm.env \
  -f infra/llm/docker-compose.trtllm.yml \
  up -d trtllm-osaka-swallow

python3 infra/llm/check_llm_endpoint.py \
  --base-url http://127.0.0.1:8010/v1 \
  --model osaka-swallow-32b-lora-v6-merged
```

Watch startup:

```bash
docker logs -f ace-trtllm-osaka-swallow-manual
```

The 32B merged model can take several minutes to expose port `8010` because TensorRT-LLM prefetches about 61GB of checkpoint files before the OpenAI-compatible API starts listening.

Troubleshooting checks used during startup:

```bash
systemctl is-active containerd docker
docker ps -a --filter 'name=ace-trtllm' \
  --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
ss -ltnp | grep ':8010'
```

If `docker compose up` fails with `dial unix:///run/containerd/containerd.sock: timeout`, Docker is running but `containerd` is not responding. Start `containerd` first, then retry the compose command:

```bash
sudo systemctl start containerd
systemctl is-active containerd docker
docker ps
```

If startup then fails with `endpoint with name ace-trtllm-osaka-swallow already exists in network host`, remove the half-created container and disconnect the stale host-network endpoint:

```bash
docker rm -f ace-trtllm-osaka-swallow
docker network disconnect -f host ace-trtllm-osaka-swallow
```

If the stale endpoint still remains, use a temporary distinct container name while leaving `TRTLLM_PORT=8010` unchanged:

```bash
TRTLLM_CONTAINER_NAME=ace-trtllm-osaka-swallow-test \
docker compose --env-file /data/ACE/secrets/llm.env \
  -f infra/llm/docker-compose.trtllm.yml \
  up -d trtllm-osaka-swallow

docker logs -f ace-trtllm-osaka-swallow-test
```

Stop with the same env file:

```bash
docker compose --env-file /data/ACE/secrets/llm.env \
  -f infra/llm/docker-compose.trtllm.yml \
  stop trtllm-osaka-swallow
```

Tokkio now defaults to the hosted Stockmark NIM. If you intentionally switch Tokkio back to this local endpoint, override the LLM settings in `infra/tokkio/.env`:

```dotenv
TOKKIO_LLM_BASE_URL=http://127.0.0.1:8010/v1
TOKKIO_LLM_MODEL=osaka-swallow-32b-lora-v6-merged
TOKKIO_LLM_API_KEY=tensorrt_llm
```

If Docker reports `endpoint with name ace-trtllm-osaka-swallow already exists in network host`, keep using a distinct `TRTLLM_CONTAINER_NAME` in `/data/ACE/secrets/llm.env` or clear the stale Docker host-network endpoint outside the Tokkio lifecycle.
