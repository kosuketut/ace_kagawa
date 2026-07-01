#!/usr/bin/env python3
"""Merge the Osaka-Swallow LoRA adapter into its Qwen3-Swallow 32B base model."""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path


BASE_MODEL_ID = "tokyotech-llm/Qwen3-Swallow-32B-SFT-v0.2"
LORA_MODEL_ID = "Koko0606/Osaka-Swallow-32B-LoRA-v6"
OUTPUT_MODEL_NAME = "osaka-swallow-32b-lora-v6-merged"
DEFAULT_PROJECT_ROOT = Path("/data/ACE")
THINKING_GENERATION_PROMPT_BLOCK = """{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\\n\\n</think>\\n\\n' }}
    {%- endif %}
{%- endif %}"""

NO_THINKING_GENERATION_PROMPT_BLOCK = """{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' }}
{%- endif %}"""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=BASE_MODEL_ID)
    parser.add_argument("--lora-model", default=LORA_MODEL_ID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PROJECT_ROOT / "models" / OUTPUT_MODEL_NAME)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_PROJECT_ROOT / "hf-cache")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--smoke-prompt", default="おすすめの昼食を一文で教えてください。")
    return parser


def import_runtime_deps():
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependencies. Install torch, transformers, accelerate, peft, and safetensors "
            "inside the model-prep environment before running this script."
        ) from exc
    return torch, PeftModel, AutoModelForCausalLM, AutoTokenizer


def resolve_dtype(torch_module, dtype: str):
    return {
        "bfloat16": torch_module.bfloat16,
        "float16": torch_module.float16,
        "float32": torch_module.float32,
    }[dtype]


def write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def disable_thinking_by_default(template: str) -> str:
    if NO_THINKING_GENERATION_PROMPT_BLOCK in template:
        return template
    if THINKING_GENERATION_PROMPT_BLOCK not in template:
        pattern = re.compile(
            r"\{%- if add_generation_prompt %\}\s*"
            r"\{\{- '<\|im_start\|>assistant\\n' \}\}\s*"
            r"\{%- if enable_thinking is defined and enable_thinking is false %\}\s*"
            r"\{\{- '<think>\\n\\n</think>\\n\\n' \}\}\s*"
            r"\{%- endif %\}\s*"
            r"\{%- endif %\}",
            flags=re.DOTALL,
        )
        updated, count = pattern.subn(NO_THINKING_GENERATION_PROMPT_BLOCK, template)
        if count == 0:
            raise ValueError("expected Qwen thinking generation prompt block was not found")
        return updated
    return template.replace(THINKING_GENERATION_PROMPT_BLOCK, NO_THINKING_GENERATION_PROMPT_BLOCK)


def patch_saved_chat_template(output_dir: Path) -> None:
    template_path = output_dir / "chat_template.jinja"
    if not template_path.exists():
        return
    template_path.write_text(disable_thinking_by_default(template_path.read_text(encoding="utf-8")), encoding="utf-8")


def main() -> int:
    args = build_arg_parser().parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required because the Osaka-Swallow LoRA repository is private.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    torch, PeftModel, AutoModelForCausalLM, AutoTokenizer = import_runtime_deps()
    torch_dtype = resolve_dtype(torch, args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        token=token,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        token=token,
        cache_dir=args.cache_dir,
        torch_dtype=torch_dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    )
    lora_model = PeftModel.from_pretrained(
        base_model,
        args.lora_model,
        token=token,
        cache_dir=args.cache_dir,
    )
    merged_model = lora_model.merge_and_unload()
    merged_model.save_pretrained(
        args.output_dir,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    tokenizer.save_pretrained(args.output_dir)
    patch_saved_chat_template(args.output_dir)

    smoke_text = None
    if not args.skip_smoke:
        messages = [
            {
                "role": "system",
                "content": (
                    "あなたは日本語で応答する対話型バーチャルアシスタントです。"
                    "あなたの名前は香川です。"
                    "標準語で自然かつ簡潔に、40から120文字、1から3文で答えてください。"
                    "マークダウン、絵文字、内部思考は出さないでください。"
                ),
            },
            {"role": "user", "content": args.smoke_prompt},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(merged_model.device)
        with torch.inference_mode():
            output = merged_model.generate(
                **inputs,
                max_new_tokens=96,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
            )
        generated_ids = output[0][inputs["input_ids"].shape[-1] :]
        smoke_text = strip_thinking(tokenizer.decode(generated_ids, skip_special_tokens=True))
        print(smoke_text)

    write_manifest(
        args.output_dir / "ace_merge_manifest.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "base_model": args.base_model,
            "lora_model": args.lora_model,
            "output_model_name": OUTPUT_MODEL_NAME,
            "dtype": args.dtype,
            "trust_remote_code": args.trust_remote_code,
            "smoke_text": smoke_text,
        },
    )
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
