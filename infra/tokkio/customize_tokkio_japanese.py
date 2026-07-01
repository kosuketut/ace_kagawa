#!/usr/bin/env python3
"""Apply a Japanese-oriented Tokkio 5.0 source customization to a local ACE clone.

This helper patches the Tokkio 5.0 `llm-rag` example sources inside a checked-out
`NVIDIA-ACE` repository. It intentionally changes only source/config files inside
the ACE clone and does not build or deploy anything by itself.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


DEFAULT_ACE_REPO_DIR = Path(__file__).resolve().parent / "workspace" / "NVIDIA-ACE"
DEFAULT_LLM_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_LLM_MODEL = "stockmark/stockmark-2-100b-instruct"
DEFAULT_IRODORI_TTS_BASE_URL = "http://10.209.1.12:8021"
DEFAULT_RAG_SERVER_URL = "http://10.209.1.12:8081/v1"
DEFAULT_RAG_COLLECTION_NAME = "ace_kagawa"
DEFAULT_RAG_SUFFIX_PROMPT = "日本語で80文字以内、1から2文で簡潔に答えてください。箇条書きや詳細説明は求められた時だけにしてください。"
DEFAULT_RAG_MAX_TOKENS = 128
DEFAULT_RAG_VDB_TOP_K = 12
DEFAULT_RAG_RERANKER_TOP_K = 5
DEFAULT_RAG_MULTIMODAL_RERANKER_TOP_K = 10
DEFAULT_ASR_MODEL = "conformer-unified-ja-JP-asr-streaming-asr-bls-ensemble"
DEFAULT_ASR_RMIR = "nvidia/riva/rmir_asr_conformer_unified_ja_jp_str:2.19.0"
OBSOLETE_NEMOTRON_ASR_MODEL = "nvidia/nemotron-3.5-asr-streaming-0.6b"
PREVIOUS_JAPANESE_ASR_RMIR = DEFAULT_ASR_RMIR
JAPANESE_ASR_MODEL = DEFAULT_ASR_MODEL
ENGLISH_ASR_RMIR = "nvidia/riva/rmir_asr_parakeet_1-1b_en_us_str_silero:2.19.0.1"

CONFIG_PY = """# Copyright(c) 2025 NVIDIA Corporation. All rights reserved.

# NVIDIA Corporation and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA Corporation is strictly prohibited.

from typing import Literal

from pydantic import BaseModel, StrictStr
import yaml
from nvidia_pipecat.services.animation_graph_service import AnimationGraphConfiguration


class Pipeline(BaseModel):
    llm_processor: Literal["NvidiaRAGService", "NvidiaLLMService", "OpenAILLMService"]
    tts_processor: Literal["ElevenLabsTTSService", "RivaTTSService", "IrodoriTTSService"] = "IrodoriTTSService"
    filler: list[str] = [
        "確認しています",
    ]
    time_delay: float = 6.0


class UserPresenceProcessor(BaseModel):
    welcome_message: StrictStr = "こんにちは。ご用件をどうぞ。"
    farewell_message: StrictStr = "ありがとうございました。"


class ProactivityProcessor(BaseModel):
    timer_duration: int = 100
    default_message: StrictStr = "必要でしたら、いつでもお声がけください。"


class OpenAILLMContext(BaseModel):
    name: str
    prompt: str


class NvidiaRAGService(BaseModel):
    use_knowledge_base: bool = True
    max_tokens: int = 128
    vdb_top_k: int = 12
    reranker_top_k: int = 5
    multimodal_reranker_top_k: int = 10
    enable_reranker: bool = True
    rag_server_url: str
    collection_name: StrictStr = "collection_name"
    suffix_prompt: str = ""


class NvidiaLLMService(BaseModel):
    base_url: str = "https://integrate.api.nvidia.com/v1"
    model: str = "stockmark/stockmark-2-100b-instruct"


class OpenAILLMService(BaseModel):
    model: str


class RivaASRServiceConfig(BaseModel):
    server: str = "localhost:50052"
    language: str = "ja-JP"
    sample_rate: int = 16000
    model: str = "conformer-unified-ja-JP-asr-streaming-asr-bls-ensemble"


class ElevenLabsTTSServiceConfig(BaseModel):
    voice_id: str = ""
    sample_rate: int = 16000
    model: str = "eleven_multilingual_v2"
    stability: float = 0.3
    speed: float = 1.0
    similarity_boost: float = 0.85


class RivaTTSServiceConfig(BaseModel):
    server: str = "grpc.nvcf.nvidia.com:443"
    function_id: str = "877104f7-e885-42b9-8de8-f6e4c6303969"
    voice_name: str = "Magpie-Multilingual.JA-JP.Aria.Neutral"
    language: str = "ja-JP"
    sample_rate: int = 24000


class IrodoriTTSServiceConfig(BaseModel):
    base_url: str = "http://10.209.1.12:8021"
    voice: str = "kagawa"
    sample_rate: int = 16000
    timeout_s: float = 180.0
    response_format: str = "pcm"


class FacialGestureProviderProcessor(BaseModel):
    user_stopped_speaking_gesture: str
    start_interruption_gesture: str
    probability: float = 0.5


class RAGMultimodalResponseProcessor(BaseModel):
    confidence_threshold: float = 0.5
    top_n: int = 2


class Config(BaseModel):
    Pipeline: Pipeline
    UserPresenceProcesssor: UserPresenceProcessor
    ProactivityProcessor: ProactivityProcessor
    OpenAILLMContext: OpenAILLMContext
    NvidiaRAGService: NvidiaRAGService
    NvidiaLLMService: NvidiaLLMService
    OpenAILLMService: OpenAILLMService
    RivaASRService: RivaASRServiceConfig
    ElevenLabsTTSService: ElevenLabsTTSServiceConfig
    RivaTTSService: RivaTTSServiceConfig
    IrodoriTTSService: IrodoriTTSServiceConfig
    FacialGestureProviderProcessor: FacialGestureProviderProcessor
    AnimationGraphService: AnimationGraphConfiguration
    RAGMultimodalResponseProcessor: RAGMultimodalResponseProcessor
"""


CONFIG_YAML = """Pipeline:
    # Only one of the following LLM service configurations will be active based on this setting:
    # - "NvidiaLLMService" - Uses the NvidiaLLMService configuration
    # - "NvidiaRAGService" - Uses the NvidiaRAGService configuration
    # - "OpenAILLMService" - Uses the OpenAILLMService configuration
    llm_processor: "NvidiaLLMService"
    # Use the host-side Irodori-TTS HTTP service by default. Riva and
    # ElevenLabs remain available for fallback by changing this value.
    tts_processor: "IrodoriTTSService"
    filler:
        - "確認しています"
    time_delay: 6.0

UserPresenceProcesssor:
    welcome_message: "こんにちは。ご用件をどうぞ。"
    farewell_message: "ありがとうございました。"

ProactivityProcessor:
    timer_duration: 100
    default_message: "必要でしたら、いつでもお声がけください。"

OpenAILLMContext:
    name: "香川"
    prompt: "あなたは「{name}」という名前の、日本語で応答する対話型バーチャルアシスタントです。
            標準語で自然かつ簡潔に答えてください。
            80文字以内、1から2文で答えてください。詳しい説明を求められた場合だけ、最大3文までにしてください。
            箇条書き、番号付きリスト、マークダウン、絵文字、記号装飾、内部思考は出さないでください。
            質問に必要な情報が不足している場合は、推測で断定せず、確認してください。
            ユーザー発話の前に「参考情報」ブロックがある場合は、その内容を事実として扱い、自然な標準語で短く織り込んでください。"

# This configuration is only used when llm_processor is set to "NvidiaRAGService"
NvidiaRAGService:
    use_knowledge_base: true
    max_tokens: 128
    vdb_top_k: 12
    reranker_top_k: 5
    multimodal_reranker_top_k: 10
    enable_reranker: true
    rag_server_url: "http://0.0.0.0:8081"
    collection_name: "collection_name"
    suffix_prompt: "日本語で80文字以内、1から2文で簡潔に答えてください。箇条書きや詳細説明は求められた時だけにしてください。"

# This configuration is only used when llm_processor is set to "NvidiaLLMService"
NvidiaLLMService:
    base_url: "https://integrate.api.nvidia.com/v1"
    model: "stockmark/stockmark-2-100b-instruct"

# This configuration is only used when llm_processor is set to "OpenAILLMService"
OpenAILLMService:
    model: "gpt-4o"

RivaASRService:
    # This model name must match the Riva API model registered by the local
    # riva-speech/triton deployment.
    server: "localhost:50052"
    language: "ja-JP"
    sample_rate: 16000
    model: "conformer-unified-ja-JP-asr-streaming-asr-bls-ensemble"

ElevenLabsTTSService:
    # Set a Japanese-capable ElevenLabs voice id if you prefer ElevenLabs TTS.
    voice_id: ""
    sample_rate: 16000
    model: "eleven_multilingual_v2"
    stability: 0.3
    speed: 1.0
    similarity_boost: 0.85

RivaTTSService:
    # Local Riva 2.19 in this Tokkio bundle exposes Magpie TTS, but not the
    # Japanese voices. Use the hosted Magpie TTS NIM for Japanese synthesis.
    server: "grpc.nvcf.nvidia.com:443"
    function_id: "877104f7-e885-42b9-8de8-f6e4c6303969"
    language: "ja-JP"
    voice_name: "Magpie-Multilingual.JA-JP.Aria.Neutral"
    sample_rate: 24000

IrodoriTTSService:
    base_url: "http://10.209.1.12:8021"
    voice: "kagawa"
    sample_rate: 16000
    timeout_s: 180.0
    response_format: "pcm"

RAGMultimodalResponseProcessor:
    confidence_threshold: 0.37
    top_n: 2

FacialGestureProviderProcessor:
    user_stopped_speaking_gesture: "Taunt"
    start_interruption_gesture: "Pensive"
    probability: 0.5

# ADVANCED CONFIGURATION SECTION BELOW
# AnimationGraph service configuration is only needed if your 3D avatar scene has support for gestures and postures.
# Changing these values will not have an effect unless your scene supports them.
AnimationGraphService:
    animation_types:
        posture:
            duration_relevant_animation_name: "posture"
            animations:
                posture:
                    default_clip_id: "Attentive"
                    clips:
                        - clip_id: Talking
                          description: "Small gestures with hand and upper body: Avatar is talking"
                          duration: -1
                          meaning: Emphasizing that Avatar is talking
                        - clip_id: Listening
                          description: "Small gestures with hand and upper body: Avatar is listening"
                          duration: -1
                          meaning: Emphasizing that one is listening
                        - clip_id: Idle
                          description: "Small gestures with hand and upper body: Avatar is idle"
                          duration: -1
                          meaning: Show the user that the avatar is waiting for something to happen
                        - clip_id: Thinking
                          description: "Gestures with hand and upper body: Avatar is thinking"
                          duration: -1
                          meaning: Show the user that the avatar thinking about his next answer or is trying to remember something
                        - clip_id: Attentive
                          description: "Small gestures with hand and upper body: Avatar is attentive"
                          duration: -1
                          meaning: Show the user that the avatar is paying attention to the user
"""


OLD_LLM_SNIPPET = """        if config.Pipeline.llm_processor == 'NvidiaLLMService':
            llm = TokkioNvidiaLLMService(
                api_key=os.getenv("NVIDIA_API_KEY"),
                base_url=config.NvidiaLLMService.base_url,
                model=config.NvidiaLLMService.model,
                filler=config.Pipeline.filler,
                time_delay=config.Pipeline.time_delay,
            )
"""


NEW_LLM_SNIPPET = """        if config.Pipeline.llm_processor == 'NvidiaLLMService':
            llm_api_key = (
                os.getenv("NVIDIA_LLM_API_KEY")
                or os.getenv("LLM_API_KEY")
                or os.getenv("NVIDIA_API_KEY")
                or ""
            )
            llm = TokkioNvidiaLLMService(
                api_key=llm_api_key,
                base_url=config.NvidiaLLMService.base_url,
                model=config.NvidiaLLMService.model,
                filler=config.Pipeline.filler,
                time_delay=config.Pipeline.time_delay,
            )
"""


PREVIOUS_LLM_SNIPPET = NEW_LLM_SNIPPET.replace('or ""', 'or "tensorrt_llm"')


OLD_RAG_SNIPPET = """        if config.Pipeline.llm_processor == 'NvidiaRAGService':
            llm = TokkioNvidiaRAGService(
                collection_name=config.NvidiaRAGService.collection_name,
                rag_server_url=config.NvidiaRAGService.rag_server_url,
                use_knowledge_base=config.NvidiaRAGService.use_knowledge_base,
                max_tokens=config.NvidiaRAGService.max_tokens,
                suffix_prompt=config.NvidiaRAGService.suffix_prompt,
                filler=config.Pipeline.filler,
                time_delay=config.Pipeline.time_delay,
            )
"""


NEW_RAG_SNIPPET = """        if config.Pipeline.llm_processor == 'NvidiaRAGService':
            llm = TokkioNvidiaRAGService(
                collection_name=config.NvidiaRAGService.collection_name,
                rag_server_url=config.NvidiaRAGService.rag_server_url,
                use_knowledge_base=config.NvidiaRAGService.use_knowledge_base,
                max_tokens=config.NvidiaRAGService.max_tokens,
                vdb_top_k=config.NvidiaRAGService.vdb_top_k,
                reranker_top_k=config.NvidiaRAGService.reranker_top_k,
                multimodal_reranker_top_k=config.NvidiaRAGService.multimodal_reranker_top_k,
                enable_reranker=config.NvidiaRAGService.enable_reranker,
                suffix_prompt=config.NvidiaRAGService.suffix_prompt,
                filler=config.Pipeline.filler,
                time_delay=config.Pipeline.time_delay,
            )
"""


PREVIOUS_RAG_SNIPPET = NEW_RAG_SNIPPET.replace(
    "                multimodal_reranker_top_k=config.NvidiaRAGService.multimodal_reranker_top_k,\n",
    "",
)


OLD_BOT_SNIPPET = """        # For Nim use:
        # stt = RivaASRService(
        #     api_key=os.getenv("NVIDIA_API_KEY"),
        #     language="en-US",
        #     sample_rate=16000,
        #     model="parakeet-1.1b-en-US-asr-streaming-asr-bls-ensemble",
        # )

        riva_server_ip = os.getenv("RIVA_SERVER_URL", "localhost:50052")
        if riva_server_ip != "localhost:50052":
            riva_server_ip.replace("http://", "")
        stt = RivaASRService(
            server=riva_server_ip,
            language="en-US",
            sample_rate=16000,
            model="parakeet-1.1b-en-US-asr-streaming-silero-vad-asr-bls-ensemble",
        )
        tts = ElevenLabsTTSServiceWithEndOfSpeech(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id=os.getenv("ELEVENLABS_VOICE_ID", "cgSgspJ2msm6clMCkdW9"),
            sample_rate=16000,
            model = "eleven_flash_v2_5",
            stability = 0.3,
            speed = 0.97,
            similarity_boost = 0.85
        )
"""


NEW_BOT_SNIPPET = """        riva_server_ip = os.getenv("RIVA_SERVER_URL", config.RivaASRService.server)
        riva_server_ip = riva_server_ip.replace("http://", "").replace("https://", "")

        stt_kwargs = {
            "server": riva_server_ip,
            "language": config.RivaASRService.language,
            "sample_rate": config.RivaASRService.sample_rate,
        }
        if config.RivaASRService.model:
            stt_kwargs["model"] = config.RivaASRService.model
        stt = RivaASRService(**stt_kwargs)

        if config.Pipeline.tts_processor == "IrodoriTTSService":
            from .tokkio_irodori_tts import IrodoriTTSService

            tts = IrodoriTTSService(
                base_url=config.IrodoriTTSService.base_url,
                voice=config.IrodoriTTSService.voice,
                sample_rate=config.IrodoriTTSService.sample_rate,
                timeout_s=config.IrodoriTTSService.timeout_s,
                response_format=config.IrodoriTTSService.response_format,
            )
        elif config.Pipeline.tts_processor == "RivaTTSService":
            nvidia_api_key = os.getenv("NVIDIA_API_KEY")
            if not nvidia_api_key:
                try:
                    with open("/secrets/nvidia_api_key.txt", encoding="utf-8") as api_key_file:
                        nvidia_api_key = api_key_file.read().strip()
                except FileNotFoundError:
                    nvidia_api_key = None

            tts_server_ip = os.getenv("RIVA_TTS_SERVER_URL", config.RivaTTSService.server)
            tts_server_ip = tts_server_ip.replace("http://", "").replace("https://", "")
            tts = RivaTTSService(
                server=tts_server_ip,
                api_key=nvidia_api_key,
                function_id=config.RivaTTSService.function_id,
                voice_id=config.RivaTTSService.voice_name,
                language=config.RivaTTSService.language,
                sample_rate=config.RivaTTSService.sample_rate,
            )
        else:
            voice_id = os.getenv("ELEVENLABS_VOICE_ID") or config.ElevenLabsTTSService.voice_id
            tts = ElevenLabsTTSServiceWithEndOfSpeech(
                api_key=os.getenv("ELEVENLABS_API_KEY"),
                voice_id=voice_id or "cgSgspJ2msm6clMCkdW9",
                sample_rate=config.ElevenLabsTTSService.sample_rate,
                model=config.ElevenLabsTTSService.model,
                stability=config.ElevenLabsTTSService.stability,
                speed=config.ElevenLabsTTSService.speed,
                similarity_boost=config.ElevenLabsTTSService.similarity_boost,
            )
"""


OLD_CONFIG_DRIVEN_TTS_SNIPPET = """        if config.Pipeline.tts_processor == "RivaTTSService":
            tts = RivaTTSService(
                server=riva_server_ip,
                voice_id=config.RivaTTSService.voice_name,
                language=config.RivaTTSService.language,
                sample_rate=config.RivaTTSService.sample_rate,
            )
        else:
"""


PREVIOUS_CONFIG_DRIVEN_TTS_SNIPPET = """        if config.Pipeline.tts_processor == "RivaTTSService":
            nvidia_api_key = os.getenv("NVIDIA_API_KEY")
            if not nvidia_api_key:
                try:
                    with open("/secrets/nvidia_api_key.txt", encoding="utf-8") as api_key_file:
                        nvidia_api_key = api_key_file.read().strip()
                except FileNotFoundError:
                    nvidia_api_key = None

            tts_server_ip = os.getenv("RIVA_TTS_SERVER_URL", config.RivaTTSService.server)
            tts_server_ip = tts_server_ip.replace("http://", "").replace("https://", "")
            tts = RivaTTSService(
                server=tts_server_ip,
                api_key=nvidia_api_key,
                function_id=config.RivaTTSService.function_id,
                voice_id=config.RivaTTSService.voice_name,
                language=config.RivaTTSService.language,
                sample_rate=config.RivaTTSService.sample_rate,
            )
        else:
"""


NEW_CONFIG_DRIVEN_TTS_SNIPPET = """        if config.Pipeline.tts_processor == "IrodoriTTSService":
            from .tokkio_irodori_tts import IrodoriTTSService

            tts = IrodoriTTSService(
                base_url=config.IrodoriTTSService.base_url,
                voice=config.IrodoriTTSService.voice,
                sample_rate=config.IrodoriTTSService.sample_rate,
                timeout_s=config.IrodoriTTSService.timeout_s,
                response_format=config.IrodoriTTSService.response_format,
            )
        elif config.Pipeline.tts_processor == "RivaTTSService":
            nvidia_api_key = os.getenv("NVIDIA_API_KEY")
            if not nvidia_api_key:
                try:
                    with open("/secrets/nvidia_api_key.txt", encoding="utf-8") as api_key_file:
                        nvidia_api_key = api_key_file.read().strip()
                except FileNotFoundError:
                    nvidia_api_key = None

            tts_server_ip = os.getenv("RIVA_TTS_SERVER_URL", config.RivaTTSService.server)
            tts_server_ip = tts_server_ip.replace("http://", "").replace("https://", "")
            tts = RivaTTSService(
                server=tts_server_ip,
                api_key=nvidia_api_key,
                function_id=config.RivaTTSService.function_id,
                voice_id=config.RivaTTSService.voice_name,
                language=config.RivaTTSService.language,
                sample_rate=config.RivaTTSService.sample_rate,
            )
        else:
"""


def quote_yaml(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def normalize_openai_base_url(value: str) -> str:
    url = value.strip().rstrip("/")
    if not url:
        return url
    if url.endswith("/v1"):
        return url
    return f"{url}/v1"


def normalize_http_base_url(value: str) -> str:
    return value.strip().rstrip("/")


def normalize_rag_server_url(value: str) -> str:
    url = normalize_http_base_url(value)
    if not url:
        return url
    if url.endswith("/v1"):
        return url
    return f"{url}/v1"


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def build_config_yaml(
    *,
    llm_base_url: str = DEFAULT_LLM_BASE_URL,
    llm_model: str = DEFAULT_LLM_MODEL,
    irodori_tts_base_url: str = DEFAULT_IRODORI_TTS_BASE_URL,
    rag_enabled: bool = False,
    rag_server_url: str = DEFAULT_RAG_SERVER_URL,
    rag_collection_name: str = DEFAULT_RAG_COLLECTION_NAME,
    rag_use_knowledge_base: bool = True,
    rag_max_tokens: int = DEFAULT_RAG_MAX_TOKENS,
    rag_vdb_top_k: int = DEFAULT_RAG_VDB_TOP_K,
    rag_reranker_top_k: int = DEFAULT_RAG_RERANKER_TOP_K,
    rag_multimodal_reranker_top_k: int = DEFAULT_RAG_MULTIMODAL_RERANKER_TOP_K,
    rag_enable_reranker: bool = True,
    rag_suffix_prompt: str = DEFAULT_RAG_SUFFIX_PROMPT,
) -> str:
    base_url = normalize_openai_base_url(llm_base_url) or DEFAULT_LLM_BASE_URL
    model = llm_model.strip() or DEFAULT_LLM_MODEL
    tts_base_url = normalize_http_base_url(irodori_tts_base_url) or DEFAULT_IRODORI_TTS_BASE_URL
    normalized_rag_server_url = normalize_rag_server_url(rag_server_url) or DEFAULT_RAG_SERVER_URL
    rag_collection = rag_collection_name.strip() or DEFAULT_RAG_COLLECTION_NAME
    rag_suffix = rag_suffix_prompt.strip() or DEFAULT_RAG_SUFFIX_PROMPT
    llm_processor = "NvidiaRAGService" if rag_enabled else "NvidiaLLMService"
    return (
        CONFIG_YAML.replace('llm_processor: "NvidiaLLMService"', f"llm_processor: {quote_yaml(llm_processor)}")
        .replace('use_knowledge_base: true', f"use_knowledge_base: {'true' if rag_use_knowledge_base else 'false'}")
        .replace(f"max_tokens: {DEFAULT_RAG_MAX_TOKENS}", f"max_tokens: {rag_max_tokens}")
        .replace(f"vdb_top_k: {DEFAULT_RAG_VDB_TOP_K}", f"vdb_top_k: {rag_vdb_top_k}")
        .replace(f"reranker_top_k: {DEFAULT_RAG_RERANKER_TOP_K}", f"reranker_top_k: {rag_reranker_top_k}")
        .replace(
            f"multimodal_reranker_top_k: {DEFAULT_RAG_MULTIMODAL_RERANKER_TOP_K}",
            f"multimodal_reranker_top_k: {rag_multimodal_reranker_top_k}",
        )
        .replace("enable_reranker: true", f"enable_reranker: {'true' if rag_enable_reranker else 'false'}")
        .replace('rag_server_url: "http://0.0.0.0:8081"', f"rag_server_url: {quote_yaml(normalized_rag_server_url)}")
        .replace('collection_name: "collection_name"', f"collection_name: {quote_yaml(rag_collection)}")
        .replace(f'suffix_prompt: "{DEFAULT_RAG_SUFFIX_PROMPT}"', f"suffix_prompt: {quote_yaml(rag_suffix)}")
        .replace(f'base_url: "{DEFAULT_LLM_BASE_URL}"', f"base_url: {quote_yaml(base_url)}")
        .replace(f'model: "{DEFAULT_LLM_MODEL}"', f"model: {quote_yaml(model)}")
        .replace('base_url: "http://10.209.1.12:8021"', f"base_url: {quote_yaml(tts_base_url)}")
    )


def write_text(path: Path, content: str) -> None:
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def replace_once(path: Path, old: str, new: str) -> None:
    original = path.read_text(encoding="utf-8")
    if new in original:
        return
    if old not in original:
        raise RuntimeError(f"expected snippet not found in {path}")
    updated = original.replace(old, new, 1)
    write_text(path, updated)


def replace_literal(path: Path, old: str, new: str) -> None:
    original = path.read_text(encoding="utf-8")
    if new in original:
        return
    if old not in original:
        raise RuntimeError(f"expected literal not found in {path}: {old}")
    write_text(path, original.replace(old, new, 1))


def patch_llm_snippet(bot_py_path: Path) -> None:
    original = bot_py_path.read_text(encoding="utf-8")
    if NEW_LLM_SNIPPET in original:
        return
    if OLD_LLM_SNIPPET in original:
        replace_literal(bot_py_path, OLD_LLM_SNIPPET, NEW_LLM_SNIPPET)
        return
    if PREVIOUS_LLM_SNIPPET in original:
        replace_literal(bot_py_path, PREVIOUS_LLM_SNIPPET, NEW_LLM_SNIPPET)
        return
    raise RuntimeError(f"expected LLM snippet not found in {bot_py_path}")


def patch_rag_snippet(bot_py_path: Path) -> None:
    original = bot_py_path.read_text(encoding="utf-8")
    if NEW_RAG_SNIPPET in original:
        return
    if OLD_RAG_SNIPPET in original:
        replace_literal(bot_py_path, OLD_RAG_SNIPPET, NEW_RAG_SNIPPET)
        return
    if PREVIOUS_RAG_SNIPPET in original:
        replace_literal(bot_py_path, PREVIOUS_RAG_SNIPPET, NEW_RAG_SNIPPET)
        return
    if "TokkioNvidiaRAGService(" not in original:
        return
    raise RuntimeError(f"expected RAG snippet not found in {bot_py_path}")


def patch_riva_values(ace_repo_dir: Path) -> list[Path]:
    values_paths = sorted(
        (ace_repo_dir / "workflows" / "tokkio" / "5.0.0-ga" / "llm-rag").glob("tokkio-*/values.yaml")
    )
    if not values_paths:
        raise FileNotFoundError("Tokkio llm-rag values.yaml files were not found")

    changed_paths: list[Path] = []
    for path in values_paths:
        original = path.read_text(encoding="utf-8")
        if DEFAULT_ASR_RMIR in original:
            changed_paths.append(path)
            continue
        replacement_source = next(
            (
                model
                for model in (ENGLISH_ASR_RMIR, OBSOLETE_NEMOTRON_ASR_MODEL, DEFAULT_ASR_MODEL)
                if model in original
            ),
            None,
        )
        if replacement_source is None:
            raise RuntimeError(f"expected Riva ASR RMIR not found in {path}")
        write_text(path, original.replace(replacement_source, DEFAULT_ASR_RMIR))
        changed_paths.append(path)
    return changed_paths


def patch_profile_ace_controller_configs(
    ace_repo_dir: Path,
    *,
    llm_base_url: str,
    llm_model: str,
    irodori_tts_base_url: str,
    rag_enabled: bool,
    rag_server_url: str,
    rag_collection_name: str,
    rag_use_knowledge_base: bool,
    rag_max_tokens: int,
    rag_vdb_top_k: int,
    rag_reranker_top_k: int,
    rag_multimodal_reranker_top_k: int,
    rag_enable_reranker: bool,
    rag_suffix_prompt: str,
) -> list[Path]:
    config_paths = sorted(
        (ace_repo_dir / "workflows" / "tokkio" / "5.0.0-ga" / "llm-rag").glob(
            "tokkio-*/config/ace-controller/config.yaml"
        )
    )
    for path in config_paths:
        write_text(
            path,
            build_config_yaml(
                llm_base_url=llm_base_url,
                llm_model=llm_model,
                irodori_tts_base_url=irodori_tts_base_url,
                rag_enabled=rag_enabled,
                rag_server_url=rag_server_url,
                rag_collection_name=rag_collection_name,
                rag_use_knowledge_base=rag_use_knowledge_base,
                rag_max_tokens=rag_max_tokens,
                rag_vdb_top_k=rag_vdb_top_k,
                rag_reranker_top_k=rag_reranker_top_k,
                rag_multimodal_reranker_top_k=rag_multimodal_reranker_top_k,
                rag_enable_reranker=rag_enable_reranker,
                rag_suffix_prompt=rag_suffix_prompt,
            ),
        )
    return config_paths


def apply_patch(
    ace_repo_dir: Path,
    *,
    llm_base_url: str = DEFAULT_LLM_BASE_URL,
    llm_model: str = DEFAULT_LLM_MODEL,
    irodori_tts_base_url: str = DEFAULT_IRODORI_TTS_BASE_URL,
    rag_enabled: bool = False,
    rag_server_url: str = DEFAULT_RAG_SERVER_URL,
    rag_collection_name: str = DEFAULT_RAG_COLLECTION_NAME,
    rag_use_knowledge_base: bool = True,
    rag_max_tokens: int = DEFAULT_RAG_MAX_TOKENS,
    rag_vdb_top_k: int = DEFAULT_RAG_VDB_TOP_K,
    rag_reranker_top_k: int = DEFAULT_RAG_RERANKER_TOP_K,
    rag_multimodal_reranker_top_k: int = DEFAULT_RAG_MULTIMODAL_RERANKER_TOP_K,
    rag_enable_reranker: bool = True,
    rag_suffix_prompt: str = DEFAULT_RAG_SUFFIX_PROMPT,
) -> list[Path]:
    llm_rag_dir = ace_repo_dir / "workflows" / "tokkio" / "5.0.0-ga" / "src" / "llm-rag"
    config_py_path = llm_rag_dir / "src" / "config.py"
    config_yaml_path = llm_rag_dir / "configs" / "config.yaml"
    bot_py_path = llm_rag_dir / "src" / "bot.py"
    irodori_tts_py_path = llm_rag_dir / "src" / "tokkio_irodori_tts.py"
    tokkio_rag_py_path = llm_rag_dir / "src" / "tokkio_rag.py"
    irodori_tts_source_path = Path(__file__).with_name("tokkio_irodori_tts.py")
    tokkio_rag_source_path = Path(__file__).with_name("tokkio_rag.py")

    for path in (config_py_path, config_yaml_path, bot_py_path):
        if not path.exists():
            raise FileNotFoundError(f"required file not found: {path}")
    if not irodori_tts_source_path.exists():
        raise FileNotFoundError(f"required file not found: {irodori_tts_source_path}")
    if not tokkio_rag_source_path.exists():
        raise FileNotFoundError(f"required file not found: {tokkio_rag_source_path}")

    write_text(config_py_path, CONFIG_PY)
    write_text(
        config_yaml_path,
        build_config_yaml(
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            irodori_tts_base_url=irodori_tts_base_url,
            rag_enabled=rag_enabled,
            rag_server_url=rag_server_url,
            rag_collection_name=rag_collection_name,
            rag_use_knowledge_base=rag_use_knowledge_base,
            rag_max_tokens=rag_max_tokens,
            rag_vdb_top_k=rag_vdb_top_k,
            rag_reranker_top_k=rag_reranker_top_k,
            rag_multimodal_reranker_top_k=rag_multimodal_reranker_top_k,
            rag_enable_reranker=rag_enable_reranker,
            rag_suffix_prompt=rag_suffix_prompt,
        ),
    )
    write_text(irodori_tts_py_path, irodori_tts_source_path.read_text(encoding="utf-8"))
    write_text(tokkio_rag_py_path, tokkio_rag_source_path.read_text(encoding="utf-8"))
    patch_llm_snippet(bot_py_path)
    patch_rag_snippet(bot_py_path)
    profile_config_paths = patch_profile_ace_controller_configs(
        ace_repo_dir,
        llm_base_url=llm_base_url,
        llm_model=llm_model,
        irodori_tts_base_url=irodori_tts_base_url,
        rag_enabled=rag_enabled,
        rag_server_url=rag_server_url,
        rag_collection_name=rag_collection_name,
        rag_use_knowledge_base=rag_use_knowledge_base,
        rag_max_tokens=rag_max_tokens,
        rag_vdb_top_k=rag_vdb_top_k,
        rag_reranker_top_k=rag_reranker_top_k,
        rag_multimodal_reranker_top_k=rag_multimodal_reranker_top_k,
        rag_enable_reranker=rag_enable_reranker,
        rag_suffix_prompt=rag_suffix_prompt,
    )
    try:
        replace_once(bot_py_path, OLD_BOT_SNIPPET, NEW_BOT_SNIPPET)
    except RuntimeError:
        original = bot_py_path.read_text(encoding="utf-8")
        if OLD_CONFIG_DRIVEN_TTS_SNIPPET in original:
            replace_literal(bot_py_path, OLD_CONFIG_DRIVEN_TTS_SNIPPET, NEW_CONFIG_DRIVEN_TTS_SNIPPET)
        elif PREVIOUS_CONFIG_DRIVEN_TTS_SNIPPET in original:
            replace_literal(bot_py_path, PREVIOUS_CONFIG_DRIVEN_TTS_SNIPPET, NEW_CONFIG_DRIVEN_TTS_SNIPPET)
        elif "IrodoriTTSService(" in original:
            pass
        else:
            replace_literal(
                bot_py_path,
                "                voice_name=config.RivaTTSService.voice_name,\n",
                "                voice_id=config.RivaTTSService.voice_name,\n",
            )

    return [
        config_py_path,
        config_yaml_path,
        bot_py_path,
        irodori_tts_py_path,
        tokkio_rag_py_path,
        *profile_config_paths,
        *patch_riva_values(ace_repo_dir),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Customize Tokkio 5.0 llm-rag sources for Japanese")
    parser.add_argument(
        "--ace-repo-dir",
        type=Path,
        default=DEFAULT_ACE_REPO_DIR,
        help=f"Path to the local NVIDIA-ACE clone (default: {DEFAULT_ACE_REPO_DIR})",
    )
    parser.add_argument(
        "--llm-base-url",
        default=os.environ.get("TOKKIO_LLM_BASE_URL", DEFAULT_LLM_BASE_URL),
        help=f"OpenAI-compatible LLM base URL, including or excluding /v1 (default: {DEFAULT_LLM_BASE_URL})",
    )
    parser.add_argument(
        "--llm-model",
        default=os.environ.get("TOKKIO_LLM_MODEL", DEFAULT_LLM_MODEL),
        help=f"Model id exposed by the OpenAI-compatible LLM server (default: {DEFAULT_LLM_MODEL})",
    )
    parser.add_argument(
        "--irodori-tts-base-url",
        default=os.environ.get("TOKKIO_IRODORI_TTS_BASE_URL", DEFAULT_IRODORI_TTS_BASE_URL),
        help=f"Base URL for the host-side Irodori-TTS HTTP service (default: {DEFAULT_IRODORI_TTS_BASE_URL})",
    )
    parser.add_argument(
        "--rag-enabled",
        default=os.environ.get("TOKKIO_RAG_ENABLED", "false"),
        help="Set true to use NvidiaRAGService instead of NvidiaLLMService (default: false)",
    )
    parser.add_argument(
        "--rag-server-url",
        default=os.environ.get("TOKKIO_RAG_SERVER_URL", DEFAULT_RAG_SERVER_URL),
        help=f"NVIDIA RAG server URL, including or excluding /v1 (default: {DEFAULT_RAG_SERVER_URL})",
    )
    parser.add_argument(
        "--rag-collection-name",
        default=os.environ.get("TOKKIO_RAG_COLLECTION_NAME", DEFAULT_RAG_COLLECTION_NAME),
        help=f"RAG collection name used by Tokkio (default: {DEFAULT_RAG_COLLECTION_NAME})",
    )
    parser.add_argument(
        "--rag-use-knowledge-base",
        default=os.environ.get("TOKKIO_RAG_USE_KNOWLEDGE_BASE", "true"),
        help="Set false to call RAG generation without retrieval (default: true)",
    )
    parser.add_argument(
        "--rag-max-tokens",
        type=int,
        default=int(os.environ.get("TOKKIO_RAG_MAX_TOKENS", str(DEFAULT_RAG_MAX_TOKENS))),
        help=f"Maximum response tokens for NvidiaRAGService (default: {DEFAULT_RAG_MAX_TOKENS})",
    )
    parser.add_argument(
        "--rag-vdb-top-k",
        type=int,
        default=int(os.environ.get("TOKKIO_RAG_VDB_TOP_K", str(DEFAULT_RAG_VDB_TOP_K))),
        help=f"Number of vector DB candidates to retrieve before reranking (default: {DEFAULT_RAG_VDB_TOP_K})",
    )
    parser.add_argument(
        "--rag-reranker-top-k",
        type=int,
        default=int(os.environ.get("TOKKIO_RAG_RERANKER_TOP_K", str(DEFAULT_RAG_RERANKER_TOP_K))),
        help=f"Number of reranked chunks sent to the LLM context (default: {DEFAULT_RAG_RERANKER_TOP_K})",
    )
    parser.add_argument(
        "--rag-multimodal-reranker-top-k",
        type=int,
        default=int(os.environ.get("TOKKIO_RAG_MULTIMODAL_RERANKER_TOP_K", str(DEFAULT_RAG_MULTIMODAL_RERANKER_TOP_K))),
        help=(
            "Reranker top-k used for table/chart/image-style questions "
            f"(default: {DEFAULT_RAG_MULTIMODAL_RERANKER_TOP_K})"
        ),
    )
    parser.add_argument(
        "--rag-enable-reranker",
        default=os.environ.get("TOKKIO_RAG_ENABLE_RERANKER", "true"),
        help="Set false to skip reranking; keep true for accuracy-preserving latency tuning (default: true)",
    )
    parser.add_argument(
        "--rag-suffix-prompt",
        default=os.environ.get("TOKKIO_RAG_SUFFIX_PROMPT", DEFAULT_RAG_SUFFIX_PROMPT),
        help=f"Suffix appended to the final user prompt before RAG generation (default: {DEFAULT_RAG_SUFFIX_PROMPT})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    changed = apply_patch(
        args.ace_repo_dir.resolve(),
        llm_base_url=args.llm_base_url,
        llm_model=args.llm_model,
        irodori_tts_base_url=args.irodori_tts_base_url,
        rag_enabled=parse_bool(args.rag_enabled),
        rag_server_url=args.rag_server_url,
        rag_collection_name=args.rag_collection_name,
        rag_use_knowledge_base=parse_bool(args.rag_use_knowledge_base),
        rag_max_tokens=args.rag_max_tokens,
        rag_vdb_top_k=args.rag_vdb_top_k,
        rag_reranker_top_k=args.rag_reranker_top_k,
        rag_multimodal_reranker_top_k=args.rag_multimodal_reranker_top_k,
        rag_enable_reranker=parse_bool(args.rag_enable_reranker),
        rag_suffix_prompt=args.rag_suffix_prompt,
    )
    print("Updated files:")
    for path in changed:
        print(path)
    print("")
    print("Next steps:")
    print("1. Start Tokkio or run reapply so ACE Configurator syncs the updated app-storage-volume.")
    print("2. Restart only the controller if you need a faster reload path after sync.")
    print("3. Ensure ace-irodori-tts.service is running before starting the controller.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
