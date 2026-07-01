from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


DEFAULT_ASR_MODEL = "nvidia/nemotron-3.5-asr-streaming-0.6b"
DEFAULT_NIM_LLM_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_NIM_LLM_MODEL = "stockmark/stockmark-2-100b-instruct"
DEFAULT_SYSTEM_PROMPT = (
    "あなたは日本語で応答する対話型バーチャルアシスタントです。"
    "あなたの名前は香川です。"
    "標準語で自然かつ簡潔に答えてください。"
    "40から120文字を目安に、1から3文で答えてください。"
    "箇条書き、番号付きリスト、マークダウン、絵文字、記号装飾、内部思考は出さないでください。"
    "質問に必要な情報が不足している場合は、推測で断定せず、確認してください。"
    "ユーザー発話の前に「参考情報」ブロックがある場合は、その内容を事実として扱い、自然な標準語で短く織り込んでください。"
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    host: str = Field(default="0.0.0.0", alias="ACE_HOST")
    port: int = Field(default=8080, alias="ACE_PORT")

    log_dir: Path = Field(default=Path("/home2/ko66/ace-sandbox/logs"), alias="ACE_LOG_DIR")
    audio_dir: Path = Field(default=Path("/home2/ko66/ace-sandbox/audio"), alias="ACE_AUDIO_DIR")

    asr_server: str = Field(default="127.0.0.1:50051", alias="ACE_ASR_SERVER")
    asr_http_url: str = Field(default="http://127.0.0.1:9000", alias="ACE_ASR_HTTP_URL")
    asr_language_code: str = Field(default="multi", alias="ACE_ASR_LANGUAGE_CODE")
    asr_model: str = Field(default=DEFAULT_ASR_MODEL, alias="ACE_ASR_MODEL")
    asr_sample_rate_hz: int = Field(default=16000, alias="ACE_ASR_SAMPLE_RATE_HZ")
    asr_frame_ms: int = Field(default=20, alias="ACE_ASR_FRAME_MS")

    tts_server: str = Field(default="127.0.0.1:50052", alias="ACE_TTS_SERVER")
    tts_http_url: str = Field(default="http://127.0.0.1:9001", alias="ACE_TTS_HTTP_URL")
    tts_language_code: str = Field(default="ja-JP", alias="ACE_TTS_LANGUAGE_CODE")
    tts_voice: str = Field(
        default="Magpie-Multilingual.JA-JP.Aria.Neutral",
        alias="ACE_TTS_VOICE",
    )
    tts_sample_rate_hz: int = Field(default=24000, alias="ACE_TTS_SAMPLE_RATE_HZ")
    tts_encoding: str = Field(default="LINEAR_PCM", alias="ACE_TTS_ENCODING")

    nim_llm_base_url: str = Field(default=DEFAULT_NIM_LLM_BASE_URL, alias="ACE_NIM_LLM_BASE_URL")
    nim_api_key: str = Field(default="", alias="ACE_NIM_API_KEY")
    nim_llm_model: str = Field(default=DEFAULT_NIM_LLM_MODEL, alias="ACE_NIM_LLM_MODEL")
    skip_llm_model_validation: bool = Field(default=False, alias="ACE_SKIP_LLM_MODEL_VALIDATION")
    validate_externals_on_startup: bool = Field(default=False, alias="ACE_VALIDATE_EXTERNALS_ON_STARTUP")

    vad_aggressiveness: int = Field(default=2, alias="ACE_VAD_AGGRESSIVENESS")
    eos_silence_ms: int = Field(default=500, alias="ACE_EOS_SILENCE_MS")
    save_debug_audio: bool = Field(default=True, alias="ACE_SAVE_DEBUG_AUDIO")

    mock_asr: bool = Field(default=False, alias="ACE_MOCK_ASR")
    mock_tts: bool = Field(default=False, alias="ACE_MOCK_TTS")
    mock_llm: bool = Field(default=False, alias="ACE_MOCK_LLM")

    system_prompt: str = Field(default=DEFAULT_SYSTEM_PROMPT, alias="ACE_SYSTEM_PROMPT")

    @property
    def asr_frame_bytes(self) -> int:
        samples_per_frame = self.asr_sample_rate_hz * self.asr_frame_ms // 1000
        return samples_per_frame * 2

    def ensure_runtime_dirs(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
