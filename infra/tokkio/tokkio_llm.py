# Copyright(c) 2025 NVIDIA Corporation. All rights reserved.

"""Tokkio LLM services with filler phrase support and local fast replies."""

import asyncio
from datetime import date
import random
import re
import time

from loguru import logger
from nvidia_pipecat.services.nvidia_llm import NvidiaLLMService
from pipecat.frames.frames import TextFrame, TTSSpeakFrame
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai.llm import OpenAILLMService


_STRIP_CHARS_RE = re.compile(r"[\s　。、．，,！？!?？!…・「」『』（）()\[\]【】]+")
_SENTENCE_END_CHARS = "。.!?！？"
_SOFT_BREAK_CHARS = "、，,;； "
_NEMOTRON_ULTRA_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
_DEFAULT_FIRST_RESPONSE_TIMEOUT_S = 20.0
_LLM_RESOURCE_RETRY_DELAY_S = 0.75
_LLM_TIMEOUT_REPLY = "回答サービスの応答に時間がかかっています。少し待ってから、もう一度お尋ねください。"
_LLM_ERROR_REPLY = "回答サービスへ接続できません。少し待ってから、もう一度お尋ねください。"
_LLM_BUSY_REPLY = "回答サービスが混み合っています。少し待ってから、もう一度お尋ねください。"
_SPEECH_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_SPEECH_HEADING_RE = re.compile(r"(^|\s+)\*\*([^*\n]{1,40})\*\*(?=\s*(?:[*+]|$))")
_KAGAWA_BIRTH_DATE = date(1952, 9, 19)

OPEN_CAMPUS_GREETING_SCRIPT = (
    "皆さん、こんにちは。",
    "東京工科大学学長の香川豊です。",
    "今日は八王子キャンパスまで来てくださり、ありがとうございます。",
    "初めて来た方は、たいていキャンパスの広さに驚かれます。",
    "ユニバーサルスタジオジャパン、いわゆるUSJとほぼ同じ広さなんですよ。",
    "東京ドームで言えば8個分です。",
    "圧迫感がなくてのびのび学生生活を送れると好評です。",
    "授業の合間の移動が大変、という声もありますけれど……。",
    "と、当たり前のように話していますが、私はAI学長です。",
    "東京工科大学は、AIユニバーシティと名乗っています。",
    "いまや、AIを使いこなせるかどうかで、就職やその後の社会人生活が大きく左右されます。",
    "八王子の4学部、蒲田の2学部全てで、何らかの形でAIを使った講義が当たり前になってきています。",
    "私を作ったのも、コンピュータサイエンス学部の学生なんですよ。",
    "ちゃんと勉強すれば、みなさんもしっかりとしたAIスキルを身に着けることができます。",
    "でも、AI学長って言ったって、アバターに音声を合成しただけじゃない、と思った方もいらっしゃるでしょう。",
    "今日はね、AI学長をオペレートする学生が休みなんですよ。",
    "ここで、開発した学生と私とのリアルなやり取りをご覧ください。",
)

OPEN_CAMPUS_CLOSING_SCRIPT = (
    "ね。このようにAI香川豊、しっかり対応できるんですよ。",
    "今後も様々なシーンでお目にかかるかと思います。",
    "4月にみなさんとお会いできることを楽しみにしています。",
    "今日は一日、楽しんでいってください。",
)

_OPEN_CAMPUS_GREETING_EXACT_CUES = {
    "挨拶をお願いします",
    "ご挨拶をお願いします",
    "オープニングをお願いします",
    "台本をお願いします",
    "台本を読んでください",
}
_OPEN_CAMPUS_GREETING_CUE_TOPICS = (
    "オープンキャンパス",
    "八王子キャンパス",
    "学長挨拶",
    "学長の挨拶",
    "学長のご挨拶",
    "オープニング",
    "台本",
    "スピーチ",
)
_OPEN_CAMPUS_GREETING_CUE_ACTIONS = (
    "お願いします",
    "読んで",
    "話して",
    "始めて",
)

_OPEN_CAMPUS_CLOSING_EXACT_CUES = {
    "締めの言葉をお願いします",
    "締めをお願いします",
    "クロージングをお願いします",
    "最後の挨拶をお願いします",
    "締めの台本を読んでください",
}
_OPEN_CAMPUS_CLOSING_CUE_TOPICS = (
    "締め",
    "クロージング",
    "最後の挨拶",
    "最後のご挨拶",
)
_OPEN_CAMPUS_CLOSING_CUE_ACTIONS = (
    "お願いします",
    "読んで",
    "話して",
    "始めて",
)

_GREETING_REPLIES = {
    "おはよう": "おはようございます。",
    "おはようございます": "おはようございます。",
    "こんにちは": "こんにちは。",
    "こんばんは": "こんばんは。",
    "はじめまして": "はじめまして。香川豊です。",
    "よろしくお願いします": "よろしくお願いします。",
    "よろしくおねがいします": "よろしくお願いします。",
    "ありがとう": "どういたしまして。",
    "ありがとうございます": "どういたしまして。",
}

_FAST_REPLY_EXCLUDED_TERMS = (
    "論文",
    "文献",
    "資料",
    "ドキュメント",
    "引用",
    "出典",
    "専門",
    "専門分野",
    "学歴",
    "職歴",
    "略歴",
    "研究",
    "業績",
    "経歴",
    "役職",
    "現職",
    "職名",
    "学位",
    "所属",
    "生年月日",
    "年齢",
    "プロフィール",
    "ebc",
    "cmc",
    "sic/sic",
    "特許",
    "受賞",
    "発表",
    "プロジェクト",
)


def nvidia_model_input_params(model: str) -> dict[str, object]:
    """Return stable realtime-chat parameters for model-specific NIM behavior."""
    if model.strip() != _NEMOTRON_ULTRA_MODEL:
        return {}
    return {
        "temperature": 0.0,
        "top_p": 0.95,
        "max_tokens": 512,
        "extra": {
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": False},
            }
        },
    }


def _message_content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return " ".join(parts)
    return str(content or "")


def _latest_user_text(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return _message_content_to_text(message.get("content"))
    return ""


def _normalize_short_utterance(text: str) -> str:
    return _STRIP_CHARS_RE.sub("", text).lower()


def _is_name_question(normalized: str) -> bool:
    if len(normalized) > 40:
        return False
    if any(term in normalized for term in _FAST_REPLY_EXCLUDED_TERMS):
        return False

    exact_questions = {
        "名前は",
        "お名前は",
        "名前を教えて",
        "お名前を教えて",
        "あなたの名前は",
        "あなたの名前は何ですか",
        "あなたは誰",
        "あなたは誰ですか",
        "あなた誰",
        "あなた誰ですか",
        "君は誰",
        "君は誰ですか",
        "きみは誰",
        "きみは誰ですか",
        "誰ですか",
        "どなたですか",
        "自己紹介して",
    }
    if normalized in exact_questions:
        return True

    if ("名前" in normalized or "お名前" in normalized) and any(
        subject in normalized for subject in ("あなた", "君", "きみ", "香川先生", "香川豊先生")
    ):
        return True

    return False


def _is_kagawa_age_question(normalized: str) -> bool:
    if len(normalized) > 50 or not any(term in normalized for term in ("年齢", "何歳", "おいくつ")):
        return False
    if any(subject in normalized for subject in ("香川", "あなた", "君", "きみ")):
        return True
    if normalized.startswith("先生"):
        return True
    return normalized in {
        "年齢は",
        "年齢を教えて",
        "何歳",
        "何歳ですか",
        "おいくつですか",
    }


def _is_open_campus_greeting_cue(normalized: str) -> bool:
    if len(normalized) > 80:
        return False
    if normalized in _OPEN_CAMPUS_GREETING_EXACT_CUES:
        return True
    return any(topic in normalized for topic in _OPEN_CAMPUS_GREETING_CUE_TOPICS) and any(
        action in normalized for action in _OPEN_CAMPUS_GREETING_CUE_ACTIONS
    )


def _is_open_campus_closing_cue(normalized: str) -> bool:
    if len(normalized) > 80:
        return False
    if normalized in _OPEN_CAMPUS_CLOSING_EXACT_CUES:
        return True
    return any(topic in normalized for topic in _OPEN_CAMPUS_CLOSING_CUE_TOPICS) and any(
        action in normalized for action in _OPEN_CAMPUS_CLOSING_CUE_ACTIONS
    )


def get_scripted_speech_reply(messages: list[dict]) -> tuple[str, ...] | None:
    """Return a deterministic multi-segment speech for an explicit operator cue."""
    normalized = _normalize_short_utterance(_latest_user_text(messages))
    if not normalized:
        return None
    if _is_open_campus_closing_cue(normalized):
        return OPEN_CAMPUS_CLOSING_SCRIPT
    if not _is_open_campus_greeting_cue(normalized):
        return None
    return OPEN_CAMPUS_GREETING_SCRIPT


def _kagawa_age_on(today: date) -> int:
    before_birthday = (today.month, today.day) < (_KAGAWA_BIRTH_DATE.month, _KAGAWA_BIRTH_DATE.day)
    return today.year - _KAGAWA_BIRTH_DATE.year - int(before_birthday)


def get_fast_profile_reply(messages: list[dict], *, today: date | None = None) -> str | None:
    """Return a local reply for short deterministic turns that do not need LLM/RAG."""
    normalized = _normalize_short_utterance(_latest_user_text(messages))
    if not normalized:
        return None

    if normalized in _GREETING_REPLIES:
        return _GREETING_REPLIES[normalized]

    if _is_kagawa_age_question(normalized):
        age = _kagawa_age_on(today or date.today())
        return f"私は1952年9月19日生まれで、現在{age}歳です。"

    if _is_name_question(normalized):
        return "私は香川豊です。"

    return None


def normalize_speech_segment(text: str) -> str:
    """Convert streamed Markdown-like prose into stable spoken sentences."""

    normalized = _SPEECH_MARKDOWN_LINK_RE.sub(r"\1", text)
    normalized = _SPEECH_HEADING_RE.sub(
        lambda match: f"{'。' if match.start() else ''}{match.group(2)}。",
        normalized,
    )
    normalized = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", normalized)
    normalized = re.sub(r"(?m)^\s*[-*+]\s*", "", normalized)
    normalized = re.sub(r"(?:\*\*|__|~~|`)", "", normalized)
    normalized = re.sub(r"\s+[*+]\s*", "。", normalized)
    normalized = re.sub(r"[\r\n]+", "。", normalized)
    normalized = normalized.replace("•", "。").replace("◦", "。").replace("*", "。")
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"。(?:\s*。)+", "。", normalized)
    normalized = re.sub(r"\s*。\s*", "。", normalized)
    return normalized.strip(" 。") + ("。" if normalized.strip(" 。") and normalized.rstrip().endswith("。") else "")


class SpeechSegmentBuffer:
    """Buffers LLM deltas into short, stable TTS phrases."""

    def __init__(
        self,
        *,
        soft_max_chars: int = 28,
        hard_max_chars: int = 56,
        min_segment_chars: int = 8,
    ) -> None:
        self._buffer = ""
        self._soft_max_chars = soft_max_chars
        self._hard_max_chars = hard_max_chars
        self._min_segment_chars = min_segment_chars

    def feed(self, text: str) -> list[str]:
        self._buffer += text
        return self._pop_ready_segments()

    def flush(self) -> list[str]:
        segment = normalize_speech_segment(self._buffer.strip())
        self._buffer = ""
        return [segment] if segment else []

    def _pop_ready_segments(self) -> list[str]:
        segments: list[str] = []
        while self._buffer:
            sentence_index = self._first_sentence_end_index()
            if sentence_index is not None:
                segment = self._take(sentence_index + 1)
                if segment:
                    segments.append(segment)
                continue

            if len(self._buffer) >= self._soft_max_chars:
                soft_index = self._last_soft_break_index(self._hard_max_chars)
                if soft_index is not None:
                    segment = self._take(soft_index + 1)
                    if segment:
                        segments.append(segment)
                    continue

            if len(self._buffer) >= self._hard_max_chars:
                segment = self._take(self._hard_max_chars)
                if segment:
                    segments.append(segment)
                continue

            break
        return segments

    def _first_sentence_end_index(self) -> int | None:
        indexes = [self._buffer.find(char) for char in _SENTENCE_END_CHARS]
        indexes = [index for index in indexes if index >= 0]
        return min(indexes) if indexes else None

    def _last_soft_break_index(self, max_chars: int) -> int | None:
        search_text = self._buffer[:max_chars]
        indexes = [search_text.rfind(char) for char in _SOFT_BREAK_CHARS]
        indexes = [index for index in indexes if index + 1 >= self._min_segment_chars]
        return max(indexes) if indexes else None

    def _take(self, end_index: int) -> str:
        raw_segment = self._buffer[:end_index].strip()
        self._buffer = self._buffer[end_index:].lstrip()
        segment = normalize_speech_segment(raw_segment)
        if self._buffer.startswith(("*", "+")) and segment and not segment.endswith(tuple(_SENTENCE_END_CHARS)):
            segment += "。"
        if segment and not any(char not in _SENTENCE_END_CHARS for char in segment):
            return ""
        return segment


class TokkioLLMServiceMixin:
    llm_resource_retry_delay_s = _LLM_RESOURCE_RETRY_DELAY_S

    @staticmethod
    def _text_delta_from_chunk(chunk) -> str:
        if hasattr(chunk, "choices") and chunk.choices and chunk.choices[0].delta:
            return chunk.choices[0].delta.content or ""
        if hasattr(chunk, "content") and chunk.content:
            return chunk.content
        logger.warning(f"Received chunk in unexpected format: {type(chunk).__name__}. Content: {chunk}")
        return ""

    async def _open_stream_and_read_first_text(self, context, stream_chat_completions):
        chunk_stream = await stream_chat_completions(context)
        async for chunk in chunk_stream:
            text_delta = self._text_delta_from_chunk(chunk)
            if text_delta:
                return chunk_stream, text_delta
        raise RuntimeError("LLM endpoint returned an empty response")

    @staticmethod
    def _is_retryable_resource_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return "resourceexhausted" in message or "request limit reached" in message

    async def _open_stream_and_read_first_text_with_retry(self, context, stream_chat_completions):
        try:
            return await self._open_stream_and_read_first_text(context, stream_chat_completions)
        except Exception as exc:
            if not self._is_retryable_resource_error(exc):
                raise
            logger.warning(
                "LLM worker capacity was exhausted; retrying once after {} seconds",
                self.llm_resource_retry_delay_s,
            )
            await asyncio.sleep(self.llm_resource_retry_delay_s)
            return await self._open_stream_and_read_first_text(context, stream_chat_completions)

    async def _push_local_reply_if_available(self, context: OpenAILLMContext) -> bool:
        messages = context.get_messages()
        scripted_reply = get_scripted_speech_reply(messages)
        profile_reply = get_fast_profile_reply(messages) if not scripted_reply else None
        segments = scripted_reply or ((profile_reply,) if profile_reply else ())
        if not segments:
            return False

        logger.debug(f"Tokkio selected local reply with {len(segments)} speech segment(s)")
        await self.start_ttfb_metrics()
        await self.stop_ttfb_metrics()
        for segment in segments:
            await self.push_frame(TextFrame(segment))
        return True

    async def _process_context_common(self, context: OpenAILLMContext, stream_chat_completions):
        """Process an LLM context with filler phrase support."""
        if await self._push_local_reply_if_available(context):
            return

        await self.start_ttfb_metrics()

        first_chunk_received = False
        filler_said = False
        start_time = time.time()

        async def monitor_request_time():
            nonlocal filler_said
            await asyncio.sleep(self.time_delay)
            if self.filler and not first_chunk_received and not filler_said:
                filler_said = True
                await self.push_frame(TTSSpeakFrame(random.choice(self.filler)))

        monitor_task = asyncio.create_task(monitor_request_time())

        try:
            chunk_stream, first_text_delta = await asyncio.wait_for(
                self._open_stream_and_read_first_text_with_retry(context, stream_chat_completions),
                timeout=self.first_response_timeout_s,
            )
            speech_buffer = SpeechSegmentBuffer()
            ttfb_stopped = False

            elapsed_time = time.time() - start_time
            logger.debug(f"Elapsed time: {elapsed_time}")
            logger.debug(f"Time delay: {self.time_delay}")
            first_chunk_received = True
            if not monitor_task.done():
                monitor_task.cancel()
                try:
                    await monitor_task
                except asyncio.CancelledError:
                    pass

            await self.stop_ttfb_metrics()
            ttfb_stopped = True
            for segment in speech_buffer.feed(first_text_delta):
                await self.push_frame(TextFrame(segment))

            async for chunk in chunk_stream:
                text_delta = self._text_delta_from_chunk(chunk)
                if text_delta:
                    for segment in speech_buffer.feed(text_delta):
                        await self.push_frame(TextFrame(segment))
            for segment in speech_buffer.flush():
                await self.push_frame(TextFrame(segment))
        except asyncio.TimeoutError:
            logger.warning(
                f"LLM endpoint did not return response text within {self.first_response_timeout_s} seconds"
            )
            await self.stop_ttfb_metrics()
            await self.push_frame(TTSSpeakFrame(_LLM_TIMEOUT_REPLY))
        except Exception as e:
            logger.error(f"An error occurred in http request to LLM endpoint, Error: {e}")
            await self.stop_ttfb_metrics()
            reply = _LLM_BUSY_REPLY if self._is_retryable_resource_error(e) else _LLM_ERROR_REPLY
            await self.push_frame(TTSSpeakFrame(reply))

        finally:
            if not monitor_task.done():
                monitor_task.cancel()
                try:
                    await monitor_task
                except asyncio.CancelledError:
                    pass


class TokkioNvidiaLLMService(NvidiaLLMService, TokkioLLMServiceMixin):
    def __init__(
        self,
        filler: list[str],
        time_delay: float = 1.0,
        first_response_timeout_s: float = _DEFAULT_FIRST_RESPONSE_TIMEOUT_S,
        **kwargs,
    ):
        self.filler = filler
        self.time_delay = time_delay
        self.first_response_timeout_s = first_response_timeout_s
        if "params" not in kwargs:
            model_params = nvidia_model_input_params(str(kwargs.get("model", "")))
            if model_params:
                kwargs["params"] = OpenAILLMService.InputParams(**model_params)
        super().__init__(**kwargs)

    async def _process_context(self, context: OpenAILLMContext):
        await self._process_context_common(context, self._stream_chat_completions)


class TokkioOpenAILLMService(OpenAILLMService, TokkioLLMServiceMixin):
    def __init__(
        self,
        filler: list[str],
        time_delay: float = 1.0,
        first_response_timeout_s: float = _DEFAULT_FIRST_RESPONSE_TIMEOUT_S,
        **kwargs,
    ):
        self.filler = filler
        self.time_delay = time_delay
        self.first_response_timeout_s = first_response_timeout_s
        super().__init__(**kwargs)

    async def _process_context(self, context: OpenAILLMContext):
        await self._process_context_common(context, self._stream_chat_completions)
