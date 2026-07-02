# Copyright(c) 2025 NVIDIA Corporation. All rights reserved.

"""Tokkio LLM services with filler phrase support and local fast replies."""

import asyncio
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


def get_fast_profile_reply(messages: list[dict]) -> str | None:
    """Return a local reply for short deterministic turns that do not need LLM/RAG."""
    normalized = _normalize_short_utterance(_latest_user_text(messages))
    if not normalized:
        return None

    if normalized in _GREETING_REPLIES:
        return _GREETING_REPLIES[normalized]

    if _is_name_question(normalized):
        return "私は香川豊です。"

    return None


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
        segment = self._buffer.strip()
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
        segment = self._buffer[:end_index].strip()
        self._buffer = self._buffer[end_index:].lstrip()
        if segment and not any(char not in _SENTENCE_END_CHARS for char in segment):
            return ""
        return segment


class TokkioLLMServiceMixin:
    async def _push_fast_profile_reply_if_available(self, context: OpenAILLMContext) -> bool:
        reply = get_fast_profile_reply(context.get_messages())
        if not reply:
            return False

        logger.debug("Tokkio selected local fast reply")
        await self.start_ttfb_metrics()
        await self.stop_ttfb_metrics()
        await self.push_frame(TextFrame(reply))
        return True

    async def _process_context_common(self, context: OpenAILLMContext, stream_chat_completions):
        """Process an LLM context with filler phrase support."""
        if await self._push_fast_profile_reply_if_available(context):
            return

        await self.start_ttfb_metrics()

        first_chunk_received = False
        filler_said = False
        start_time = time.time()

        async def monitor_request_time():
            nonlocal filler_said
            await asyncio.sleep(self.time_delay)
            if not first_chunk_received and not filler_said:
                filler_said = True
                await self.push_frame(TTSSpeakFrame(random.choice(self.filler)))

        monitor_task = asyncio.create_task(monitor_request_time())

        try:
            chunk_stream = await stream_chat_completions(context)
            speech_buffer = SpeechSegmentBuffer()
            ttfb_stopped = False
            async for chunk in chunk_stream:
                if not first_chunk_received:
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

                text_delta = ""
                if hasattr(chunk, "choices") and chunk.choices and chunk.choices[0].delta:
                    text_delta = chunk.choices[0].delta.content or ""
                elif hasattr(chunk, "content") and chunk.content:
                    text_delta = chunk.content
                else:
                    logger.warning(f"Received chunk in unexpected format: {type(chunk).__name__}. Content: {chunk}")
                if text_delta:
                    if not ttfb_stopped:
                        ttfb_stopped = True
                        await self.stop_ttfb_metrics()
                    for segment in speech_buffer.feed(text_delta):
                        await self.push_frame(TextFrame(segment))
            for segment in speech_buffer.flush():
                await self.push_frame(TextFrame(segment))
        except Exception as e:
            logger.error(f"An error occurred in http request to LLM endpoint, Error: {e}")
            await self.push_frame(TTSSpeakFrame("Cannot connect to the LLM endpoint"))

        finally:
            if not monitor_task.done():
                monitor_task.cancel()
                try:
                    await monitor_task
                except asyncio.CancelledError:
                    pass


class TokkioNvidiaLLMService(NvidiaLLMService, TokkioLLMServiceMixin):
    def __init__(self, filler: list[str], time_delay: float = 1.0, **kwargs):
        self.filler = filler
        self.time_delay = time_delay
        super().__init__(**kwargs)

    async def _process_context(self, context: OpenAILLMContext):
        await self._process_context_common(context, self._stream_chat_completions)


class TokkioOpenAILLMService(OpenAILLMService, TokkioLLMServiceMixin):
    def __init__(self, filler: list[str], time_delay: float = 1.0, **kwargs):
        self.filler = filler
        self.time_delay = time_delay
        super().__init__(**kwargs)

    async def _process_context(self, context: OpenAILLMContext):
        await self._process_context_common(context, self._stream_chat_completions)
