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

                if hasattr(chunk, "choices") and chunk.choices and chunk.choices[0].delta:
                    if chunk.choices[0].delta.content:
                        await self.stop_ttfb_metrics()
                        await self.push_frame(TextFrame(chunk.choices[0].delta.content))
                elif hasattr(chunk, "content") and chunk.content:
                    await self.stop_ttfb_metrics()
                    await self.push_frame(TextFrame(chunk.content))
                else:
                    logger.warning(f"Received chunk in unexpected format: {type(chunk).__name__}. Content: {chunk}")
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
