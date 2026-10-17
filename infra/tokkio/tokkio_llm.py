# Copyright(c) 2025 NVIDIA Corporation. All rights reserved.

"""Tokkio LLM services with filler phrase support and local fast replies."""

import asyncio
from datetime import date
import random
import re
import time

from loguru import logger
from nvidia_pipecat.services.nvidia_llm import NvidiaLLMService
from pipecat.frames.frames import (
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TextFrame,
    TTSSpeakFrame,
)
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai.llm import OpenAILLMService


_STRIP_CHARS_RE = re.compile(r"[\s　。、．.，,！？!?？!…・「」『』（）()\[\]【】]+")
_SENTENCE_END_CHARS = "。.!?！？"
_SOFT_BREAK_CHARS = "、，,;； "
_CLOSING_BRACKET_CHARS = "」』）)】〉》］]”\""
_NEMOTRON_MODELS_WITH_THINKING = (
    "nvidia/nemotron-3-ultra-550b-a55b",
    "nvidia/nemotron-3.5-lightning-30b-a3b",
)
_DEFAULT_FIRST_RESPONSE_TIMEOUT_S = 20.0
_LLM_RESOURCE_RETRY_DELAY_S = 0.75
_LLM_TIMEOUT_REPLY = "回答サービスの応答に時間がかかっています。少し待ってから、もう一度お尋ねください。"
_LLM_ERROR_REPLY = "回答サービスへ接続できません。少し待ってから、もう一度お尋ねください。"
_LLM_BUSY_REPLY = "回答サービスが混み合っています。少し待ってから、もう一度お尋ねください。"
_SPEECH_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_SPEECH_HEADING_RE = re.compile(r"(^|\s+)\*\*([^*\n]{1,40})\*\*(?=\s*(?:[*+]|$))")
# "→" / "⇒" are read as a pause instead of being deleted ("A→B" must not become "AB").
_SPEECH_ARROW_RE = re.compile(r"\s*[→⇒]\s*")
# Emoji / pictographs / dingbats / decorative symbols that must never be spoken.
# ○ (U+25CB) and ◯ (U+25EF) are kept: they stand for "maru" and for placeholders.
# ▲ (U+25B2) is left to _SPEECH_DECORATIVE_TRIANGLE_RE: before a digit it is a
# minus sign (▲5% = マイナス5%) that the TTS server reads.
_SPEECH_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\u2600-\u27bf\u2b00-\u2bff\ufe0f\ufe0e"
    "\u2190-\u21ff\u2300-\u23ff\u25a0-\u25b1\u25b3-\u25ca\u25cc-\u25ee\u25f0-\u25ff"
    "\u00a9\u00ae\u2122\u3030\u303d\u3297\u3299]"
)
_SPEECH_DECORATIVE_TRIANGLE_RE = re.compile(r"▲(?!\d)")
# Meta-instruction echoes like （100文字以内、2文） leaked from the system prompt.
# A bare 以内 is not enough: （徒歩10分以内） is real content.  A character count
# is removed only on its own (（約58文字）) or in the prompt's constraint shapes;
# （半角英数字8文字） and （最大20文字） are content.
_SPEECH_META_NOTE_RE = re.compile(
    r"[（(]\s*(?:約|全角|計|合計)?\s*\d+\s*文字\s*(?:程度|前後|くらい)?\s*[）)]"
    r"|[（(][^（）()]{0,40}?"
    r"(?:\d+\s*(?:から|〜|～|~|-)\s*\d+\s*文字|\d+\s*文字(?:以内|まで|程度|で|に)|文字(?:以内|程度|で|に)"
    r"|\d+\s*文(?![字書献化学章庫])(?:まで|で|以内)?)"
    r"(?!以上|以下|未満)[^（）()]{0,20}?[）)]"
)
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

SYMPOSIUM_OPENING_SCRIPT = (
    "Good afternoon, everyone.",
    "Centuries ago, the poet Saigyo wrote of a blue gale blowing through Hachioji, the City of Mulberries.",
    "Inspired by this poem, we named our supercomputer Seiran, the Blue Gale.",
    "Today, a new wind of knowledge is rising from Hachioji.",
    "Welcome to the Commemorative Symposium celebrating the establishment of the Tokyo University of Technology and NVIDIA AI Technology Center Joint Lab.",
    "Please enjoy the symposium, along with the rich history and culture of Hachioji.",
    "And now, please welcome Professor Kagawa, the President of the Tokyo University of Technology.",
)

_SYMPOSIUM_OPENING_EXACT_CUES = {
    "シンポジウムのオープニングをお願いします",
    "シンポジウムの開会挨拶をお願いします",
    "シンポジウムの台本を読んでください",
    "readthesymposiumopening",
    "pleasereadthesymposiumopening",
    "start",
}
# Cue patterns must match the whole normalized utterance.  Substring matching
# fired the scripts on questions such as "What time does the symposium start?"
# or "シンポジウムの説明をお願いします".  Only a short, known lead-in and
# ASR/IME spelling variants are allowed around the whitelisted nouns.
_CUE_REQUEST = (
    r"(?:を?(?:お願い|おねがい)(?:します|いたします|致します)?"
    r"|を?(?:読んで|よんで|話して|始めて)(?:ください|下さい)?)"
)
_CUE_LEAD_IN = r"(?:(?:それ)?では|じゃあ|はい|香川(?:先生|学長)|学長|先生)?"
# 最後に is a closing lead-in only: "最後にご挨拶" must not pick the greeting.
_CLOSING_CUE_LEAD_IN = r"(?:(?:それ)?では|じゃあ|はい|最後に|香川(?:先生|学長)|学長|先生)?"
_CUE_GREETING_NOUN = r"(?:ご|御)?(?:挨拶|あいさつ)"
# Fillers that ASR leaves around a cue ("えー、…お願いします。えー").
_CUE_TRAILING_FILLER_RE = re.compile(r"(?:えー|あー|えっと)+$")
_CUE_LEADING_FILLER_RE = re.compile(r"^(?:えー+|あー+|えっと|あのー*)+")
_SYMPOSIUM_OPENING_CUE_RE = re.compile(
    rf"{_CUE_LEAD_IN}シンポジウムの(?:オープニング|開会挨拶|開会の挨拶|台本){_CUE_REQUEST}"
    r"|(?:okay|ok|so|now|let's|lets)?(?:please)?(?:read|start|begin)(?:the)?symposium(?:opening)?(?:please)?"
)
_OPERATOR_STOP_CUE_RE = re.compile(
    r"(?:ok(?:ay)?|please)?stop(?:it|please|now|speaking|talking|stop)*"
    r"|(?:ちょっと)?(?:(?:ストップ)+(?:して)?|[止と]めて|停止(?:して)?)(?:ください)?"
)

_OPEN_CAMPUS_GREETING_EXACT_CUES = {
    "挨拶をお願いします",
    "ご挨拶をお願いします",
    "オープニングをお願いします",
    "台本をお願いします",
    "台本を読んでください",
}
_OPEN_CAMPUS_GREETING_CUE_RE = re.compile(
    rf"{_CUE_LEAD_IN}(?:(?:オープンキャンパス|八王子キャンパス)の)?"
    rf"(?:学長の?{_CUE_GREETING_NOUN}|{_CUE_GREETING_NOUN}|オープニング|台本|スピーチ){_CUE_REQUEST}"
)

_OPEN_CAMPUS_CLOSING_EXACT_CUES = {
    "締めの言葉をお願いします",
    "締めをお願いします",
    "クロージングをお願いします",
    "最後の挨拶をお願いします",
    "締めの台本を読んでください",
}
_OPEN_CAMPUS_CLOSING_CUE_RE = re.compile(
    rf"{_CLOSING_CUE_LEAD_IN}(?:オープンキャンパスの)?"
    rf"(?:締め(?:の(?:言葉|{_CUE_GREETING_NOUN}|台本))?|クロージング|最後の{_CUE_GREETING_NOUN}){_CUE_REQUEST}"
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
    if model.strip() not in _NEMOTRON_MODELS_WITH_THINKING:
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


def _is_symposium_opening_cue(normalized: str) -> bool:
    if len(normalized) > 100:
        return False
    if normalized in _SYMPOSIUM_OPENING_EXACT_CUES:
        return True
    return _SYMPOSIUM_OPENING_CUE_RE.fullmatch(normalized) is not None


def _is_open_campus_greeting_cue(normalized: str) -> bool:
    if len(normalized) > 80:
        return False
    if normalized in _OPEN_CAMPUS_GREETING_EXACT_CUES:
        return True
    return _OPEN_CAMPUS_GREETING_CUE_RE.fullmatch(normalized) is not None


def _is_open_campus_closing_cue(normalized: str) -> bool:
    if len(normalized) > 80:
        return False
    if normalized in _OPEN_CAMPUS_CLOSING_EXACT_CUES:
        return True
    return _OPEN_CAMPUS_CLOSING_CUE_RE.fullmatch(normalized) is not None


def _is_operator_stop_cue(normalized: str) -> bool:
    return _OPERATOR_STOP_CUE_RE.fullmatch(normalized) is not None


def get_scripted_speech_reply(messages: list[dict]) -> tuple[str, ...] | None:
    """Return a deterministic multi-segment speech for an explicit operator cue."""
    normalized = _normalize_short_utterance(_latest_user_text(messages))
    normalized = _CUE_LEADING_FILLER_RE.sub("", _CUE_TRAILING_FILLER_RE.sub("", normalized))
    if not normalized:
        return None
    if _is_symposium_opening_cue(normalized):
        return SYMPOSIUM_OPENING_SCRIPT
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
    normalized = _SPEECH_META_NOTE_RE.sub("", normalized)
    normalized = re.sub(r"\.{2,}", "…", normalized)
    normalized = _SPEECH_ARROW_RE.sub("、", normalized)
    normalized = _SPEECH_EMOJI_RE.sub("", normalized)
    normalized = _SPEECH_DECORATIVE_TRIANGLE_RE.sub("", normalized)
    normalized = _SPEECH_HEADING_RE.sub(
        lambda match: f"{'。' if match.start() else ''}{match.group(2)}。",
        normalized,
    )
    normalized = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", normalized)
    # A sign before a digit is not a bullet: "-40℃" is read as マイナス.
    normalized = re.sub(r"(?m)^\s*[-*+](?!\d)\s*", "", normalized)
    normalized = re.sub(r"(?:\*\*|__|~~|`)", "", normalized)
    normalized = re.sub(r"\s+\+\s+", "プラス", normalized)
    normalized = re.sub(r"\s+\*\s*", "。", normalized)
    normalized = re.sub(r"[\r\n]+", "。", normalized)
    normalized = normalized.replace("•", "。").replace("◦", "。").replace("*", "。")
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"。(?:\s*。)+", "。", normalized)
    normalized = re.sub(r"\s*。\s*", "。", normalized)
    return normalized.strip(" 。") + ("。" if normalized.strip(" 。") and normalized.rstrip().endswith("。") else "")


def _is_ascii_alnum(value: str) -> bool:
    return value.isascii() and value.isalnum()


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

    def flush_complete(self) -> list[str]:
        """Emit a held complete sentence; drop an unfinished fragment."""
        if self._buffer.rstrip().rstrip(_CLOSING_BRACKET_CHARS).endswith(tuple(_SENTENCE_END_CHARS)):
            return self.flush()
        self._buffer = ""
        return []

    def _pop_ready_segments(self) -> list[str]:
        segments: list[str] = []
        while self._buffer:
            sentence_index = self._first_sentence_end_index()
            if sentence_index is not None:
                end_index = sentence_index + 1
                while end_index < len(self._buffer) and self._buffer[end_index] in _CLOSING_BRACKET_CHARS:
                    end_index += 1
                if end_index == len(self._buffer):
                    # The next delta may still close a bracket (「…です。」と) or
                    # bring the space that separates two words.
                    break
                segment = self._take(end_index)
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
        for index, char in enumerate(self._buffer):
            if char in _SENTENCE_END_CHARS and not self._is_inside_token(index):
                return index
        return None

    def _last_soft_break_index(self, max_chars: int) -> int | None:
        search_text = self._buffer[:max_chars]
        indexes = [
            index
            for index, char in enumerate(search_text)
            if char in _SOFT_BREAK_CHARS
            and index + 1 >= self._min_segment_chars
            and not self._is_inside_token(index)
            # Only the next delta shows whether a space follows an ASCII comma.
            and not (char in ",;" and index + 1 == len(self._buffer))
        ]
        return max(indexes) if indexes else None

    def _is_inside_token(self, index: int) -> bool:
        """Keep "1.5", "250,000", "v4.1" and "..." in one segment.

        A "." after an ASCII letter/digit, or a "," after a digit, is only a
        boundary once the next delta shows it is not followed by the same kind
        of character; splitting "1." / "5倍" makes TTS read "いち。ごばい".
        A run of dots is an ellipsis, never a sentence end.
        """
        char = self._buffer[index]
        if char == "." and "." in (self._buffer[index - 1 : index], self._buffer[index + 1 : index + 2]):
            return True
        if index == 0 or char not in ".,，":
            return False
        joins = _is_ascii_alnum if char == "." else str.isdigit
        following = self._buffer[index + 1 : index + 2]
        return joins(self._buffer[index - 1]) and (not following or joins(following))

    def _take(self, end_index: int) -> str:
        # pipecat's sentence aggregator joins segments with no separator, so the
        # space consumed by this cut must stay on the segment ("of Technology").
        cut_at_space = (
            self._buffer[end_index - 1 : end_index].isspace() or self._buffer[end_index : end_index + 1].isspace()
        )
        raw_segment = self._buffer[:end_index].strip()
        self._buffer = self._buffer[end_index:].lstrip()
        segment = normalize_speech_segment(raw_segment)
        if self._buffer.startswith(("*", "+")) and segment and not segment.endswith(tuple(_SENTENCE_END_CHARS)):
            segment += "。"
        if segment and not any(char not in _SENTENCE_END_CHARS for char in segment):
            return ""
        # Either side may be the English word: "campus — and", "students’ research".
        if cut_at_space and segment and (
            segment[-1].isascii() or not self._buffer or _is_ascii_alnum(self._buffer[0])
        ):
            segment += " "
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
        return (
            "resourceexhausted" in message
            or "request limit reached" in message
            or "temporarily overloaded" in message
        )

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
        if _is_operator_stop_cue(_normalize_short_utterance(_latest_user_text(messages))):
            logger.debug("Tokkio handled operator stop cue without a spoken reply")
            await self.start_ttfb_metrics()
            await self.stop_ttfb_metrics()
            return True

        scripted_reply = get_scripted_speech_reply(messages)
        profile_reply = get_fast_profile_reply(messages) if not scripted_reply else None
        segments = scripted_reply or ((profile_reply,) if profile_reply else ())
        if not segments:
            return False

        logger.debug(f"Tokkio selected local reply with {len(segments)} speech segment(s)")
        await self.start_ttfb_metrics()
        await self.stop_ttfb_metrics()
        # The assistant context aggregator only records TextFrames received
        # between LLMFullResponseStartFrame and LLMFullResponseEndFrame.
        await self.push_frame(LLMFullResponseStartFrame())
        for segment in segments:
            await self.push_frame(TextFrame(segment))
        await self.push_frame(LLMFullResponseEndFrame())
        return True

    async def _process_context_common(self, context: OpenAILLMContext, stream_chat_completions):
        """Process an LLM context with filler phrase support."""
        if await self._push_local_reply_if_available(context):
            return

        await self.start_ttfb_metrics()

        first_chunk_received = False
        filler_said = False
        response_started = False
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
            # Wrap streamed text so the assistant context aggregator records it.
            await self.push_frame(LLMFullResponseStartFrame())
            response_started = True
            for segment in speech_buffer.feed(first_text_delta):
                await self.push_frame(TextFrame(segment))

            async for chunk in chunk_stream:
                text_delta = self._text_delta_from_chunk(chunk)
                if text_delta:
                    for segment in speech_buffer.feed(text_delta):
                        await self.push_frame(TextFrame(segment))
            for segment in speech_buffer.flush():
                await self.push_frame(TextFrame(segment))
            await self.push_frame(LLMFullResponseEndFrame())
            response_started = False
        except asyncio.TimeoutError:
            logger.warning(
                f"LLM endpoint did not return response text within {self.first_response_timeout_s} seconds"
            )
            await self.stop_ttfb_metrics()
            await self.push_frame(TTSSpeakFrame(_LLM_TIMEOUT_REPLY))
        except Exception as e:
            logger.error(f"An error occurred in http request to LLM endpoint, Error: {e}")
            await self.stop_ttfb_metrics()
            if response_started:
                # The buffer holds a finished sentence until the next delta.
                # Not in finally: after a barge-in that text is stale.
                for segment in speech_buffer.flush_complete():
                    await self.push_frame(TextFrame(segment))
            reply = _LLM_BUSY_REPLY if self._is_retryable_resource_error(e) else _LLM_ERROR_REPLY
            await self.push_frame(TTSSpeakFrame(reply))

        finally:
            if response_started:
                response_started = False
                await self.push_frame(LLMFullResponseEndFrame())
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
