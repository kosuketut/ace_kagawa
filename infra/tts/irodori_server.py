#!/usr/bin/env python3
"""HTTP wrapper for Irodori-TTS inference.

The module is intentionally cheap to import. FastAPI, torch, and Irodori are
loaded only when the app or runtime is created, so repo-level tests can validate
configuration and audio conversion without loading model dependencies.
"""

import argparse
import asyncio
from collections import OrderedDict
from dataclasses import dataclass
from io import BytesIO
import os
from pathlib import Path
import re
import subprocess
from threading import Lock
from typing import Iterator, Mapping
import wave

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = Path("/data/ACE/irodori")
DEFAULT_HF_CHECKPOINT = "Aratako/Irodori-TTS-v4.1-Small"
DEFAULT_REFERENCE_SOURCE = ROOT / "Irodori-TTS" / "data" / "kagawa_voice.m4a"
DEFAULT_REFERENCE_WAV = DEFAULT_DATA_ROOT / "reference" / "kagawa_voice_ref_48k_mono.wav"
DEFAULT_FIXED_OPEN_CAMPUS_GREETING_PCM = (
    DEFAULT_DATA_ROOT / "fixed-phrases" / "open_campus_greeting_16k_mono.pcm"
)
DEFAULT_VOICE = "kagawa"
# Irodori's SamplingRequest.max_seconds default.
MAX_SYNTHESIS_SECONDS = 30.0
MIN_CAPPED_SECONDS = 1.2
FIXED_OPEN_CAMPUS_GREETING_TEXT = "皆さん、こんにちは"
PRONUNCIATION_HINTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"香川\s*豊"), "香川ゆたか"),
    (re.compile(r"(?<![A-Za-z])USJ(?![A-Za-z])", re.IGNORECASE), "ユーエスジェイ"),
    (
        re.compile(
            r"青嵐\s*[（(]\s*(?:せいらん(?:\s*[,、]\s*SEIRAN)?|SEIRAN)\s*[）)]",
            re.IGNORECASE,
        ),
        "せいらん",
    ),
    (re.compile(r"青嵐"), "せいらん"),
    (re.compile(r"(?<![A-Za-z])TOP\s*500(?!\d)", re.IGNORECASE), "トップごひゃく"),
    (re.compile(r"(?<![A-Za-z])AO(?![A-Za-z])", re.IGNORECASE), "エーオー"),
)
FIXED_SPEECH_NORMALIZATIONS: dict[str, str] = {
    "皆さん、こんにちは。": "皆さん、こんにちは",
    "ありがとうございました。": "ありがとうございました",
}
SENTENCE_END_CHARS = "。.!?！？"
SOFT_BREAK_CHARS = "、，,;； "
_FULL_WIDTH_NUMBER_TRANSLATION = str.maketrans("０１２３４５６７８９，．％", "0123456789,.%")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_SPEECH_HEADING_RE = re.compile(r"(^|\s+)\*\*([^*\n]{1,40})\*\*(?=\s*(?:[*+]|$))")
_PHONE_NUMBER_RE = re.compile(r"(?<!\d)(0\d{1,4})[-‐‑–—](\d{1,4})[-‐‑–—](\d{3,4})(?!\d)")
_SLASH_FULL_DATE_RE = re.compile(r"(?<![\d/.])(\d{4})/(\d{1,2})/(\d{1,2})(?![\d/])")
_SLASH_MONTH_DAY_RE = re.compile(r"(?<![A-Za-z\d/.:])(\d{1,2})/(\d{1,2})(?![\d/])")
_FULL_DATE_RE = re.compile(r"(?<![A-Za-z0-9])(\d{4})年(\d{1,2})月(\d{1,2})日")
_MONTH_DAY_RE = re.compile(r"(?<![A-Za-z0-9])(\d{1,2})月(\d{1,2})日")
_DAY_SPAN_RE = re.compile(r"(?<![A-Za-z0-9])(\d{1,3})日(間|目)")
_TIME_RANGE_RE = re.compile(r"(?<!\d)(\d{1,2}):(\d{2})\s*[~〜～−-]\s*(\d{1,2}):(\d{2})(?!\d)")
_OPEN_TIME_RE = re.compile(r"(?<!\d)(\d{1,2}):(\d{2})\s*[~〜～](?!\d)")
_CLOCK_TIME_RE = re.compile(r"(?<!\d)(\d{1,2}):(\d{2})(?!\d)")
_WEEKDAY_PAREN_RE = re.compile(r"[(（]([月火水木金土日])[)）]")
_ACADEMIC_YEAR_RE = re.compile(r"(?<![A-Za-z0-9])(\d{4})年度")
_YEAR_ORDINAL_RE = re.compile(r"(?<![A-Za-z0-9])(\d+)年目")
_YEAR_DURATION_RE = re.compile(r"(?<![A-Za-z0-9])(\d+)年間")
_YEAR_RE = re.compile(r"(?<![A-Za-z0-9,.])(\d{1,4})年")
_MONTH_RE = re.compile(r"(?<![A-Za-z0-9])(\d{1,2})月")
_DAY_RE = re.compile(r"(?<![A-Za-z0-9])(\d{1,2})日")
_YEN_SIGN_RE = re.compile(r"[¥￥]\s*(\d[\d,]*)(?![\d.])")
_YEN_RE = re.compile(r"(?<![A-Za-z0-9])(\d[\d,]*)\s*円")
_PERCENT_RE = re.compile(r"(?<![A-Za-z0-9])(\d[\d,]*(?:\.\d+)?)\s*(?:%|パーセント)")
_TECH_UNIT_RE = re.compile(
    r"(?<![A-Za-z0-9])(\d[\d,]*(?:\.\d+)?)\s*"
    r"(EFLOPS|PFLOPS|TFLOPS|GFLOPS|TB|GB|MB|KB|GHz|MHz|Gbps|Mbps|cores?)(?![A-Za-z0-9])",
    re.IGNORECASE,
)
# Case-sensitive so that e.g. "5G" or "3D" are not mistaken for grams or other units.
_METRIC_UNIT_RE = re.compile(
    r"(?<![A-Za-z0-9])(\d[\d,]*(?:\.\d+)?)\s*"
    r"(km²|km2|m²|m2|㎡|km|cm|mm|kg|mg|mL|ml|m|g|L|ha|℃|°C)(?![A-Za-z0-9])"
)
_DOTTED_VERSION_RE = re.compile(r"(?<![A-Za-z0-9])(\d+(?:\.\d+){2,})(?![A-Za-z0-9])")
_DECIMAL_RE = re.compile(r"(?<![A-Za-z0-9])(\d[\d,]*\.\d+)(?![A-Za-z0-9])")
_FRACTION_RE = re.compile(r"(?<![A-Za-z0-9])(\d[\d,]*)分の(\d[\d,]*)(?![\d.])")
_LARGE_KANJI_UNIT_RE = re.compile(r"(?<![A-Za-z0-9.])(\d[\d,]*)(?=[万億兆])")
_COUNTERS = (
    r"科目|教科|時間|週間|周年|[かヶケカヵ箇](?:月|所)|歳|才|位|基|期|級|台|枚|個|本|冊|組|倍|匹|"
    r"名|人|件|校|回|階|点|泊|杯|週|分(?![野析類担散布割岐解])|秒|時|つ"
)
_COUNTER_RE = re.compile(rf"(?<![A-Za-z0-9])(\d[\d,]*)\s*({_COUNTERS})")
# "2〜3人" -> "2人から3人" so that each side gets its own counter reading.
_NUMBER_RANGE_RE = re.compile(
    rf"(?<![A-Za-z0-9.,:/])(\d[\d,]*)\s*[〜~～]\s*(\d[\d,]*)"
    rf"(?=\s*({_COUNTERS}|年|月|日|円|割|%|パーセント|万|億))"
)
_STANDALONE_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])(\d[\d,]*)(?![A-Za-z0-9])")
_DIGIT_READINGS = ("ぜろ", "いち", "に", "さん", "よん", "ご", "ろく", "なな", "はち", "きゅう")
_LARGE_NUMBER_UNITS = ("", "まん", "おく", "ちょう", "けい")
_TECH_UNIT_READINGS = {
    "eflops": "エクサフロップス",
    "pflops": "ペタフロップス",
    "tflops": "テラフロップス",
    "gflops": "ギガフロップス",
    "tb": "テラバイト",
    "gb": "ギガバイト",
    "mb": "メガバイト",
    "kb": "キロバイト",
    "ghz": "ギガヘルツ",
    "mhz": "メガヘルツ",
    "gbps": "ギガビット毎秒",
    "mbps": "メガビット毎秒",
    "core": "コア",
    "cores": "コア",
}
_METRIC_UNIT_READINGS = {
    "km²": "へいほうキロメートル",
    "km2": "へいほうキロメートル",
    "m²": "へいほうメートル",
    "m2": "へいほうメートル",
    "㎡": "へいほうメートル",
    "km": "キロメートル",
    "cm": "センチメートル",
    "mm": "ミリメートル",
    "kg": "キログラム",
    "mg": "ミリグラム",
    "mL": "ミリリットル",
    "ml": "ミリリットル",
    "m": "メートル",
    "g": "グラム",
    "L": "リットル",
    "ha": "ヘクタール",
    "℃": "ど",
    "°C": "ど",
}
# Final digits whose reading becomes a geminate (いっ/ろっ/はっ/じゅっ) before the counter.
_K_ROW_SOKUON_DIGITS = frozenset({0, 1, 6, 8})
_S_ROW_SOKUON_DIGITS = frozenset({0, 1, 8})
_P_ROW_DIGITS = frozenset({0, 1, 3, 4, 6, 8})
_SOKUON_COUNTERS: dict[str, tuple[str, frozenset[int]]] = {
    "回": ("かい", _K_ROW_SOKUON_DIGITS),
    "個": ("こ", _K_ROW_SOKUON_DIGITS),
    "件": ("けん", _K_ROW_SOKUON_DIGITS),
    "校": ("こう", _K_ROW_SOKUON_DIGITS),
    "基": ("き", _K_ROW_SOKUON_DIGITS),
    "期": ("き", _K_ROW_SOKUON_DIGITS),
    "級": ("きゅう", _K_ROW_SOKUON_DIGITS),
    "か月": ("かげつ", _K_ROW_SOKUON_DIGITS),
    "か所": ("かしょ", _K_ROW_SOKUON_DIGITS),
    "歳": ("さい", _S_ROW_SOKUON_DIGITS),
    "才": ("さい", _S_ROW_SOKUON_DIGITS),
    "冊": ("さつ", _S_ROW_SOKUON_DIGITS),
    "週": ("しゅう", _S_ROW_SOKUON_DIGITS),
    "週間": ("しゅうかん", _S_ROW_SOKUON_DIGITS),
    "周年": ("しゅうねん", _S_ROW_SOKUON_DIGITS),
    "点": ("てん", _S_ROW_SOKUON_DIGITS),
}
# counter: (plain, geminated p-sound, b-sound after 3)
_H_ROW_COUNTERS = {
    "本": ("ほん", "ぽん", "ぼん"),
    "匹": ("ひき", "ぴき", "びき"),
    "杯": ("はい", "ぱい", "ばい"),
}
_PLAIN_COUNTER_SUFFIXES = {
    "位": "い",
    "台": "だい",
    "枚": "まい",
    "倍": "ばい",
    "名": "めい",
    "科目": "かもく",
    "教科": "きょうか",
    "秒": "びょう",
}
_TSU_READINGS = {
    1: "ひとつ",
    2: "ふたつ",
    3: "みっつ",
    4: "よっつ",
    5: "いつつ",
    6: "むっつ",
    7: "ななつ",
    8: "やっつ",
    9: "ここのつ",
    10: "とお",
}


def _read_under_ten_thousand(value: int) -> str:
    if not 0 <= value < 10_000:
        raise ValueError("value must be between 0 and 9999")
    if value == 0:
        return ""

    thousands, remainder = divmod(value, 1000)
    hundreds, remainder = divmod(remainder, 100)
    tens, ones = divmod(remainder, 10)
    parts: list[str] = []

    if thousands:
        if thousands == 1:
            parts.append("せん")
        elif thousands == 3:
            parts.append("さんぜん")
        elif thousands == 8:
            parts.append("はっせん")
        else:
            parts.append(f"{_DIGIT_READINGS[thousands]}せん")
    if hundreds:
        if hundreds == 1:
            parts.append("ひゃく")
        elif hundreds == 3:
            parts.append("さんびゃく")
        elif hundreds == 6:
            parts.append("ろっぴゃく")
        elif hundreds == 8:
            parts.append("はっぴゃく")
        else:
            parts.append(f"{_DIGIT_READINGS[hundreds]}ひゃく")
    if tens:
        parts.append("じゅう" if tens == 1 else f"{_DIGIT_READINGS[tens]}じゅう")
    if ones:
        parts.append(_DIGIT_READINGS[ones])
    return "".join(parts)


def _read_japanese_integer(raw_value: str, *, before_large_unit: bool = False) -> str:
    """Read an integer; ``before_large_unit`` marks a number followed by 万/億/兆."""
    digits = raw_value.replace(",", "")
    if not digits or not digits.isdigit():
        return raw_value
    if len(digits) > 1 and digits.startswith("0"):
        return "".join(_DIGIT_READINGS[int(digit)] for digit in digits)

    value = int(digits)
    if value == 0:
        return _DIGIT_READINGS[0]

    groups: list[int] = []
    while value:
        value, group = divmod(value, 10_000)
        groups.append(group)
    if len(groups) > len(_LARGE_NUMBER_UNITS):
        return "".join(_DIGIT_READINGS[int(digit)] for digit in digits)

    parts: list[str] = []
    for group_index in range(len(groups) - 1, -1, -1):
        group = groups[group_index]
        if not group:
            continue
        reading = _read_under_ten_thousand(group)
        # 1000万 is "いっせんまん", while a bare 1000 stays "せん".
        if (group_index > 0 or before_large_unit) and reading.startswith("せん"):
            reading = "いっ" + reading
        parts.append(reading)
        parts.append(_LARGE_NUMBER_UNITS[group_index])
    return "".join(parts)


def _read_year_value(raw_value: str) -> str:
    reading = _read_japanese_integer(raw_value)
    return reading[:-2] + "よ" if reading.endswith("よん") else reading


def _read_digits(raw_value: str) -> str:
    return "".join(_DIGIT_READINGS[int(digit)] for digit in raw_value)


def _read_decimal_value(raw_value: str) -> str:
    integer, separator, fraction = raw_value.replace(",", "").partition(".")
    if not separator:
        return _read_japanese_integer(integer)
    reading = "れい" if integer == "0" else _read_japanese_integer(integer)
    return f"{reading}てん{_read_digits(fraction)}"


def _read_dotted_version(raw_value: str) -> str:
    head, *tail = raw_value.split(".")
    parts = [_read_japanese_integer(head)]
    for component in tail:
        parts.extend(("てん", _read_digits(component)))
    return "".join(parts)


def _read_month_value(raw_value: str) -> str:
    value = int(raw_value)
    special = {4: "しがつ", 7: "しちがつ", 9: "くがつ"}
    return special.get(value, f"{_read_japanese_integer(raw_value)}がつ")


def _read_day_value(raw_value: str, *, calendar: bool = True) -> str:
    """Read N日; only 1日 differs between a date (ついたち) and a span (いちにち)."""
    value = int(raw_value)
    special = {
        1: "ついたち" if calendar else "いちにち",
        2: "ふつか",
        3: "みっか",
        4: "よっか",
        5: "いつか",
        6: "むいか",
        7: "なのか",
        8: "ようか",
        9: "ここのか",
        10: "とおか",
        14: "じゅうよっか",
        20: "はつか",
        24: "にじゅうよっか",
    }
    return special.get(value, f"{_read_japanese_integer(raw_value)}にち")


def _read_p_counter(raw_value: str, suffix: str, *, geminate_hundreds: bool = True) -> str:
    """Geminate the final digit (いっ/ろっ/はっ/じゅっ) before a counter.

    百 only geminates before k/h-row counters (ひゃっこ, ひゃっぽん), not before
    s/t-row ones (ひゃくさい, ろっぴゃくてん), so callers opt out for those.
    """
    value = int(raw_value.replace(",", ""))
    reading = _read_japanese_integer(raw_value)
    final_digit = value % 10
    if final_digit == 0:
        terminal_replacements = {"じゅう": "じゅっ"}
        if geminate_hundreds:
            terminal_replacements.update({"ひゃく": "ひゃっ", "びゃく": "びゃっ", "ぴゃく": "ぴゃっ"})
        for ending, replacement in terminal_replacements.items():
            if reading.endswith(ending):
                reading = reading[: -len(ending)] + replacement
                break
    elif final_digit == 1 and reading.endswith("いち"):
        reading = reading[:-2] + "いっ"
    elif final_digit == 6 and reading.endswith("ろく"):
        reading = reading[:-2] + "ろっ"
    elif final_digit == 8 and reading.endswith("はち"):
        reading = reading[:-2] + "はっ"
    return f"{reading}{suffix}"


def _read_hour(raw_value: str, *, seven: str = "しち") -> str:
    """Read N時: 4/7/9 in the ones place become よ/しち/く (14時 -> じゅうよじ)."""
    value = int(raw_value.replace(",", ""))
    if value == 0:
        return "れいじ"
    reading = _read_japanese_integer(raw_value)
    final_digit = value % 10
    if final_digit == 4 and reading.endswith("よん"):
        return f"{reading[:-2]}よじ"
    if final_digit == 7 and reading.endswith("なな"):
        return f"{reading[:-2]}{seven}じ"
    if final_digit == 9 and reading.endswith("きゅう"):
        return f"{reading[:-3]}くじ"
    return f"{reading}じ"


def _read_counter(raw_value: str, counter: str) -> str:
    value = int(raw_value.replace(",", ""))
    reading = _read_japanese_integer(raw_value)
    final_digit = value % 10
    if counter[0] in "かヶケカヵ箇" and counter[1:] in {"月", "所"}:
        counter = "か" + counter[1:]

    if counter == "時":
        return _read_hour(raw_value)
    if counter == "時間":
        return f"{_read_hour(raw_value, seven='なな')}かん"
    if counter == "人":
        if value in {1, 2}:
            return {1: "ひとり", 2: "ふたり"}[value]
        if final_digit == 4 and reading.endswith("よん"):
            return f"{reading[:-2]}よにん"
        return f"{reading}にん"
    if counter == "つ":
        return _TSU_READINGS.get(value, f"{reading}つ")
    if counter in {"歳", "才"} and value == 20:
        return "はたち"
    if counter == "組" and value in {1, 2}:
        return {1: "ひとくみ", 2: "ふたくみ"}[value]
    if counter == "組":
        return f"{reading}くみ"
    if counter == "階" and final_digit == 3 and reading.endswith("さん"):
        return f"{reading}がい"
    if counter == "階":
        counter = "回"
    if counter in {"分", "泊"}:
        plain, geminated = {"分": ("ふん", "ぷん"), "泊": ("はく", "ぱく")}[counter]
        if final_digit in _P_ROW_DIGITS:
            return _read_p_counter(raw_value, geminated)
        return f"{reading}{plain}"
    if counter in _H_ROW_COUNTERS:
        plain, geminated, voiced = _H_ROW_COUNTERS[counter]
        if final_digit == 3 and reading.endswith("さん"):
            return f"{reading}{voiced}"
        if final_digit in _K_ROW_SOKUON_DIGITS:
            return _read_p_counter(raw_value, geminated)
        return f"{reading}{plain}"
    if counter in _SOKUON_COUNTERS:
        suffix, sokuon_digits = _SOKUON_COUNTERS[counter]
        if final_digit in sokuon_digits:
            return _read_p_counter(
                raw_value,
                suffix,
                geminate_hundreds=sokuon_digits is _K_ROW_SOKUON_DIGITS,
            )
        return f"{reading}{suffix}"
    return f"{reading}{_PLAIN_COUNTER_SUFFIXES[counter]}"


def _read_clock_time(raw_hour: str, raw_minute: str) -> str:
    hour = _read_hour(raw_hour)
    if int(raw_minute) == 0:
        return hour
    return f"{hour}{_read_counter(raw_minute, '分')}"


def _read_slash_date(raw_year: str | None, raw_month: str, raw_day: str) -> str | None:
    month, day = int(raw_month), int(raw_day)
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    year = "" if raw_year is None else f"{_read_year_value(raw_year)}ねん"
    return f"{year}{_read_month_value(raw_month)}{_read_day_value(raw_day)}"


def _strip_speech_markdown(text: str) -> str:
    text = _MARKDOWN_LINK_RE.sub(r"\1", text)
    text = _SPEECH_HEADING_RE.sub(
        lambda match: f"{'。' if match.start() else ''}{match.group(2)}。",
        text,
    )
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)
    text = re.sub(r"(?m)^\s*[-*+]\s*", "", text)
    text = re.sub(r"(?:\*\*|__|~~|`)", "", text)
    text = re.sub(r"\s+[*+]\s*", "。", text)
    text = re.sub(r"[\r\n]+", "。", text)
    text = text.replace("•", "。").replace("◦", "。").replace("*", "。")
    text = re.sub(r"。(?:\s*。)+", "。", text)
    text = re.sub(r"\s*。\s*", "。", text)
    return text.strip(" 。") + ("。" if text.strip(" 。") and text.rstrip().endswith("。") else "")


def _normalize_japanese_numbers(text: str) -> str:
    normalized = text.translate(_FULL_WIDTH_NUMBER_TRANSLATION)
    normalized = _WEEKDAY_PAREN_RE.sub(lambda match: f"、{match.group(1)}曜日、", normalized)
    normalized = _PHONE_NUMBER_RE.sub(
        lambda match: "の".join(_read_digits(group) for group in match.groups()),
        normalized,
    )
    normalized = _SLASH_FULL_DATE_RE.sub(
        lambda match: _read_slash_date(*match.groups()) or match.group(0),
        normalized,
    )
    normalized = _SLASH_MONTH_DAY_RE.sub(
        lambda match: _read_slash_date(None, *match.groups()) or match.group(0),
        normalized,
    )
    normalized = _YEN_SIGN_RE.sub(lambda match: f"{_read_year_value(match.group(1))}えん", normalized)
    normalized = _TIME_RANGE_RE.sub(
        lambda match: (
            f"{_read_clock_time(match.group(1), match.group(2))}から"
            f"{_read_clock_time(match.group(3), match.group(4))}まで"
        ),
        normalized,
    )
    normalized = _OPEN_TIME_RE.sub(
        lambda match: f"{_read_clock_time(match.group(1), match.group(2))}から",
        normalized,
    )
    normalized = _CLOCK_TIME_RE.sub(
        lambda match: _read_clock_time(match.group(1), match.group(2)),
        normalized,
    )
    normalized = _NUMBER_RANGE_RE.sub(
        lambda match: f"{match.group(1)}{match.group(3)}から{match.group(2)}",
        normalized,
    )
    normalized = _FULL_DATE_RE.sub(
        lambda match: (
            f"{_read_year_value(match.group(1))}ねん"
            f"{_read_month_value(match.group(2))}"
            f"{_read_day_value(match.group(3))}"
        ),
        normalized,
    )
    normalized = _MONTH_DAY_RE.sub(
        lambda match: f"{_read_month_value(match.group(1))}{_read_day_value(match.group(2))}",
        normalized,
    )
    normalized = _DAY_SPAN_RE.sub(
        lambda match: (
            f"{_read_day_value(match.group(1), calendar=False)}"
            f"{'かん' if match.group(2) == '間' else 'め'}"
        ),
        normalized,
    )
    normalized = _ACADEMIC_YEAR_RE.sub(
        lambda match: f"{_read_year_value(match.group(1))}ねんど",
        normalized,
    )
    normalized = _YEAR_ORDINAL_RE.sub(
        lambda match: f"{_read_year_value(match.group(1))}ねんめ",
        normalized,
    )
    normalized = _YEAR_DURATION_RE.sub(
        lambda match: f"{_read_year_value(match.group(1))}ねんかん",
        normalized,
    )
    normalized = _YEAR_RE.sub(
        lambda match: f"{_read_year_value(match.group(1))}ねん",
        normalized,
    )
    normalized = _MONTH_RE.sub(lambda match: _read_month_value(match.group(1)), normalized)
    normalized = _DAY_RE.sub(
        lambda match: _read_day_value(
            match.group(1),
            # 毎月1日 is a date; a bare 1日 (1日に2回) is a span.
            calendar=match.string[max(0, match.start() - 1) : match.start()] == "月",
        ),
        normalized,
    )
    normalized = _YEN_RE.sub(
        lambda match: f"{_read_year_value(match.group(1))}えん",
        normalized,
    )
    normalized = _TECH_UNIT_RE.sub(
        lambda match: f"{_read_decimal_value(match.group(1))}{_TECH_UNIT_READINGS[match.group(2).lower()]}",
        normalized,
    )
    normalized = _METRIC_UNIT_RE.sub(
        lambda match: f"{_read_decimal_value(match.group(1))}{_METRIC_UNIT_READINGS[match.group(2)]}",
        normalized,
    )
    normalized = _PERCENT_RE.sub(
        lambda match: f"{_read_decimal_value(match.group(1))}ぱーせんと",
        normalized,
    )
    normalized = _DOTTED_VERSION_RE.sub(
        lambda match: _read_dotted_version(match.group(1)),
        normalized,
    )
    normalized = _DECIMAL_RE.sub(lambda match: _read_decimal_value(match.group(1)), normalized)
    normalized = _FRACTION_RE.sub(
        lambda match: (
            f"{_read_japanese_integer(match.group(1))}ぶんの{_read_japanese_integer(match.group(2))}"
        ),
        normalized,
    )
    normalized = _LARGE_KANJI_UNIT_RE.sub(
        lambda match: _read_japanese_integer(match.group(1), before_large_unit=True),
        normalized,
    )
    normalized = _COUNTER_RE.sub(
        lambda match: _read_counter(match.group(1), match.group(2)),
        normalized,
    )
    normalized = _STANDALONE_NUMBER_RE.sub(
        lambda match: _read_japanese_integer(match.group(1)),
        normalized,
    )
    normalized = re.sub(r"、(?:\s*、)+", "、", normalized)
    normalized = re.sub(r"、\s*。", "。", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _naturalize_speech_punctuation(text: str) -> str:
    text = re.sub(r"([A-Za-zぁ-ゖァ-ヶ一-鿿]+)[：:]", r"\1は、", text)
    text = re.sub(r"[(（]([^()（）]{1,40})[)）]", r"、\1、", text)
    text = text.replace("・", "、").replace("~", "から").replace("〜", "から").replace("～", "から")
    text = text.replace("：", "、").replace(":", "、")
    text = re.sub(r"、(?:\s*、)+", "、", text)
    text = re.sub(r"、\s*。", "。", text)
    return text.strip()


def normalize_speech_input(text: str) -> str:
    normalized = _strip_speech_markdown(text.strip())
    normalized = _normalize_japanese_numbers(normalized)
    for pattern, replacement in PRONUNCIATION_HINTS:
        normalized = pattern.sub(replacement, normalized)
    normalized = _naturalize_speech_punctuation(normalized)
    return FIXED_SPEECH_NORMALIZATIONS.get(normalized, normalized)


_SMALL_KANA = frozenset("ゃゅょぁぃぅぇぉゎャュョァィゥェォヮ")
_PAUSE_CHARS = frozenset("、，,。．！？!?")


def estimate_speech_mora(text: str) -> float:
    """Rough spoken length in mora.

    Kana count 1, kanji 1.7 (typical on'yomi), ASCII letters/digits 1.5, and a
    pause at punctuation inside the text 2.
    """
    mora = 0.0
    last_index = len(text.rstrip()) - 1
    for index, char in enumerate(text):
        if char in _SMALL_KANA:
            continue
        if "ぁ" <= char <= "ゖ" or "ァ" <= char <= "ヺ" or char == "ー":
            mora += 1.0
        elif "一" <= char <= "鿿" or char == "々":
            mora += 1.7
        elif char.isascii() and char.isalnum():
            mora += 1.5
        elif char in _PAUSE_CHARS and index < last_index:
            mora += 2.0
    return mora


def split_progressive_speech_input(
    text: str,
    *,
    soft_max_chars: int = 32,
    hard_max_chars: int = 64,
    min_segment_chars: int = 8,
) -> list[str]:
    normalized = normalize_speech_input(text)
    if not normalized:
        return []

    segments: list[str] = []
    buffer = normalized
    while buffer:
        sentence_indexes = [buffer.find(char) for char in SENTENCE_END_CHARS]
        sentence_indexes = [index for index in sentence_indexes if index >= 0]
        if sentence_indexes:
            segment, buffer = _take_progressive_segment(buffer, min(sentence_indexes) + 1)
        elif len(buffer) >= soft_max_chars:
            soft_index = _last_progressive_soft_break_index(buffer, soft_max_chars, min_segment_chars)
            if soft_index is not None:
                segment, buffer = _take_progressive_segment(buffer, soft_index + 1)
            elif len(buffer) >= hard_max_chars:
                segment, buffer = _take_progressive_segment(buffer, hard_max_chars)
            else:
                break
        elif len(buffer) >= hard_max_chars:
            segment, buffer = _take_progressive_segment(buffer, hard_max_chars)
        else:
            break
        if segment and any(char not in SENTENCE_END_CHARS for char in segment):
            segments.append(segment)

    tail = buffer.strip()
    if tail:
        segments.append(tail)
    return segments


def coalesce_progressive_segments(segments: list[str], *, max_segments: int) -> list[str]:
    if max_segments <= 0 or len(segments) <= max_segments:
        return segments
    head_count = max(1, max_segments - 1)
    return segments[:head_count] + ["".join(segments[head_count:])]


def _last_progressive_soft_break_index(
    text: str,
    max_chars: int,
    min_segment_chars: int,
) -> int | None:
    search_text = text[:max_chars]
    indexes = [search_text.rfind(char) for char in SOFT_BREAK_CHARS]
    indexes = [index for index in indexes if index + 1 >= min_segment_chars]
    return max(indexes) if indexes else None


def _take_progressive_segment(text: str, end_index: int) -> tuple[str, str]:
    return text[:end_index].strip(), text[end_index:].lstrip()


def _env_bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    raw = env.get(key)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _env_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key, "").strip()
    return int(raw) if raw else default


def _env_optional_int(env: Mapping[str, str], key: str) -> int | None:
    raw = env.get(key, "").strip()
    return int(raw) if raw else None


def _env_float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key, "").strip()
    return float(raw) if raw else default


def _env_optional_float(env: Mapping[str, str], key: str) -> float | None:
    raw = env.get(key, "").strip()
    return float(raw) if raw else None


def _env_path(env: Mapping[str, str], key: str, default: Path) -> Path:
    raw = env.get(key, "").strip()
    return Path(raw).expanduser() if raw else default


def _env_texts(env: Mapping[str, str], key: str) -> tuple[str, ...]:
    raw = env.get(key, "")
    return tuple(text.strip() for text in raw.split("|") if text.strip())


@dataclass(frozen=True)
class IrodoriSettings:
    host: str = "0.0.0.0"
    port: int = 8021
    data_root: Path = DEFAULT_DATA_ROOT
    hf_checkpoint: str = DEFAULT_HF_CHECKPOINT
    checkpoint: Path | None = None
    lora_adapter: Path | None = None
    reference_source: Path = DEFAULT_REFERENCE_SOURCE
    reference_wav: Path = DEFAULT_REFERENCE_WAV
    fixed_open_campus_greeting_pcm: Path = DEFAULT_FIXED_OPEN_CAMPUS_GREETING_PCM
    voice: str = DEFAULT_VOICE
    caption: str | None = None
    model_device: str = "cuda:0"
    model_precision: str = "bf16"
    codec_device: str = "cuda:0"
    codec_precision: str = "fp32"
    codec_repo: str = "Aratako/Semantic-DACVAE-Japanese-32dim"
    # None lets the runtime pick the checkpoint default (RF: 40 steps, MeanFlow: 4 steps).
    num_steps: int | None = None
    duration_scale: float = 1.0
    # None uses the checkpoint's recommended reference length (30 s for legacy checkpoints).
    max_ref_seconds: float | None = None
    response_sample_rate_hz: int = 16000
    preload_model: bool = False
    short_cache_enabled: bool = False
    short_cache_max_chars: int = 40
    short_cache_max_entries: int = 128
    short_cache_prewarm_texts: tuple[str, ...] = ()
    short_cache_prewarm_num_steps: int | None = None
    stream_chunk_bytes: int = 3200
    progressive_stream_enabled: bool = False
    progressive_max_segments: int = 2
    short_duration_max_chars: int = 16
    short_duration_scale: float = 0.8
    # v4 checkpoints overestimate the length of very short phrases and fill the extra
    # time with babble ("はい。" -> 3 s of speech), so cap the predicted duration at
    # base + estimated mora / rate. A rate <= 0 disables the cap.
    duration_cap_base_seconds: float = 0.7
    duration_cap_mora_per_second: float = 6.0

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "IrodoriSettings":
        values = os.environ if env is None else env
        checkpoint_raw = values.get("IRODORI_TTS_CHECKPOINT", "").strip()
        lora_raw = values.get("IRODORI_TTS_LORA_ADAPTER", "").strip()
        caption_raw = values.get("IRODORI_TTS_CAPTION", "").strip()
        return cls(
            host=values.get("IRODORI_TTS_HOST", "0.0.0.0").strip() or "0.0.0.0",
            port=_env_int(values, "IRODORI_TTS_PORT", 8021),
            data_root=_env_path(values, "IRODORI_TTS_DATA_ROOT", DEFAULT_DATA_ROOT),
            hf_checkpoint=values.get("IRODORI_TTS_HF_CHECKPOINT", DEFAULT_HF_CHECKPOINT).strip()
            or DEFAULT_HF_CHECKPOINT,
            checkpoint=Path(checkpoint_raw).expanduser() if checkpoint_raw else None,
            lora_adapter=Path(lora_raw).expanduser() if lora_raw else None,
            reference_source=_env_path(values, "IRODORI_TTS_REFERENCE_SOURCE", DEFAULT_REFERENCE_SOURCE),
            reference_wav=_env_path(values, "IRODORI_TTS_REFERENCE_WAV", DEFAULT_REFERENCE_WAV),
            fixed_open_campus_greeting_pcm=_env_path(
                values,
                "IRODORI_TTS_FIXED_OPEN_CAMPUS_GREETING_PCM",
                DEFAULT_FIXED_OPEN_CAMPUS_GREETING_PCM,
            ),
            voice=values.get("IRODORI_TTS_VOICE", DEFAULT_VOICE).strip() or DEFAULT_VOICE,
            caption=caption_raw or None,
            model_device=values.get("IRODORI_TTS_MODEL_DEVICE", "cuda:0").strip() or "cuda:0",
            model_precision=values.get("IRODORI_TTS_MODEL_PRECISION", "bf16").strip() or "bf16",
            codec_device=values.get("IRODORI_TTS_CODEC_DEVICE", "cuda:0").strip() or "cuda:0",
            codec_precision=values.get("IRODORI_TTS_CODEC_PRECISION", "fp32").strip() or "fp32",
            codec_repo=values.get(
                "IRODORI_TTS_CODEC_REPO",
                "Aratako/Semantic-DACVAE-Japanese-32dim",
            ).strip()
            or "Aratako/Semantic-DACVAE-Japanese-32dim",
            num_steps=_env_optional_int(values, "IRODORI_TTS_NUM_STEPS"),
            duration_scale=_env_float(values, "IRODORI_TTS_DURATION_SCALE", 1.0),
            short_duration_max_chars=_env_int(
                values,
                "IRODORI_TTS_SHORT_DURATION_MAX_CHARS",
                16,
            ),
            short_duration_scale=_env_float(values, "IRODORI_TTS_SHORT_DURATION_SCALE", 0.8),
            duration_cap_base_seconds=_env_float(values, "IRODORI_TTS_DURATION_CAP_BASE_SECONDS", 0.7),
            duration_cap_mora_per_second=_env_float(
                values,
                "IRODORI_TTS_DURATION_CAP_MORA_PER_SECOND",
                6.0,
            ),
            max_ref_seconds=_env_optional_float(values, "IRODORI_TTS_MAX_REF_SECONDS"),
            response_sample_rate_hz=_env_int(values, "IRODORI_TTS_RESPONSE_SAMPLE_RATE_HZ", 16000),
            preload_model=_env_bool(values, "IRODORI_TTS_PRELOAD_MODEL", False),
            short_cache_enabled=_env_bool(values, "IRODORI_TTS_SHORT_CACHE_ENABLED", False),
            short_cache_max_chars=_env_int(values, "IRODORI_TTS_SHORT_CACHE_MAX_CHARS", 40),
            short_cache_max_entries=_env_int(values, "IRODORI_TTS_SHORT_CACHE_MAX_ENTRIES", 128),
            short_cache_prewarm_texts=_env_texts(values, "IRODORI_TTS_SHORT_CACHE_PREWARM_TEXTS"),
            short_cache_prewarm_num_steps=_env_optional_int(
                values,
                "IRODORI_TTS_SHORT_CACHE_PREWARM_NUM_STEPS",
            ),
            stream_chunk_bytes=_env_int(values, "IRODORI_TTS_STREAM_CHUNK_BYTES", 3200),
            progressive_stream_enabled=_env_bool(
                values,
                "IRODORI_TTS_PROGRESSIVE_STREAM_ENABLED",
                False,
            ),
            progressive_max_segments=_env_int(values, "IRODORI_TTS_PROGRESSIVE_MAX_SEGMENTS", 2),
        )

    def ensure_reference_wav(self) -> Path:
        if self.reference_wav.is_file():
            return self.reference_wav
        if not self.reference_source.is_file():
            raise FileNotFoundError(f"reference audio not found: {self.reference_source}")
        self.reference_wav.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(self.reference_source),
                "-ac",
                "1",
                "-ar",
                "48000",
                str(self.reference_wav),
            ],
            check=True,
        )
        return self.reference_wav


def _to_mono_float32(audio: object) -> np.ndarray:
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim == 2:
        if arr.shape[0] <= arr.shape[1]:
            arr = arr.mean(axis=0)
        else:
            arr = arr.mean(axis=1)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr.astype(np.float32, copy=False)


def _resample_linear(audio: np.ndarray, source_sample_rate: int, target_sample_rate: int) -> np.ndarray:
    if source_sample_rate == target_sample_rate or audio.size == 0:
        return audio
    target_len = max(1, int(round(audio.size * target_sample_rate / source_sample_rate)))
    if audio.size == 1:
        return np.repeat(audio, target_len)
    old_positions = np.linspace(0.0, float(audio.size - 1), num=audio.size, dtype=np.float64)
    new_positions = np.linspace(0.0, float(audio.size - 1), num=target_len, dtype=np.float64)
    return np.interp(new_positions, old_positions, audio).astype(np.float32)


def audio_to_pcm16(audio: object, *, source_sample_rate: int, target_sample_rate: int) -> bytes:
    mono = _to_mono_float32(audio)
    resampled = _resample_linear(mono, source_sample_rate, target_sample_rate)
    clipped = np.clip(resampled, -1.0, 1.0)
    int_audio = np.where(clipped >= 0, clipped * 32767.0, clipped * 32768.0).astype("<i2")
    return int_audio.tobytes()


def pcm16_to_wav_bytes(pcm: bytes, *, sample_rate_hz: int) -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate_hz)
        wav_file.writeframes(pcm)
    return buffer.getvalue()


def iter_pcm_chunks(pcm: bytes, *, chunk_bytes: int) -> Iterator[bytes]:
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    sample_aligned_chunk_bytes = max(2, chunk_bytes - (chunk_bytes % 2))
    for start in range(0, len(pcm), sample_aligned_chunk_bytes):
        yield pcm[start : start + sample_aligned_chunk_bytes]


class IrodoriSynthesizer:
    def __init__(self, settings: IrodoriSettings) -> None:
        self._settings = settings
        self._runtime = None
        self._short_cache: OrderedDict[tuple[str, str], tuple[bytes, str, int]] = OrderedDict()
        self._short_cache_lock = Lock()

    def load_runtime(self):
        if self._runtime is not None:
            return self._runtime

        from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, download_hf_checkpoint

        if self._settings.checkpoint is not None:
            checkpoint = str(self._settings.checkpoint)
        else:
            # v4 checkpoints need the bundled tokenizer/ next to model.safetensors, and
            # quantized variants are addressed as "repo_id/subfolder".
            checkpoint = download_hf_checkpoint(self._settings.hf_checkpoint)

        self._runtime = InferenceRuntime.from_key(
            RuntimeKey(
                checkpoint=checkpoint,
                model_device=self._settings.model_device,
                codec_repo=self._settings.codec_repo,
                model_precision=self._settings.model_precision,
                codec_device=self._settings.codec_device,
                codec_precision=self._settings.codec_precision,
            )
        )
        return self._runtime

    def _synthesize_normalized(
        self,
        normalized_text: str,
        *,
        response_format: str,
        num_steps: int | None = None,
    ) -> tuple[bytes, str, int]:
        from irodori_tts.inference_runtime import SamplingRequest

        ref_wav = self._settings.ensure_reference_wav()
        runtime = self.load_runtime()
        duration_scale = self._settings.duration_scale
        if (
            self._settings.short_duration_max_chars > 0
            and len(normalized_text) <= self._settings.short_duration_max_chars
        ):
            duration_scale = self._settings.short_duration_scale
        result = runtime.synthesize(
            SamplingRequest(
                text=normalized_text,
                caption=self._settings.caption,
                ref_wav=str(ref_wav),
                no_ref=False,
                num_steps=self._settings.num_steps if num_steps is None else num_steps,
                duration_scale=duration_scale,
                max_seconds=self._max_seconds_for(normalized_text),
                max_ref_seconds=self._settings.max_ref_seconds,
                lora_adapter=None
                if self._settings.lora_adapter is None
                else str(self._settings.lora_adapter),
            ),
            log_fn=None,
        )
        pcm = audio_to_pcm16(
            result.audio,
            source_sample_rate=int(result.sample_rate),
            target_sample_rate=self._settings.response_sample_rate_hz,
        )
        if response_format == "wav":
            return (
                pcm16_to_wav_bytes(pcm, sample_rate_hz=self._settings.response_sample_rate_hz),
                "audio/wav",
                self._settings.response_sample_rate_hz,
            )
        return pcm, "audio/L16", self._settings.response_sample_rate_hz

    def _max_seconds_for(self, normalized_text: str) -> float:
        rate = self._settings.duration_cap_mora_per_second
        if rate <= 0:
            return MAX_SYNTHESIS_SECONDS
        cap = self._settings.duration_cap_base_seconds + estimate_speech_mora(normalized_text) / rate
        return round(min(MAX_SYNTHESIS_SECONDS, max(MIN_CAPPED_SECONDS, cap)), 3)

    def _is_short_cacheable(self, normalized_text: str) -> bool:
        return (
            self._settings.short_cache_enabled
            and self._settings.short_cache_max_entries > 0
            and len(normalized_text) <= self._settings.short_cache_max_chars
        )

    def _get_short_cache(self, key: tuple[str, str]) -> tuple[bytes, str, int] | None:
        with self._short_cache_lock:
            cached = self._short_cache.get(key)
            if cached is None:
                return None
            self._short_cache.move_to_end(key)
            return cached

    def _put_short_cache(self, key: tuple[str, str], value: tuple[bytes, str, int]) -> None:
        with self._short_cache_lock:
            self._short_cache[key] = value
            self._short_cache.move_to_end(key)
            while len(self._short_cache) > self._settings.short_cache_max_entries:
                self._short_cache.popitem(last=False)

    def short_cache_size(self) -> int:
        with self._short_cache_lock:
            return len(self._short_cache)

    def prewarm_short_cache(self, texts: tuple[str, ...], *, num_steps: int | None = None) -> int:
        """Synthesize configured fixed phrases before the service accepts requests."""
        warmed = 0
        for text in texts:
            normalized_text = normalize_speech_input(text)
            if not normalized_text or not self._is_short_cacheable(normalized_text):
                continue
            self._synthesize_cached_normalized(
                normalized_text,
                response_format="pcm",
                num_steps=num_steps,
            )
            warmed += 1
        return warmed

    def _synthesize_cached_normalized(
        self,
        normalized_text: str,
        *,
        response_format: str,
        num_steps: int | None = None,
    ) -> tuple[bytes, str, int]:
        fixed_audio = self._fixed_audio_for_normalized(normalized_text, response_format=response_format)
        if fixed_audio is not None:
            return fixed_audio

        cache_key = (response_format, normalized_text)
        if self._is_short_cacheable(normalized_text):
            cached = self._get_short_cache(cache_key)
            if cached is not None:
                return cached

        result = self._synthesize_normalized(
            normalized_text,
            response_format=response_format,
            num_steps=num_steps,
        )
        if self._is_short_cacheable(normalized_text):
            self._put_short_cache(cache_key, result)
        return result

    def _fixed_audio_for_normalized(
        self,
        normalized_text: str,
        *,
        response_format: str,
    ) -> tuple[bytes, str, int] | None:
        if normalized_text != FIXED_OPEN_CAMPUS_GREETING_TEXT:
            return None

        fixed_path = self._settings.fixed_open_campus_greeting_pcm
        if not fixed_path.is_file():
            return None

        pcm = fixed_path.read_bytes()
        if not pcm or len(pcm) % 2:
            raise RuntimeError(f"invalid fixed greeting PCM: {fixed_path}")
        if response_format == "wav":
            return (
                pcm16_to_wav_bytes(pcm, sample_rate_hz=self._settings.response_sample_rate_hz),
                "audio/wav",
                self._settings.response_sample_rate_hz,
            )
        return pcm, "audio/L16", self._settings.response_sample_rate_hz

    def synthesize(self, text: str, *, response_format: str = "pcm") -> tuple[bytes, str, int]:
        normalized_text = normalize_speech_input(text)
        if not normalized_text:
            raise ValueError("input text must be non-empty")
        if response_format not in {"pcm", "wav"}:
            raise ValueError("response_format must be one of: pcm, wav")

        return self._synthesize_cached_normalized(normalized_text, response_format=response_format)

    def iter_pcm_stream(self, text: str, *, chunk_bytes: int) -> Iterator[bytes]:
        segments = coalesce_progressive_segments(
            split_progressive_speech_input(text),
            max_segments=self._settings.progressive_max_segments,
        )
        if not segments:
            raise ValueError("input text must be non-empty")

        for segment in segments:
            pcm, _media_type, _sample_rate = self._synthesize_cached_normalized(
                segment,
                response_format="pcm",
            )
            yield from iter_pcm_chunks(pcm, chunk_bytes=chunk_bytes)


def iter_progressive_pcm_response(
    synthesizer: object,
    text: str,
    *,
    chunk_bytes: int,
    lock: object | None = None,
) -> Iterator[bytes]:
    def _iter() -> Iterator[bytes]:
        yield from synthesizer.iter_pcm_stream(text, chunk_bytes=chunk_bytes)

    if lock is None:
        yield from _iter()
        return

    with lock:
        yield from _iter()


def iter_streaming_pcm_response(
    synthesizer: object,
    text: str,
    *,
    settings: IrodoriSettings,
    lock: object | None = None,
) -> Iterator[bytes]:
    def _iter() -> Iterator[bytes]:
        if settings.progressive_stream_enabled:
            yield from synthesizer.iter_pcm_stream(text, chunk_bytes=settings.stream_chunk_bytes)
            return
        pcm, _media_type, _sample_rate = synthesizer.synthesize(text, response_format="pcm")
        yield from iter_pcm_chunks(pcm, chunk_bytes=settings.stream_chunk_bytes)

    if lock is None:
        yield from _iter()
        return

    with lock:
        yield from _iter()


def create_app(settings: IrodoriSettings | None = None, synthesizer: IrodoriSynthesizer | None = None):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import Response, StreamingResponse
    from pydantic import BaseModel, Field

    resolved_settings = settings or IrodoriSettings.from_env()
    resolved_synthesizer = synthesizer or IrodoriSynthesizer(resolved_settings)

    class SpeechRequest(BaseModel):
        input: str = Field(min_length=1)
        voice: str = resolved_settings.voice
        response_format: str = "pcm"
        stream: bool = False

    app = FastAPI(title="Irodori TTS Service")
    app.state.settings = resolved_settings
    app.state.synthesizer = resolved_synthesizer
    app.state.synthesis_lock = None
    app.state.streaming_lock = Lock()

    @app.on_event("startup")
    async def _startup() -> None:
        app.state.synthesis_lock = asyncio.Lock()
        if resolved_settings.preload_model:
            await asyncio.to_thread(resolved_synthesizer.load_runtime)
        if resolved_settings.short_cache_prewarm_texts:
            await asyncio.to_thread(
                resolved_synthesizer.prewarm_short_cache,
                resolved_settings.short_cache_prewarm_texts,
                num_steps=resolved_settings.short_cache_prewarm_num_steps,
            )

    @app.get("/healthz")
    async def healthz() -> dict[str, object]:
        return {
            "status": "ok",
            "voice": resolved_settings.voice,
            "checkpoint": str(resolved_settings.checkpoint or resolved_settings.hf_checkpoint),
            "model_loaded": resolved_synthesizer._runtime is not None,
            "reference_source_exists": resolved_settings.reference_source.is_file(),
            "reference_wav_exists": resolved_settings.reference_wav.is_file(),
            "fixed_open_campus_greeting_ready": (
                resolved_settings.fixed_open_campus_greeting_pcm.is_file()
            ),
            "sample_rate_hz": resolved_settings.response_sample_rate_hz,
            "short_cache_enabled": resolved_settings.short_cache_enabled,
            "short_cache_entries": resolved_synthesizer.short_cache_size()
            if hasattr(resolved_synthesizer, "short_cache_size")
            else 0,
            "short_cache_max_chars": resolved_settings.short_cache_max_chars,
            "short_cache_prewarm_texts": len(resolved_settings.short_cache_prewarm_texts),
            "short_cache_prewarm_num_steps": resolved_settings.short_cache_prewarm_num_steps,
            "progressive_stream_enabled": resolved_settings.progressive_stream_enabled,
        }

    @app.post("/v1/audio/speech")
    async def speech(request: SpeechRequest) -> Response:
        if request.voice != resolved_settings.voice:
            raise HTTPException(status_code=400, detail=f"unsupported voice: {request.voice}")
        if request.response_format not in {"pcm", "wav"}:
            raise HTTPException(status_code=400, detail="response_format must be pcm or wav")
        if request.stream and request.response_format != "pcm":
            raise HTTPException(status_code=400, detail="stream=true requires response_format=pcm")
        if request.stream:
            headers = {
                "X-Sample-Rate-Hz": str(resolved_settings.response_sample_rate_hz),
                "X-Audio-Channels": "1",
                "X-Audio-Encoding": "pcm_s16le",
            }
            if resolved_settings.progressive_stream_enabled:
                return StreamingResponse(
                    iter_progressive_pcm_response(
                        app.state.synthesizer,
                        request.input,
                        chunk_bytes=resolved_settings.stream_chunk_bytes,
                        lock=app.state.streaming_lock,
                    ),
                    media_type="audio/L16",
                    headers=headers,
                )
            if app.state.synthesis_lock is None:
                app.state.synthesis_lock = asyncio.Lock()
            async with app.state.synthesis_lock:
                try:
                    content, _media_type, _sample_rate = await asyncio.to_thread(
                        app.state.synthesizer.synthesize,
                        request.input,
                        response_format="pcm",
                    )
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
                except Exception as exc:
                    raise HTTPException(status_code=500, detail=str(exc)) from exc
            return StreamingResponse(
                iter_pcm_chunks(content, chunk_bytes=resolved_settings.stream_chunk_bytes),
                media_type="audio/L16",
                headers=headers,
            )
        if app.state.synthesis_lock is None:
            app.state.synthesis_lock = asyncio.Lock()
        async with app.state.synthesis_lock:
            try:
                content, media_type, sample_rate = await asyncio.to_thread(
                    app.state.synthesizer.synthesize,
                    request.input,
                    response_format=request.response_format,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
        headers = {
            "X-Sample-Rate-Hz": str(sample_rate),
            "X-Audio-Channels": "1",
            "X-Audio-Encoding": "pcm_s16le" if request.response_format == "pcm" else "wav",
        }
        return Response(
            content=content,
            media_type=media_type,
            headers=headers,
        )

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Irodori-TTS HTTP service")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    import uvicorn

    settings = IrodoriSettings.from_env()
    host = args.host or settings.host
    port = args.port or settings.port
    uvicorn.run(create_app(settings), host=host, port=port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
