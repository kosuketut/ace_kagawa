#!/usr/bin/env python3
"""HTTP wrapper for Irodori-TTS inference.

The module is intentionally cheap to import. FastAPI, torch, and Irodori are
loaded only when the app or runtime is created, so repo-level tests can validate
configuration and audio conversion without loading model dependencies.
"""

import argparse
import asyncio
import base64
from collections import OrderedDict
from dataclasses import dataclass
from io import BytesIO
import json
import os
from pathlib import Path
import queue
import re
import subprocess
from threading import Lock, Thread
import time
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
BACKENDS = ("python", "speechcpp")
DEFAULT_SPEECHCPP_BIN = Path("/data/ACE/speech-cpp/runtime/v0.8.7/speech")
DEFAULT_SPEECHCPP_MODEL = Path("/data/ACE/speech-cpp/models/Irodori-TTS-859M-v4.1-F16.gguf")
DEFAULT_SPEECHCPP_VOICE = DEFAULT_DATA_ROOT / "speech-cpp" / "kagawa_voice_ref_48k_mono.voice.gguf"
# Irodori's SamplingRequest.max_seconds default.
MAX_SYNTHESIS_SECONDS = 30.0
MIN_CAPPED_SECONDS = 1.2
FIXED_OPEN_CAMPUS_GREETING_TEXT = "皆さん、こんにちは"
PRONUNCIATION_HINTS: tuple[tuple[re.Pattern[str], str], ...] = (
    # The right boundary keeps other names such as 香川豊子 intact.
    (re.compile(r"香川\s*豊(?![子島江美])"), "香川ゆたか"),
    (re.compile(r"(?<![A-Za-z])USJ(?![A-Za-z])", re.IGNORECASE), "ユーエスジェイ"),
    (re.compile(r"(?<![A-Za-z])Q&A(?![A-Za-z])", re.IGNORECASE), "キューアンドエー"),
    (
        re.compile(
            r"青嵐\s*[（(]\s*(?:読み\s*[：:]\s*)?(?:せいらん|セイラン|SEIRAN)"
            r"(?:\s*[,、・/]\s*(?:せいらん|セイラン|SEIRAN))*\s*[）)]",
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
# Full-width ASCII (U+FF01-U+FF5E) and squared units, so the regexes below see one form.
# Ellipses, circled numbers and Japanese punctuation are left to Irodori's own normalizer.
# ［］ stay full width so that ［重要］（…） is not mistaken for a Markdown link.
_FULL_WIDTH_ASCII_TRANSLATION = str.maketrans(
    {
        **{chr(code): chr(code - 0xFEE0) for code in range(0xFF01, 0xFF5F) if code not in (0xFF3B, 0xFF3D)},
        "㎞": "km",
        "㎏": "kg",
        "㎝": "cm",
        "㎜": "mm",
        "㎡": "m2",
        "㎥": "m3",
    }
)
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_URL_RE = re.compile(
    r"(?:https?://|(?<![A-Za-z0-9.@])www\.)[A-Za-z0-9\-._~:/?#@!$&'*+,;=%]*[A-Za-z0-9\-_~/#=&%]"
)
_URL_AS_NOUN_RE = re.compile(r"\s*(?:を|に|で|から|へ|の|は|が|です|や|と)")
_SPEECH_HEADING_RE = re.compile(r"(^|\s+)\*\*([^*\n]{1,40})\*\*(?=\s*(?:[*+]|$))")
# A hiragana reading right after a kanji word: 片倉町（かたくらまち）.
_FURIGANA_RE = re.compile(
    r"(?<![一-鿿々〆ヵヶ])([一-鿿々〆ヵヶ]+)[(（]\s*([ぁ-ゖ]+(?:[・\s][ぁ-ゖ]+)*)\s*[)）]"
)
# Hiragana glosses that are not readings: 受付（なし）, 対象（どなたでも）.
_NON_READING_GLOSSES = frozenset({"なし", "あり", "ふよう", "みてい", "こちら", "どなたでも", "とくになし"})
# 〜 after kana is a long vowel (よろしくね〜); after a number it still means から.
_WAVE_DASH_LONG_VOWEL_RE = re.compile(r"(?<=[ぁ-ゖァ-ヺ])[〜~](?=[。、!?」』\s]|$)")
_CJK_AMPERSAND_RE = re.compile(r"(?<=[一-鿿々ぁ-ゖァ-ヺー])\s*&\s*(?=[一-鿿々ぁ-ゖァ-ヺー])")
_MULTIPLICATION_RE = re.compile(r"(?<=[0-9A-Za-z])\s*×\s*(?=\d)")
# PDF-extracted dates such as "9 月 1 日" or "6/ 7 日".
_SPACED_DATE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?<![A-Za-z0-9.])(\d{1,4}) (?=[年月](?![額曜]))"), r"\1"),
    (re.compile(r"(?:(?<=[年月])|(?<=\d/)) (?=\d)"), ""),
    (re.compile(r"(?<=[月/])(\d{1,2}) (?=日)"), r"\1"),
)
# A comma after a digit that is not a thousands separator is a pause: 2,3人 -> 2、3人.
_NON_THOUSANDS_COMMA_RE = re.compile(r"(?<=\d),(?!\d{3}(?!\d))")
_WEEKDAY_CHAR = r"(?:[月火水木金土日]|祝(?!日))"
_WEEKDAY_PAREN_RE = re.compile(rf"[(（]({_WEEKDAY_CHAR}(?:[・･、]?{_WEEKDAY_CHAR}){{0,2}})[)）]")
_ISO_DATE_RE = re.compile(r"(?<![\d-])(\d{4})-(\d{1,2})-(\d{1,2})(?![\dT-])")
_POSTAL_CODE_RE = re.compile(r"〒\s*(\d{3})\s*[-‐‑–—−]\s*(\d{4})(?!\d)")
_PHONE_NUMBER_RE = re.compile(r"(?<!\d)(0\d{1,4})[-‐‑–—](\d{1,4})[-‐‑–—](\d{3,4})(?!\d)")
# Block and lot numbers after 町/丁目/番地: 片倉町1404-1 -> せんよんひゃくよんのいち.
_ADDRESS_NUMBER_RE = re.compile(
    r"(町|丁目|番地)\s*(\d+)\s*[-‐‑–—−]\s*(\d+)(?:\s*[-‐‑–—−]\s*(\d+))?(?![\d.])"
)
# Three block numbers right after a place name: 西蒲田5-23-22 -> ごのにじゅうさんのにじゅうに.
# Phone numbers (leading 0) and ISO dates are consumed before this runs.
_PLACE_BLOCK_NUMBER_RE = re.compile(
    r"(?<=[一-鿿々ヶ])\s*(\d{1,4})\s*[-‐‑–—−]\s*(\d{1,4})\s*[-‐‑–—−]\s*(\d{1,4})(?![\d.]|\s*[-‐‑–—−]\s*\d)"
)
_SLASH_FULL_DATE_RE = re.compile(r"(?<![\d/.])(\d{4})/(\d{1,2})/(\d{1,2})(?:日(?![曜本]))?(?![\d/])")
_SLASH_MONTH_DAY_RE = re.compile(
    r"(?<![A-Za-z\d/.])(?<!\d:)(\d{1,2})/(\d{1,2})(?:日(?![曜本]))?(?![\d/])"
)
_FULL_DATE_RE = re.compile(r"(?<![A-Za-z0-9])(\d{4})年(\d{1,2})月(\d{1,2})日")
_MONTH_DAY_RE = re.compile(r"(?<![A-Za-z0-9])(\d{1,2})月(\d{1,2})日")
_DAY_SPAN_RE = re.compile(r"(?<![A-Za-z0-9])(\d{1,3}(?:,\d{3})+|\d+)日(間|目)")
# A unit after "N:M" means a numbered label, not a time or ratio (質問1：10月, ポイント1：4年間).
# 時 is not in the set (10:30時点 is a time), nor are words such as 17:00月〜金 or 10:30日本時間.
_LABEL_UNIT = r"(?:[年人回件分%つ]|月(?![曜〜~～・])|日(?![本曜])|名(?!古)|校(?![門舎]))"
# A unit right after the minutes (1:10月) marks a label only when the minutes are
# not :00/:30; 9:00〜17:00年中無休 and 13:00分科会 are times.
_TIME_RANGE_RE = re.compile(
    rf"(?<!\d)(\d{{1,2}}):(\d{{2}})\s*[~〜～−‐–—-]\s*(\d{{1,2}}):(\d{{2}})"
    rf"(?!\d)(?:(?<=[03]0)|(?!{_LABEL_UNIT}))"
)
_OPEN_TIME_RE = re.compile(r"(?<!\d)(\d{1,2}):(\d{2})\s*[~〜～](?:\s*から)?(?!\d)")
# The last alternative keeps the label in その1：10〜12月, but not in a date range
# such as 10:00〜9月7日.
_CLOCK_TIME_RE = re.compile(
    rf"(?<!\d)(\d{{1,2}}):(\d{{2}})(?::(\d{{2}}))?"
    rf"(?!\d)(?:(?<=[03]0)|(?!{_LABEL_UNIT}|\s*[~〜～]\s*\d{{1,2}}\s*{_LABEL_UNIT}(?!\d)))"
)
# A numbered label such as 質問1： or ポイント1: is a pause, not a clock time.
# その counts only before a unit, so その1:00の回 stays a time.
_LABEL_COLON_RE = re.compile(
    r"((?:質問|ポイント|ステップ|項目|手順|課題|パート|問題|問|例"
    rf"|その(?=\d:\d+(?:\s*[~〜～]\s*\d+)?\s*{_LABEL_UNIT}))\s*)(\d{{1,2}})\s*:\s*(?=\d)"
)
_HOUR_MINUTE_RE = re.compile(r"(?<![A-Za-z0-9])(\d{1,2})時(\d{1,2})分(?![野析類担散布割岐解])")
# A one-digit right side is never a clock time: 16:9 -> じゅうろくたいきゅう.
_RATIO_RE = re.compile(rf"(?<![A-Za-z0-9:.])(\d+):(\d)(?![\d:]|{_LABEL_UNIT})")
_WEEKDAY_AFTER_DAY_RE = re.compile(r"、(?:[月火水木金土日]曜日|祝日)")
_CALENDAR_DAY_PREFIXES = ("月", "今月の", "来月の", "先月の", "毎月の")
_ACADEMIC_YEAR_RE = re.compile(r"(?<![A-Za-z0-9])(\d{4})年度")
_YEAR_ORDINAL_RE = re.compile(r"(?<![A-Za-z0-9])(\d+)年目")
_YEAR_DURATION_RE = re.compile(r"(?<![A-Za-z0-9])(\d+)年間")
_YEAR_RE = re.compile(r"(?<![A-Za-z0-9,.])(\d{1,4})年")
_ORDINAL_PREFIX_RE = re.compile(r"(?<!次)第(?=\s*\d)")
# 第1月曜日 is a weekday (げつようび), not a month.
_MONTH_RE = re.compile(r"(?<![A-Za-z0-9])(\d{1,2})月(?!曜)")
_DAY_RE = re.compile(r"(?<![A-Za-z0-9])(\d{1,3}(?:,\d{3})+|\d+)日")
_YEN_SIGN_RE = re.compile(r"[¥￥]\s*(\d[\d,]*)(?![\d.])")
_YEN_RE = re.compile(r"(?<![A-Za-z0-9])(\d[\d,]*)\s*円")
_PERCENT_RE = re.compile(r"(?<![A-Za-z0-9])(\d[\d,]*(?:\.\d+)?)\s*(?:%|パーセント)")
_TECH_UNIT_RE = re.compile(
    r"(?<![A-Za-z0-9])(\d[\d,]*(?:\.\d+)?)([万億]?)\s*"
    r"(EFLOPS|PFLOPS|TFLOPS|GFLOPS|GHz|MHz|kHz|Hz|fps|cores?)(?![A-Za-z0-9])",
    re.IGNORECASE,
)
# Case-sensitive: "Gb" is gigabits and "GB" gigabytes. Lowercase "gb" stays bytes, and
# "bps" is bits per second.
_DATA_UNIT_RE = re.compile(
    r"(?<![A-Za-z0-9])(\d[\d,]*(?:\.\d+)?)([万億]?)\s*([KMGT][Bb]|[kmgt]b|kB)(/s|ps)?(?![A-Za-z0-9])"
)
# Case-sensitive so that e.g. "5G" or "3D" are not mistaken for grams or other units.
_METRIC_UNIT_RE = re.compile(
    r"(?<![A-Za-z0-9])(\d[\d,]*(?:\.\d+)?)([万億]?)\s*"
    r"(km²|km2|m²|m2|m³|m3|㎡|km|cm|mm|kg|mg|mL|ml|kWh|kW|dB|m|g|L|W(?!杯)|V|ha|℃|°C)(?![A-Za-z0-9])"
)
# Digits right after "<letter or digit>." are part of a token such as No.1 or 4.1.9a1.
_DOTTED_VERSION_RE = re.compile(r"(?<!\d)(?<!\d\.)(\d+(?:\.\d+){2,})(?![A-Za-z0-9])")
_DECIMAL_RE = re.compile(r"(?<!\d)(?<!\d\.)(\d[\d,]*\.\d+)(?![A-Za-z0-9]|\.\d)")
_FRACTION_RE = re.compile(r"(?<![A-Za-z0-9])(\d[\d,]*)分の(\d[\d,]*)(?![\d.])")
_LARGE_KANJI_UNIT_RE = re.compile(
    r"(?<![A-Za-z0-9.])(\d[\d,]*)(?:(?=[万億])|(兆|京(?![都浜急阪葉王])))"
)
# 回答/回収 are words (100回答, 1回収), but 1回答えて and 1回収集 are counts.
_COUNTERS = (
    r"丁目|科目|教科|時間|週間|周年|試合|種類|世紀|[かヶケカヵ箇](?:月|所|国)|歳|才|位|基|期|級|台|枚|個|"
    r"本(?![体部日当格来質学])|冊|組(?![織合])|倍|匹|名|人|件|校|回(?!答(?!え)|収(?![集録]))|階(?![層段])|"
    r"点(?!検)|泊|杯|週|軒|頭|足(?![しさすせそ])|着(?!用)|曲(?!線)|社(?![会員長])|種(?![目別])|室(?!長)|"
    r"席|章|周(?![辺囲年])|缶|巻|歩(?![道行])|発(?![表見言売生展信達行明電音想])|箱|通り|"
    r"分(?![野析類担散布割岐解])|秒|時|つ"
)
_COUNTER_RE = re.compile(rf"(?<![A-Za-z0-9])(\d[\d,]*)\s*({_COUNTERS})")
# "2〜3人" -> "2人から3人" so that each side gets its own counter reading.
_NUMBER_RANGE_RE = re.compile(
    rf"(?<![A-Za-z0-9.,:/-])(\d[\d,]*)\s*[〜~～-]\s*(\d[\d,]*)"
    rf"(?=\s*({_COUNTERS}|年|月|日|円|割|%|パーセント|万|億))"
)
# A dash after a number with a unit is a range, not a sign: 9時-17時, 9月1日(火)-9月7日(月).
_WEEKDAY_READING = r"(?:[月火水木金土日]曜日|祝日)"
_UNIT_RANGE_DASH_RE = re.compile(
    rf"(?<=\d)(\s*(?:{_COUNTERS}|年度|年|月|日|円|割|%|パーセント|℃|度)(?:半|生)?"
    rf"(?:、{_WEEKDAY_READING}(?:、{_WEEKDAY_READING})*、)?)\s*[-−]\s*(?=\d)"
)
# Run after the phone, date, address and range passes, which own the hyphens between digits.
_MINUS_SIGN_RE = re.compile(r"(?<![0-9A-Za-z+])(?<!\d\s)[-−－▲](?=\d)")
_PLUS_SIGN_RE = re.compile(r"(?<![A-Za-z+])\+(?=\d)")
# 1年1組 is a class name (いちくみ), and 分 after 割 is a tenth of it (3割5分 -> ごぶ).
_CLASS_GROUP_RE = re.compile(r"(?<=ねん)(\d{1,2})組")
_WARI_BU_RE = re.compile(r"(?<=割)(\d)分")
_VOICELESS_INITIALS = frozenset(
    "かきくけこさしすせそたちつてとぱぴぷぺぽカキクケコサシスセソタチツテトパピプペポ"
)
# Up to three digits: four-digit numbers before katakana are mostly years (2010シーズン).
_KATAKANA_UNIT_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9,])(?<![A-Za-z0-9]\.)(\d{1,3})(?=[カキクケコサシスセソタチツテトパピプペポ])"
)
_STANDALONE_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9])(?<![A-Za-z0-9]\.)(\d[\d,]*)(?![A-Za-z0-9]|\.\d)"
)
_DIGIT_READINGS = ("ぜろ", "いち", "に", "さん", "よん", "ご", "ろく", "なな", "はち", "きゅう")
_LARGE_NUMBER_UNITS = ("", "まん", "おく", "ちょう", "けい")
_LARGE_UNIT_READINGS = {"万": "まん", "億": "おく"}
_TECH_UNIT_READINGS = {
    "eflops": "エクサフロップス",
    "pflops": "ペタフロップス",
    "tflops": "テラフロップス",
    "gflops": "ギガフロップス",
    "ghz": "ギガヘルツ",
    "mhz": "メガヘルツ",
    "khz": "キロヘルツ",
    "hz": "ヘルツ",
    "fps": "エフピーエス",
    "core": "コア",
    "cores": "コア",
}
_DATA_UNIT_PREFIXES = {"k": "キロ", "m": "メガ", "g": "ギガ", "t": "テラ"}
_METRIC_UNIT_READINGS = {
    "km²": "へいほうキロメートル",
    "km2": "へいほうキロメートル",
    "m²": "へいほうメートル",
    "m2": "へいほうメートル",
    "m³": "りっぽうメートル",
    "m3": "りっぽうメートル",
    "㎡": "へいほうメートル",
    "km": "キロメートル",
    "cm": "センチメートル",
    "mm": "ミリメートル",
    "kg": "キログラム",
    "mg": "ミリグラム",
    "mL": "ミリリットル",
    "ml": "ミリリットル",
    "kWh": "キロワットアワー",
    "kW": "キロワット",
    "dB": "デシベル",
    "m": "メートル",
    "g": "グラム",
    "L": "リットル",
    "W": "ワット",
    "V": "ボルト",
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
    "か国": ("かこく", _K_ROW_SOKUON_DIGITS),
    "軒": ("けん", _K_ROW_SOKUON_DIGITS),
    "曲": ("きょく", _K_ROW_SOKUON_DIGITS),
    "缶": ("かん", _K_ROW_SOKUON_DIGITS),
    "巻": ("かん", _K_ROW_SOKUON_DIGITS),
    "歳": ("さい", _S_ROW_SOKUON_DIGITS),
    "才": ("さい", _S_ROW_SOKUON_DIGITS),
    "冊": ("さつ", _S_ROW_SOKUON_DIGITS),
    "週": ("しゅう", _S_ROW_SOKUON_DIGITS),
    "週間": ("しゅうかん", _S_ROW_SOKUON_DIGITS),
    "周年": ("しゅうねん", _S_ROW_SOKUON_DIGITS),
    "点": ("てん", _S_ROW_SOKUON_DIGITS),
    "頭": ("とう", _S_ROW_SOKUON_DIGITS),
    "足": ("そく", _S_ROW_SOKUON_DIGITS),
    "着": ("ちゃく", _S_ROW_SOKUON_DIGITS),
    "社": ("しゃ", _S_ROW_SOKUON_DIGITS),
    "種類": ("しゅるい", _S_ROW_SOKUON_DIGITS),
    "種": ("しゅ", _S_ROW_SOKUON_DIGITS),
    "世紀": ("せいき", _S_ROW_SOKUON_DIGITS),
    "室": ("しつ", _S_ROW_SOKUON_DIGITS),
    "席": ("せき", _S_ROW_SOKUON_DIGITS),
    "丁目": ("ちょうめ", _S_ROW_SOKUON_DIGITS),
    "章": ("しょう", _S_ROW_SOKUON_DIGITS),
    "試合": ("しあい", _S_ROW_SOKUON_DIGITS),
    "周": ("しゅう", _S_ROW_SOKUON_DIGITS),
}
# counter: (plain, geminated p-sound, sound after 3 and after せん/まん)
_H_ROW_COUNTERS = {
    "本": ("ほん", "ぽん", "ぼん"),
    "匹": ("ひき", "ぴき", "びき"),
    "杯": ("はい", "ぱい", "ばい"),
    "歩": ("ほ", "ぽ", "ぽ"),
    "発": ("はつ", "ぱつ", "ぱつ"),
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
# 1箱/2通り use native numerals; other counts keep the counter kanji.
_HITO_FUTA_COUNTERS = {"箱": "はこ", "通り": "とおり"}


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


def _geminate_final_digit(reading: str, *, hundreds: bool = False) -> str:
    """Geminate a final いち/はち/じゅう before a voiceless sound (いってん, じゅっちょう).

    ろく, せん and まん never change; 百 only geminates before k-sounds (ひゃっけい).
    """
    endings = {"いち": "いっ", "はち": "はっ", "じゅう": "じゅっ"}
    if hundreds:
        endings.update({"ひゃく": "ひゃっ", "びゃく": "びゃっ", "ぴゃく": "ぴゃっ"})
    for ending, replacement in endings.items():
        if reading.endswith(ending):
            return reading[: -len(ending)] + replacement
    return reading


def _geminate_ten(reading: str) -> str:
    """じゅう -> じゅっ before a voiceless unit (10% -> じゅっぱーせんと)."""
    return reading[:-1] + "っ" if reading.endswith("じゅう") else reading


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
        # 兆 and 京 start with a voiceless sound: いっちょう, じゅっちょう, ひゃっけい.
        if group_index >= 3:
            reading = _geminate_final_digit(reading, hundreds=group_index == 4)
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
    reading = "れい" if integer == "0" else _geminate_final_digit(_read_japanese_integer(integer))
    return f"{reading}てん{_read_digits(fraction)}"


def _read_dotted_version(raw_value: str) -> str:
    head, *tail = raw_value.split(".")
    readings = [_read_japanese_integer(head), *(_read_digits(component) for component in tail)]
    # Every component before a てん geminates: 13.2.1.1 -> じゅうさんてんにてんいってんいち.
    return "てん".join([*(_geminate_final_digit(reading) for reading in readings[:-1]), readings[-1]])


def _read_unit_quantity(raw_value: str, large_unit: str, unit_reading: str) -> str:
    """Read a number before a unit, e.g. 40万m² or 10km (じゅっキロメートル)."""
    if large_unit:
        if "." in raw_value:
            reading = _read_decimal_value(raw_value)
        else:
            reading = _read_japanese_integer(raw_value, before_large_unit=True)
        return f"{reading}{_LARGE_UNIT_READINGS[large_unit]}{unit_reading}"
    reading = _read_decimal_value(raw_value)
    if "." not in raw_value and unit_reading[:1] in _VOICELESS_INITIALS:
        reading = _geminate_ten(reading)
    return f"{reading}{unit_reading}"


def _read_data_unit(unit: str, rate: str | None) -> str:
    prefix = _DATA_UNIT_PREFIXES[unit[0].lower()]
    # "Gb" is bits; a lowercase "gb" is a sloppy "GB". "bps" is always bits per second.
    bits = unit[1] == "b" and (rate == "ps" or unit[0].isupper())
    return f"{prefix}{'ビット' if bits else 'バイト'}{'毎秒' if rate else ''}"


def _read_month_value(raw_value: str) -> str:
    value = int(raw_value)
    special = {4: "しがつ", 7: "しちがつ", 9: "くがつ"}
    return special.get(value, f"{_read_japanese_integer(str(value))}がつ")


def _read_day_value(raw_value: str, *, calendar: bool = True) -> str:
    """Read N日; only 1日 differs between a date (ついたち) and a span (いちにち)."""
    value = int(raw_value.replace(",", ""))
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
        19: "じゅうくにち",
        20: "はつか",
        24: "にじゅうよっか",
        29: "にじゅうくにち",
    }
    return special.get(value, f"{_read_japanese_integer(str(value))}にち")


def _is_calendar_day(match: re.Match[str]) -> bool:
    # 毎月1日 and 1日（土） are dates; a bare 1日 (1日に2回) is a span.
    if match.string[: match.start()].endswith(_CALENDAR_DAY_PREFIXES):
        return True
    # The weekday paren has already become "、土曜日、".
    return _WEEKDAY_AFTER_DAY_RE.match(match.string, match.end()) is not None


def _read_weekdays(days: str) -> str:
    return "、".join("祝日" if day == "祝" else f"{day}曜日" for day in re.sub(r"[・･、]", "", days))


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
    reading = _read_japanese_integer(str(value))
    final_digit = value % 10
    if final_digit == 4 and reading.endswith("よん"):
        return f"{reading[:-2]}よじ"
    if final_digit == 7 and reading.endswith("なな"):
        return f"{reading[:-2]}{seven}じ"
    if final_digit == 9 and reading.endswith("きゅう"):
        return f"{reading[:-3]}くじ"
    return f"{reading}じ"


def _read_counter(raw_value: str, counter: str) -> str:
    if counter in {"時", "分", "秒", "時間"}:
        # Clock fields are often zero-padded (05分); the zero is not spoken.
        raw_value = str(int(raw_value.replace(",", "")))
    value = int(raw_value.replace(",", ""))
    reading = _read_japanese_integer(raw_value)
    final_digit = value % 10
    if counter[0] in "かヶケカヵ箇" and counter[1:] in {"月", "所", "国"}:
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
    if counter in _HITO_FUTA_COUNTERS:
        if value in {1, 2}:
            return f"{'ひと' if value == 1 else 'ふた'}{_HITO_FUTA_COUNTERS[counter]}"
        return f"{reading}{counter}"
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
    if counter in {"軒", "足"} and final_digit == 3 and reading.endswith("さん"):
        return f"{reading}{'げん' if counter == '軒' else 'ぞく'}"
    if counter in {"分", "泊"}:
        plain, geminated = {"分": ("ふん", "ぷん"), "泊": ("はく", "ぱく")}[counter]
        if value and final_digit in _P_ROW_DIGITS:
            return _read_p_counter(raw_value, geminated)
        return f"{reading}{plain}"
    if counter in _H_ROW_COUNTERS:
        plain, geminated, voiced = _H_ROW_COUNTERS[counter]
        # After ん the h-row voices (さんぼん, せんぼん, いちまんぼん); よんほん stays plain.
        if value and final_digit in {0, 3} and reading.endswith("ん"):
            return f"{reading}{voiced}"
        if value and final_digit in _K_ROW_SOKUON_DIGITS:
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


def _read_clock_time(raw_hour: str, raw_minute: str, raw_second: str | None = None) -> str:
    hour = _read_hour(raw_hour)
    minute = "" if int(raw_minute) == 0 else _read_counter(raw_minute, "分")
    second = "" if raw_second is None or int(raw_second) == 0 else _read_counter(raw_second, "秒")
    return f"{hour}{minute}{second}"


def _read_time_range(match: re.Match[str]) -> str:
    start = _read_clock_time(match.group(1), match.group(2))
    end = _read_clock_time(match.group(3), match.group(4))
    # Keep the text's own まで, and do not turn "〜12:00に" into a deadline (までに).
    following = match.string[match.end() :].lstrip()
    return f"{start}から{end}{'' if following.startswith(('まで', 'に')) else 'まで'}"


def _read_month_day(raw_month: str, raw_day: str) -> str:
    month = _read_month_value(raw_month)
    day = _read_day_value(raw_day)
    # The model slurs the doubled つ in "はちがつついたち"; a space keeps both.
    return f"{month} {day}" if day.startswith("つ") else f"{month}{day}"


def _read_slash_date(raw_year: str | None, raw_month: str, raw_day: str) -> str | None:
    month, day = int(raw_month), int(raw_day)
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    year = "" if raw_year is None else f"{_read_year_value(raw_year)}ねん"
    return f"{year}{_read_month_day(raw_month, raw_day)}"


def _read_large_kanji_unit(match: re.Match[str]) -> str:
    reading = _read_japanese_integer(match.group(1), before_large_unit=True)
    unit = match.group(2)
    if not unit:
        return reading
    # 兆/京 are spelled out so that the geminate is not left before a kanji.
    return f"{_geminate_final_digit(reading, hundreds=unit == '京')}{'ちょう' if unit == '兆' else 'けい'}"


def _read_furigana(match: re.Match[str]) -> str:
    """片倉町（かたくらまち） -> かたくらまち; a long or number-led word keeps its kanji."""
    kanji, reading = match.group(1), re.sub(r"[・\s]", "", match.group(2))
    previous = match.string[max(0, match.start() - 1) : match.start()]
    if (
        len(reading) < min(len(kanji) + 1, 4)
        or reading in _NON_READING_GLOSSES
        # The reading also spells the okurigana before the run: 持ち物（もちもの）.
        or ("ぁ" <= previous <= "ゖ" and previous in reading)
    ):
        # Not a reading of the kanji alone (受付（なし）); leave it to the paren rule.
        return match.group(0)
    if len(kanji) > 4 or previous.isdigit():
        return kanji
    return reading


def _strip_speech_markdown(text: str) -> str:
    text = _MARKDOWN_LINK_RE.sub(r"\1", text)
    # URLs are not read aloud. One used as a noun ("…をご覧ください") becomes
    # ウェブサイト; one in parentheses or on its own is dropped.
    text = _URL_RE.sub(
        lambda match: (
            "ウェブサイト"
            if _URL_AS_NOUN_RE.match(text, match.end())
            and not re.search(r"[(（]\s*$", text[: match.start()])
            else ""
        ),
        text,
    )
    text = re.sub(r"[(（]\s*[)）]", "", text)
    text = _SPEECH_HEADING_RE.sub(
        lambda match: f"{'。' if match.start() else ''}{match.group(2)}。",
        text,
    )
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)
    # A marker right before a digit is a sign (-5℃), not a bullet.
    text = re.sub(r"(?m)^\s*[-*+](?!\d)\s*", "", text)
    text = re.sub(r"(?:\*\*|__|~~|`)", "", text)
    text = re.sub(r"\s+\+\s+", "プラス", text)
    text = re.sub(r"\s+(?:\*|\+(?!\d))\s*", "。", text)
    text = re.sub(r"[\r\n]+", "。", text)
    text = text.replace("•", "。").replace("◦", "。").replace("*", "。")
    text = re.sub(r"\s*[→⇒]\s*", "、", text)
    text = re.sub(r"。(?:\s*。)+", "。", text)
    text = re.sub(r"\s*。\s*", "。", text)
    return text.strip(" 。") + ("。" if text.strip(" 。") and text.rstrip().endswith("。") else "")


def _verbalize_speech_symbols(text: str) -> str:
    # A note mark inside a sentence keeps a pause (不要です※ただし -> 不要です、ただし).
    text = re.sub(r"(?<=[^\s。、!?])※", "、", text).replace("※", "")
    text = _WAVE_DASH_LONG_VOWEL_RE.sub("ー", text)
    text = _CJK_AMPERSAND_RE.sub("と", text)
    return _MULTIPLICATION_RE.sub("かける", text)


def _normalize_japanese_numbers(text: str) -> str:
    # The model garbles a mixed 第いち…; spell the ordinal prefix in kana as well.
    normalized = _ORDINAL_PREFIX_RE.sub("だい", text)
    for pattern, replacement in _SPACED_DATE_PATTERNS:
        normalized = pattern.sub(replacement, normalized)
    normalized = _NON_THOUSANDS_COMMA_RE.sub("、", normalized)
    normalized = _WEEKDAY_PAREN_RE.sub(lambda match: f"、{_read_weekdays(match.group(1))}、", normalized)
    normalized = _ISO_DATE_RE.sub(
        lambda match: _read_slash_date(*match.groups()) or match.group(0),
        normalized,
    )
    normalized = _POSTAL_CODE_RE.sub(
        lambda match: "ゆうびんばんごう" + "の".join(_read_digits(group) for group in match.groups()),
        normalized,
    )
    normalized = _PHONE_NUMBER_RE.sub(
        lambda match: "の".join(_read_digits(group) for group in match.groups()),
        normalized,
    )
    normalized = _ADDRESS_NUMBER_RE.sub(
        lambda match: match.group(1)
        + "の".join(_read_japanese_integer(group) for group in match.groups()[1:] if group),
        normalized,
    )
    normalized = _PLACE_BLOCK_NUMBER_RE.sub(
        lambda match: "の".join(_read_japanese_integer(group) for group in match.groups()),
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
    normalized = _LABEL_COLON_RE.sub(r"\1\2、", normalized)
    normalized = _TIME_RANGE_RE.sub(_read_time_range, normalized)
    normalized = _OPEN_TIME_RE.sub(
        lambda match: f"{_read_clock_time(match.group(1), match.group(2))}から",
        normalized,
    )
    normalized = _CLOCK_TIME_RE.sub(lambda match: _read_clock_time(*match.groups()), normalized)
    normalized = _HOUR_MINUTE_RE.sub(lambda match: _read_clock_time(*match.groups()), normalized)
    normalized = _RATIO_RE.sub(
        lambda match: (
            f"{_read_japanese_integer(match.group(1))}たい{_read_japanese_integer(match.group(2))}"
        ),
        normalized,
    )
    normalized = _NUMBER_RANGE_RE.sub(
        lambda match: f"{match.group(1)}{match.group(3)}から{match.group(2)}",
        normalized,
    )
    normalized = _UNIT_RANGE_DASH_RE.sub(r"\1から", normalized)
    normalized = _MINUS_SIGN_RE.sub("マイナス", normalized)
    normalized = _PLUS_SIGN_RE.sub("プラス", normalized)
    normalized = _FULL_DATE_RE.sub(
        lambda match: (
            f"{_read_year_value(match.group(1))}ねん"
            f"{_read_month_day(match.group(2), match.group(3))}"
        ),
        normalized,
    )
    normalized = _MONTH_DAY_RE.sub(
        lambda match: _read_month_day(match.group(1), match.group(2)),
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
        lambda match: _read_day_value(match.group(1), calendar=_is_calendar_day(match)),
        normalized,
    )
    normalized = _YEN_RE.sub(
        lambda match: f"{_read_year_value(match.group(1))}えん",
        normalized,
    )
    normalized = _DATA_UNIT_RE.sub(
        lambda match: _read_unit_quantity(
            match.group(1), match.group(2), _read_data_unit(match.group(3), match.group(4))
        ),
        normalized,
    )
    normalized = _TECH_UNIT_RE.sub(
        lambda match: _read_unit_quantity(
            match.group(1), match.group(2), _TECH_UNIT_READINGS[match.group(3).lower()]
        ),
        normalized,
    )
    normalized = _METRIC_UNIT_RE.sub(
        lambda match: _read_unit_quantity(
            match.group(1), match.group(2), _METRIC_UNIT_READINGS[match.group(3)]
        ),
        normalized,
    )
    normalized = _PERCENT_RE.sub(
        lambda match: _read_unit_quantity(match.group(1), "", "ぱーせんと"),
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
    normalized = _LARGE_KANJI_UNIT_RE.sub(_read_large_kanji_unit, normalized)
    normalized = _CLASS_GROUP_RE.sub(
        lambda match: f"{_read_japanese_integer(match.group(1))}くみ",
        normalized,
    )
    normalized = _WARI_BU_RE.sub(lambda match: f"{_read_japanese_integer(match.group(1))}ぶ", normalized)
    normalized = _COUNTER_RE.sub(
        lambda match: _read_counter(match.group(1), match.group(2)),
        normalized,
    )
    normalized = _KATAKANA_UNIT_NUMBER_RE.sub(
        lambda match: _geminate_ten(_read_japanese_integer(match.group(1))),
        normalized,
    )
    normalized = _STANDALONE_NUMBER_RE.sub(
        lambda match: _read_japanese_integer(match.group(1)),
        normalized,
    )
    normalized = re.sub(r"、(?:\s*、)+", "、", normalized)
    normalized = re.sub(r"、\s*。", "。", normalized)
    normalized = re.sub(r"、\s+", "、", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _naturalize_speech_punctuation(text: str) -> str:
    # Only a kanji, katakana or Latin label takes は (日時：, CMC:); です： and ratios do not.
    text = re.sub(r"([A-Za-zァ-ヶー一-鿿]+)[：:]", r"\1は、", text)
    # Repeat so that nested parentheses are flattened too.
    for _ in range(3):
        text, count = re.subn(r"[(（]([^()（）]{1,40})[)）]", r"、\1、", text)
        if not count:
            break
    text = text.replace("・", "、").replace("~", "から").replace("〜", "から").replace("～", "から")
    text = text.replace("：", "、").replace(":", "、")
    text = re.sub(r"、(?:\s*、)+", "、", text)
    text = re.sub(r"、\s*。", "。", text)
    return text.strip().lstrip("、").lstrip()


def normalize_speech_input(text: str) -> str:
    normalized = _strip_speech_markdown(text.strip().translate(_FULL_WIDTH_ASCII_TRANSLATION))
    normalized = _FURIGANA_RE.sub(_read_furigana, normalized)
    # Hints run before numbers so that e.g. "TOP 500" is still intact.
    for pattern, replacement in PRONUNCIATION_HINTS:
        normalized = pattern.sub(replacement, normalized)
    normalized = _verbalize_speech_symbols(normalized)
    normalized = _normalize_japanese_numbers(normalized)
    normalized = _naturalize_speech_punctuation(normalized)
    return FIXED_SPEECH_NORMALIZATIONS.get(normalized, normalized)


_SMALL_KANA = frozenset("ゃゅょぁぃぅぇぉゎャュョァィゥェォヮ")
_PAUSE_CHARS = frozenset("、，,。．！？!?")
# Upstream Irodori NFKC-folds its input, so the model speaks these like their normal forms.
_WIDTH_VARIANT_RE = re.compile(r"[０-９Ａ-Ｚａ-ｚ\uff61-\uff9f]+")
# Spoken letter names: ダブリュー/エックス 4, エイチ/アール/ゼット 3, the rest (エー, ビー, ...) 2.
_LETTER_NAME_MORA = {"H": 3.0, "R": 3.0, "W": 4.0, "X": 4.0, "Z": 3.0}


def _fold_width_variants(text: str) -> str:
    import unicodedata

    return _WIDTH_VARIANT_RE.sub(lambda match: unicodedata.normalize("NFKC", match.group()), text)


def _is_kana(char: str) -> bool:
    return "ぁ" <= char <= "ゖ" or "ァ" <= char <= "ヺ" or char == "ー"


def _is_kanji(char: str) -> bool:
    return "一" <= char <= "鿿" or char in "々〇"


def has_speakable_text(text: str) -> bool:
    return any(
        (char.isascii() and char.isalnum()) or _is_kana(char) or _is_kanji(char)
        for char in _fold_width_variants(text)
    )


def estimate_speech_mora(text: str) -> float:
    """Rough spoken length in mora.

    Kana count 1, kanji 1.7 (typical on'yomi), 〇 2, uppercase letters their
    spoken letter name, lowercase letters 1, digits 1.5, and a pause at
    punctuation inside the text 2. Full-width Latin and half-width katakana
    count like their normal forms.
    """
    text = _fold_width_variants(text)
    mora = 0.0
    last_index = len(text.rstrip()) - 1
    for index, char in enumerate(text):
        if char in _SMALL_KANA:
            continue
        if _is_kana(char):
            mora += 1.0
        elif char == "〇":
            mora += 2.0
        elif _is_kanji(char):
            mora += 1.7
        elif "A" <= char <= "Z":
            mora += _LETTER_NAME_MORA.get(char, 2.0)
        elif "a" <= char <= "z":
            mora += 1.0
        elif "0" <= char <= "9":
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
    # 1.0 since the mora cap below handles sparse short phrases; 0.8 squeezed dense
    # ones such as 東京工科大学の学長です。 to ~9.6 mora/s (2026-10-08 benchmark).
    short_duration_scale: float = 1.0
    # v4 checkpoints overestimate the length of very short phrases and fill the extra
    # time with babble ("はい。" -> 3 s of speech), so cap the predicted duration at
    # base + estimated mora / rate. A rate <= 0 disables the cap.
    duration_cap_base_seconds: float = 0.7
    duration_cap_mora_per_second: float = 6.0
    # Every clip carries ~0.23 s lead and ~0.42 s trail of model noise padding. Keep this
    # much around the speech (a longer trail after a sentence end), fade the cut edges,
    # and leave headroom for the codec's overshoot above full scale.
    trim_enabled: bool = True
    trim_lead_ms: float = 60.0
    sentence_trail_ms: float = 300.0
    clause_trail_ms: float = 120.0
    fade_ms: float = 10.0
    output_gain_db: float = -0.5
    # "speechcpp" swaps only the synthesis for a resident speech.cpp worker (Vulkan); the
    # HTTP contract, normalization, caches and output stage stay the same. It has no
    # watermark and no LoRA, and CUDA_VISIBLE_DEVICES does not select its GPU: the Vulkan
    # device name does, checked against speechcpp_gpu_uuid when that is set.
    backend: str = "python"
    speechcpp_bin: Path = DEFAULT_SPEECHCPP_BIN
    speechcpp_model: Path = DEFAULT_SPEECHCPP_MODEL
    # Built once on the CPU from reference_wav; a GPU-built voice file cloned less closely
    # (2026-10-16 benchmark), and building takes ~100 s, so the service never builds it.
    speechcpp_voice: Path = DEFAULT_SPEECHCPP_VOICE
    speechcpp_device: str = "Vulkan0"
    speechcpp_gpu_uuid: str | None = None
    speechcpp_threads: int = 8
    speechcpp_timeout_seconds: float = 120.0

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "IrodoriSettings":
        values = os.environ if env is None else env
        checkpoint_raw = values.get("IRODORI_TTS_CHECKPOINT", "").strip()
        lora_raw = values.get("IRODORI_TTS_LORA_ADAPTER", "").strip()
        caption_raw = values.get("IRODORI_TTS_CAPTION", "").strip()
        stream_chunk_bytes = _env_int(values, "IRODORI_TTS_STREAM_CHUNK_BYTES", 3200)
        if stream_chunk_bytes <= 0:
            raise ValueError("IRODORI_TTS_STREAM_CHUNK_BYTES must be positive")
        backend = values.get("IRODORI_TTS_BACKEND", "python").strip() or "python"
        if backend not in BACKENDS:
            raise ValueError(f"IRODORI_TTS_BACKEND must be one of: {', '.join(BACKENDS)}")
        if backend == "speechcpp" and lora_raw:
            raise ValueError("the speechcpp backend has no LoRA support; unset IRODORI_TTS_LORA_ADAPTER")
        gpu_uuid_raw = values.get("IRODORI_TTS_SPEECHCPP_GPU_UUID", "").strip()
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
            short_duration_scale=_env_float(values, "IRODORI_TTS_SHORT_DURATION_SCALE", 1.0),
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
            stream_chunk_bytes=stream_chunk_bytes,
            progressive_stream_enabled=_env_bool(
                values,
                "IRODORI_TTS_PROGRESSIVE_STREAM_ENABLED",
                False,
            ),
            progressive_max_segments=_env_int(values, "IRODORI_TTS_PROGRESSIVE_MAX_SEGMENTS", 2),
            trim_enabled=_env_bool(values, "IRODORI_TTS_TRIM_ENABLED", True),
            trim_lead_ms=_env_float(values, "IRODORI_TTS_TRIM_LEAD_MS", 60.0),
            sentence_trail_ms=_env_float(values, "IRODORI_TTS_SENTENCE_TRAIL_MS", 300.0),
            clause_trail_ms=_env_float(values, "IRODORI_TTS_CLAUSE_TRAIL_MS", 120.0),
            fade_ms=_env_float(values, "IRODORI_TTS_FADE_MS", 10.0),
            output_gain_db=_env_float(values, "IRODORI_TTS_OUTPUT_GAIN_DB", -0.5),
            backend=backend,
            speechcpp_bin=_env_path(values, "IRODORI_TTS_SPEECHCPP_BIN", DEFAULT_SPEECHCPP_BIN),
            speechcpp_model=_env_path(values, "IRODORI_TTS_SPEECHCPP_MODEL", DEFAULT_SPEECHCPP_MODEL),
            speechcpp_voice=_env_path(values, "IRODORI_TTS_SPEECHCPP_VOICE", DEFAULT_SPEECHCPP_VOICE),
            speechcpp_device=values.get("IRODORI_TTS_SPEECHCPP_DEVICE", "Vulkan0").strip() or "Vulkan0",
            speechcpp_gpu_uuid=gpu_uuid_raw or None,
            speechcpp_threads=_env_int(values, "IRODORI_TTS_SPEECHCPP_THREADS", 8),
            speechcpp_timeout_seconds=_env_float(values, "IRODORI_TTS_SPEECHCPP_TIMEOUT_SECONDS", 120.0),
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


# Fixed-phrase PCM files are 16 kHz mono s16le (*_16k_mono.pcm).
FIXED_PHRASE_SAMPLE_RATE_HZ = 16000
# numpy fallback low-pass: 48 taps per phase, Kaiser beta 8, cutoff at 0.95 x target Nyquist.
_FIR_TAPS_PER_PHASE = 48
_FIR_KAISER_BETA = 8.0
_FIR_CUTOFF_RATIO = 0.95
_TRIM_FLOOR_PERCENTILE = 10.0
_TRIM_SPEECH_ABOVE_FLOOR_DB = 12.0


def _logger():
    import logging

    return logging.getLogger("irodori_server")


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


def _decimate_windowed_sinc(audio: np.ndarray, factor: int) -> np.ndarray:
    half = _FIR_TAPS_PER_PHASE * factor // 2
    offsets = np.arange(-half, half + 1)
    cutoff = _FIR_CUTOFF_RATIO * 0.5 / factor
    taps = np.sinc(2.0 * cutoff * offsets) * np.kaiser(offsets.size, _FIR_KAISER_BETA)
    taps /= taps.sum()
    filtered = np.convolve(audio.astype(np.float64), taps)[half : half + audio.size]
    return filtered[::factor]


def resample_audio(audio: np.ndarray, source_sample_rate: int, target_sample_rate: int) -> np.ndarray:
    """Band-limited resampling to round(len * target / source) samples.

    scipy is imported lazily. Without it, integer decimation (48 kHz -> 16 kHz) uses a
    windowed-sinc low-pass and other ratios fall back to linear interpolation.
    """
    if source_sample_rate == target_sample_rate or audio.size <= 1:
        return _resample_linear(audio, source_sample_rate, target_sample_rate)
    target_len = max(1, int(round(audio.size * target_sample_rate / source_sample_rate)))
    try:
        from scipy.signal import resample_poly
    except ImportError:
        if source_sample_rate % target_sample_rate:
            return _resample_linear(audio, source_sample_rate, target_sample_rate)
        resampled = _decimate_windowed_sinc(audio, source_sample_rate // target_sample_rate)
    else:
        divisor = int(np.gcd(source_sample_rate, target_sample_rate))
        resampled = resample_poly(audio, target_sample_rate // divisor, source_sample_rate // divisor)
    # Both paths return ceil(len * up / down) samples.
    return resampled[:target_len].astype(np.float32)


def _ms_to_samples(milliseconds: float, sample_rate: int) -> int:
    return max(0, int(round(milliseconds * sample_rate / 1000.0)))


def _speech_frame_mask(audio: np.ndarray, frame: int) -> np.ndarray:
    """Frames more than 12 dB above the signal's noise floor (its 10th-percentile frame level)."""
    frame_count = -(-audio.size // frame)
    padded = np.zeros(frame_count * frame, dtype=np.float64)
    padded[: audio.size] = audio
    counts = np.full(frame_count, frame, dtype=np.float64)
    counts[-1] = audio.size - (frame_count - 1) * frame
    levels_db = 10.0 * np.log10((padded.reshape(frame_count, frame) ** 2).sum(axis=1) / counts + 1e-12)
    return levels_db > np.percentile(levels_db, _TRIM_FLOOR_PERCENTILE) + _TRIM_SPEECH_ABOVE_FLOOR_DB


def trim_speech_padding(audio: np.ndarray, *, sample_rate: int, lead_ms: float, trail_ms: float) -> np.ndarray:
    """Cut the noise padding before the first and after the last speech frame.

    Speech frames are 10 ms frames more than 12 dB above the clip's noise floor (its
    10th-percentile frame level). Internal pauses stay, nothing is padded, and a clip
    without any speech frame is returned unchanged.

    s/sh/h and a devoiced final す are 3-8 kHz noise below the broadband threshold, so the
    ends are then extended through contiguous speech frames of the pre-emphasized signal.
    Only contiguous ones: isolated high-frequency bursts in the padding stay out.
    """
    frame = max(1, sample_rate // 100)
    if audio.size == 0:
        return audio
    speech_frames = np.flatnonzero(_speech_frame_mask(audio, frame))
    if speech_frames.size == 0:
        return audio
    first, last = int(speech_frames[0]), int(speech_frames[-1])
    fricative = _speech_frame_mask(np.diff(audio, prepend=audio[:1]), frame)
    while first > 0 and fricative[first - 1]:
        first -= 1
    while last < fricative.size - 1 and fricative[last + 1]:
        last += 1
    start = max(0, first * frame - _ms_to_samples(lead_ms, sample_rate))
    end = min(audio.size, (last + 1) * frame + _ms_to_samples(trail_ms, sample_rate))
    return audio[start:end]


def apply_edge_fades(audio: np.ndarray, *, sample_rate: int, fade_ms: float) -> np.ndarray:
    """Raised-cosine fade-in and fade-out, so the clip starts and ends at zero."""
    fade = min(_ms_to_samples(fade_ms, sample_rate), audio.size // 2)
    if fade <= 0:
        return audio
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(fade) / fade)
    faded = audio.astype(np.float32, copy=True)
    faded[:fade] *= ramp
    faded[-fade:] *= ramp[::-1]
    return faded


def _float_to_pcm16(audio: np.ndarray, *, gain_db: float = 0.0) -> bytes:
    scaled = audio * (10.0 ** (gain_db / 20.0)) if gain_db else audio
    clipped = np.clip(scaled, -1.0, 1.0)
    return np.rint(np.where(clipped >= 0, clipped * 32767.0, clipped * 32768.0)).astype("<i2").tobytes()


def _pcm16_to_float32(pcm: bytes) -> np.ndarray:
    # Inverse of _float_to_pcm16's asymmetric scale, so unprocessed samples round-trip exactly.
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
    return np.where(samples >= 0, samples / 32767.0, samples / 32768.0).astype(np.float32)


def audio_to_pcm16(audio: object, *, source_sample_rate: int, target_sample_rate: int) -> bytes:
    mono = _to_mono_float32(audio)
    return _float_to_pcm16(resample_audio(mono, source_sample_rate, target_sample_rate))


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


def nvidia_gpu_uuid_for_pid(pid: int) -> str | None:
    """The GPU UUID nvidia-smi lists for a process, or None when it is on no NVIDIA GPU."""
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 2 and fields[0] == str(pid):
            return fields[1]
    return None


def _read_json_lines(stream, messages: "queue.Queue[dict | None]") -> None:
    try:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                messages.put(json.loads(line))
            except json.JSONDecodeError:
                _logger().warning("speech.cpp worker printed a non-JSON line: %s", line)
    except (OSError, ValueError):
        pass  # the pipe was closed by _stop_locked
    finally:
        messages.put(None)


class SpeechCppWorker:
    """A resident `speech worker` process (speech.cpp's JSON-lines protocol), one request at a time.

    The synthesis lock and the progressive-stream lock of the app are independent, so this
    class serializes requests itself. A worker that exits is restarted by the next request.
    """

    # Irodori's codec rate; the worker sends mono s16le chunks at this rate.
    sample_rate = 48000

    def __init__(self, settings: IrodoriSettings, *, gpu_probe=nvidia_gpu_uuid_for_pid) -> None:
        self._settings = settings
        self._gpu_probe = gpu_probe
        self._process: subprocess.Popen | None = None
        self._messages: queue.Queue | None = None
        self._lock = Lock()
        self._next_id = 0
        self.gpu_uuid: str | None = None

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self) -> None:
        with self._lock:
            self._start_locked()

    def close(self) -> None:
        with self._lock:
            self._stop_locked()

    def synthesize(self, request: dict[str, object]) -> np.ndarray:
        """Mono float32 audio at sample_rate for one synthesize request (without type/id)."""
        with self._lock:
            if not self.running:
                self._start_locked()
            self._next_id += 1
            request_id = str(self._next_id)
            line = json.dumps({"type": "synthesize", "id": request_id, **request}, ensure_ascii=False)
            try:
                self._process.stdin.write(line + "\n")
                self._process.stdin.flush()
            except OSError as exc:
                self._stop_locked()
                raise RuntimeError("speech.cpp worker exited before the request") from exc
            parts: list[bytes] = []
            while True:
                message = self._receive_locked()
                if message.get("id") != request_id:
                    continue
                kind = message.get("type")
                if kind == "chunk":
                    parts.append(base64.b64decode(message["pcm"], validate=True))
                elif kind == "end":
                    break
                elif kind == "error":
                    error = message.get("error") or {}
                    raise RuntimeError(f"speech.cpp: {error.get('code')}: {error.get('message')}")
        pcm = b"".join(parts)
        if len(pcm) % 2:
            raise RuntimeError("speech.cpp worker returned a partial sample")
        return _pcm16_to_float32(pcm)

    def _start_locked(self) -> None:
        if self.running:
            return
        self._stop_locked()
        settings = self._settings
        if not settings.speechcpp_voice.is_file():
            raise FileNotFoundError(
                f"speech.cpp voice file not found: {settings.speechcpp_voice}; build it once (~100 s) with "
                f"`{settings.speechcpp_bin} voice {settings.speechcpp_model} {settings.reference_wav} "
                f"{settings.speechcpp_voice} --device cpu`"
            )
        # stderr is inherited, so the worker's own log lands in the service journal.
        process = subprocess.Popen(
            [
                str(settings.speechcpp_bin),
                "worker",
                str(settings.speechcpp_model),
                "--device",
                settings.speechcpp_device,
                "--threads",
                str(settings.speechcpp_threads),
                "--add-voice",
                f"{settings.voice}={settings.speechcpp_voice}",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        messages: queue.Queue = queue.Queue()
        Thread(target=_read_json_lines, args=(process.stdout, messages), daemon=True).start()
        self._process, self._messages = process, messages
        try:
            ready = self._receive_locked()
            if ready.get("type") != "ready":
                raise RuntimeError(f"speech.cpp worker did not start: {ready}")
            gpu_uuid = self._gpu_probe(process.pid) if self._gpu_probe is not None else None
            expected = settings.speechcpp_gpu_uuid
            if expected and gpu_uuid != expected:
                raise RuntimeError(
                    f"speech.cpp worker runs on {gpu_uuid or 'no NVIDIA GPU'}, expected {expected}; "
                    "set IRODORI_TTS_SPEECHCPP_DEVICE (Vulkan numbers GPUs differently from CUDA)"
                )
        except BaseException:
            self._stop_locked()
            raise
        self.gpu_uuid = gpu_uuid
        _logger().info(
            "speech.cpp worker %d ready on %s (%s)", process.pid, settings.speechcpp_device, gpu_uuid or "unknown GPU"
        )

    def _receive_locked(self) -> dict:
        try:
            message = self._messages.get(timeout=self._settings.speechcpp_timeout_seconds)
        except queue.Empty:
            self._stop_locked()
            raise RuntimeError(
                f"speech.cpp worker gave no answer in {self._settings.speechcpp_timeout_seconds:g} s; stopped it"
            ) from None
        if message is None:
            code = self._process.wait(timeout=10) if self._process is not None else None
            self._stop_locked()
            raise RuntimeError(f"speech.cpp worker exited (code {code})")
        return message

    def _stop_locked(self) -> None:
        process, self._process, self._messages = self._process, None, None
        self.gpu_uuid = None
        if process is None:
            return
        try:
            process.stdin.close()
        except OSError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        try:
            process.stdout.close()
        except OSError:
            pass


class IrodoriSynthesizer:
    def __init__(self, settings: IrodoriSettings, *, gpu_probe=nvidia_gpu_uuid_for_pid) -> None:
        self._settings = settings
        self._gpu_probe = gpu_probe
        self._runtime = None
        # Concurrent first requests (synthesis, progressive stream, prewarm) must load one runtime.
        self._runtime_lock = Lock()
        self._short_cache: OrderedDict[tuple[str, str], tuple[bytes, str, int]] = OrderedDict()
        # Prewarmed phrases (the filler, the endpoint check's text) are never evicted by live speech.
        self._pinned_short_cache: dict[tuple[str, str], tuple[bytes, str, int]] = {}
        self._pinned_texts = {normalize_speech_input(text) for text in settings.short_cache_prewarm_texts}
        self._short_cache_lock = Lock()

    def load_runtime(self):
        if self._runtime is not None:
            return self._runtime
        with self._runtime_lock:
            if self._runtime is None:
                self._runtime = self._create_runtime()
        return self._runtime

    def _create_runtime(self):
        if self._settings.backend == "speechcpp":
            worker = SpeechCppWorker(self._settings, gpu_probe=self._gpu_probe)
            worker.start()
            return worker

        from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, download_hf_checkpoint

        if self._settings.checkpoint is not None:
            checkpoint = str(self._settings.checkpoint)
        else:
            # v4 checkpoints need the bundled tokenizer/ next to model.safetensors, and
            # quantized variants are addressed as "repo_id/subfolder".
            checkpoint = download_hf_checkpoint(self._settings.hf_checkpoint)

        return InferenceRuntime.from_key(
            RuntimeKey(
                checkpoint=checkpoint,
                model_device=self._settings.model_device,
                codec_repo=self._settings.codec_repo,
                model_precision=self._settings.model_precision,
                codec_device=self._settings.codec_device,
                codec_precision=self._settings.codec_precision,
            )
        )

    def _synthesize_normalized(
        self,
        normalized_text: str,
        *,
        response_format: str,
        num_steps: int | None = None,
    ) -> tuple[bytes, str, int]:
        if self._settings.backend == "speechcpp":
            audio = self.load_runtime().synthesize(self._speechcpp_request(normalized_text, num_steps))
            pcm = self._speech_pcm16(
                audio,
                source_sample_rate=SpeechCppWorker.sample_rate,
                normalized_text=normalized_text,
            )
            return self._pcm_response(pcm, response_format)

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
        pcm = self._speech_pcm16(
            result.audio,
            source_sample_rate=int(result.sample_rate),
            normalized_text=normalized_text,
        )
        return self._pcm_response(pcm, response_format)

    def _speechcpp_request(self, normalized_text: str, num_steps: int | None) -> dict[str, object]:
        settings = self._settings
        request: dict[str, object] = {"text": normalized_text, "voice": settings.voice}
        steps = settings.num_steps if num_steps is None else num_steps
        if steps is not None:
            request["steps"] = steps
        if settings.caption:
            request["instructions"] = settings.caption
        short = settings.short_duration_max_chars > 0 and len(normalized_text) <= settings.short_duration_max_chars
        if short and settings.duration_cap_mora_per_second > 0:
            # speech.cpp has no max_seconds for Irodori, and v4 over-predicts short phrases and
            # fills the extra time with babble, so a short text gets the mora cap as its length
            # (2026-10-16: babble after 少々お待ちください。 9/10 -> 0/10).
            request["seconds"] = self._max_seconds_for(normalized_text)
        else:
            duration_scale = settings.short_duration_scale if short else settings.duration_scale
            if duration_scale != 1.0:
                request["duration_scale"] = duration_scale
        return request

    def model_loaded(self) -> bool:
        if isinstance(self._runtime, SpeechCppWorker):
            return self._runtime.running
        return self._runtime is not None

    def runtime_gpu_uuid(self) -> str | None:
        return self._runtime.gpu_uuid if isinstance(self._runtime, SpeechCppWorker) else None

    def close(self) -> None:
        if isinstance(self._runtime, SpeechCppWorker):
            self._runtime.close()

    def _speech_pcm16(self, audio: object, *, source_sample_rate: int, normalized_text: str) -> bytes:
        """Resample, trim the noise padding, fade the edges, then apply the output gain."""
        settings = self._settings
        sample_rate = settings.response_sample_rate_hz
        speech = resample_audio(_to_mono_float32(audio), source_sample_rate, sample_rate)
        if settings.trim_enabled:
            ends_sentence = normalized_text.endswith(tuple(SENTENCE_END_CHARS))
            speech = trim_speech_padding(
                speech,
                sample_rate=sample_rate,
                lead_ms=settings.trim_lead_ms,
                trail_ms=settings.sentence_trail_ms if ends_sentence else settings.clause_trail_ms,
            )
        speech = apply_edge_fades(speech, sample_rate=sample_rate, fade_ms=settings.fade_ms)
        return _float_to_pcm16(speech, gain_db=settings.output_gain_db)

    def _pcm_response(self, pcm: bytes, response_format: str) -> tuple[bytes, str, int]:
        sample_rate = self._settings.response_sample_rate_hz
        if response_format == "wav":
            return pcm16_to_wav_bytes(pcm, sample_rate_hz=sample_rate), "audio/wav", sample_rate
        return pcm, "audio/L16", sample_rate

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
            pinned = self._pinned_short_cache.get(key)
            if pinned is not None:
                return pinned
            cached = self._short_cache.get(key)
            if cached is None:
                return None
            self._short_cache.move_to_end(key)
            return cached

    def _put_short_cache(self, key: tuple[str, str], value: tuple[bytes, str, int]) -> None:
        with self._short_cache_lock:
            if key[1] in self._pinned_texts:
                self._pinned_short_cache[key] = value
                return
            self._short_cache[key] = value
            self._short_cache.move_to_end(key)
            while len(self._short_cache) > self._settings.short_cache_max_entries:
                self._short_cache.popitem(last=False)

    def short_cache_size(self) -> int:
        with self._short_cache_lock:
            return len(self._pinned_short_cache) + len(self._short_cache)

    def prewarm_short_cache(self, texts: tuple[str, ...], *, num_steps: int | None = None) -> int:
        """Synthesize fixed phrases before the service accepts requests and pin them in the cache."""
        warmed = 0
        for text in texts:
            normalized_text = normalize_speech_input(text)
            if not normalized_text or not self._is_short_cacheable(normalized_text):
                continue
            with self._short_cache_lock:
                self._pinned_texts.add(normalized_text)
                for key in [key for key in self._short_cache if key[1] == normalized_text]:
                    self._pinned_short_cache[key] = self._short_cache.pop(key)
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
        ready_audio = self._ready_audio_for_normalized(normalized_text, response_format=response_format)
        if ready_audio is not None:
            return ready_audio

        result = self._synthesize_normalized(
            normalized_text,
            response_format=response_format,
            num_steps=num_steps,
        )
        if self._is_short_cacheable(normalized_text):
            self._put_short_cache((response_format, normalized_text), result)
        return result

    def _ready_audio_for_normalized(
        self,
        normalized_text: str,
        *,
        response_format: str,
    ) -> tuple[bytes, str, int] | None:
        """Audio that needs no GPU: silence for unspeakable text, the fixed greeting, or a cache hit."""
        if not has_speakable_text(normalized_text):
            return self._pcm_response(b"", response_format)

        fixed_audio = self._fixed_audio_for_normalized(normalized_text, response_format=response_format)
        if fixed_audio is not None:
            return fixed_audio

        if self._is_short_cacheable(normalized_text):
            return self._get_short_cache((response_format, normalized_text))
        return None

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
            _logger().warning("invalid fixed greeting PCM %s (%d bytes); synthesizing instead", fixed_path, len(pcm))
            return None
        pcm = self._speech_pcm16(
            _pcm16_to_float32(pcm),
            source_sample_rate=FIXED_PHRASE_SAMPLE_RATE_HZ,
            normalized_text=normalized_text,
        )
        return self._pcm_response(pcm, response_format)

    def _normalize_request(self, text: str, *, response_format: str) -> str:
        if not text.strip():
            raise ValueError("input text must be non-empty")
        if response_format not in {"pcm", "wav"}:
            raise ValueError("response_format must be one of: pcm, wav")
        # Text that normalizes to nothing (a bare URL, symbols) is answered with
        # empty audio by _ready_audio_for_normalized instead of an error.
        return normalize_speech_input(text)

    def lookup_ready_audio(self, text: str, *, response_format: str = "pcm") -> tuple[bytes, str, int] | None:
        """Answer text without the GPU when possible; None means it needs synthesize()."""
        normalized_text = self._normalize_request(text, response_format=response_format)
        return self._ready_audio_for_normalized(normalized_text, response_format=response_format)

    def synthesize(self, text: str, *, response_format: str = "pcm") -> tuple[bytes, str, int]:
        normalized_text = self._normalize_request(text, response_format=response_format)
        return self._synthesize_cached_normalized(normalized_text, response_format=response_format)

    def plan_progressive_stream(self, text: str) -> tuple[bytes | None, list[str]]:
        """GPU-free part of a progressive stream: ready PCM for the whole text, or its segments."""
        normalized_text = self._normalize_request(text, response_format="pcm")
        ready_audio = self._ready_audio_for_normalized(normalized_text, response_format="pcm")
        if ready_audio is not None:
            return ready_audio[0], []

        segments = coalesce_progressive_segments(
            split_progressive_speech_input(text),
            max_segments=self._settings.progressive_max_segments,
        )
        if not segments:
            raise ValueError("input text must be non-empty")
        return None, segments

    def iter_pcm_stream(self, text: str, *, chunk_bytes: int) -> Iterator[bytes]:
        ready_pcm, segments = self.plan_progressive_stream(text)
        if ready_pcm is not None:
            yield from iter_pcm_chunks(ready_pcm, chunk_bytes=chunk_bytes)
            return

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
        try:
            yield from synthesizer.iter_pcm_stream(text, chunk_bytes=chunk_bytes)
        except Exception:
            # The 200 headers are already sent, so the client only sees a truncated body.
            _logger().exception("progressive speech stream failed")
            raise

    if lock is None:
        yield from _iter()
        return

    with lock:
        yield from _iter()


def create_app(settings: IrodoriSettings | None = None, synthesizer: IrodoriSynthesizer | None = None):
    from fastapi import FastAPI, HTTPException, Request
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

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        if hasattr(resolved_synthesizer, "close"):
            await asyncio.to_thread(resolved_synthesizer.close)

    speechcpp = resolved_settings.backend == "speechcpp"

    @app.get("/healthz")
    async def healthz() -> dict[str, object]:
        return {
            "status": "ok",
            "backend": resolved_settings.backend,
            "voice": resolved_settings.voice,
            "checkpoint": str(
                resolved_settings.speechcpp_model
                if speechcpp
                else resolved_settings.checkpoint or resolved_settings.hf_checkpoint
            ),
            "model_loaded": resolved_synthesizer.model_loaded()
            if hasattr(resolved_synthesizer, "model_loaded")
            else resolved_synthesizer._runtime is not None,
            "watermark": not speechcpp,
            "speechcpp_device": resolved_settings.speechcpp_device if speechcpp else None,
            "speechcpp_gpu_uuid": resolved_synthesizer.runtime_gpu_uuid()
            if speechcpp and hasattr(resolved_synthesizer, "runtime_gpu_uuid")
            else None,
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

    async def _run_synthesizer(function, *args, **kwargs):
        try:
            return await asyncio.to_thread(function, *args, **kwargs)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/v1/audio/speech")
    async def speech(request: SpeechRequest, http_request: Request) -> Response:
        if request.voice != resolved_settings.voice:
            raise HTTPException(status_code=400, detail=f"unsupported voice: {request.voice}")
        if request.response_format not in {"pcm", "wav"}:
            raise HTTPException(status_code=400, detail="response_format must be pcm or wav")
        if request.stream and request.response_format != "pcm":
            raise HTTPException(status_code=400, detail="stream=true requires response_format=pcm")
        stream_headers = {
            "X-Sample-Rate-Hz": str(resolved_settings.response_sample_rate_hz),
            "X-Audio-Channels": "1",
            "X-Audio-Encoding": "pcm_s16le",
        }
        if request.stream and resolved_settings.progressive_stream_enabled:
            # Validate before the 200 headers go out; later failures can only truncate the body.
            ready_pcm, _segments = await _run_synthesizer(
                app.state.synthesizer.plan_progressive_stream,
                request.input,
            )
            if ready_pcm is not None:
                body = iter_pcm_chunks(ready_pcm, chunk_bytes=resolved_settings.stream_chunk_bytes)
            else:
                body = iter_progressive_pcm_response(
                    app.state.synthesizer,
                    request.input,
                    chunk_bytes=resolved_settings.stream_chunk_bytes,
                    lock=app.state.streaming_lock,
                )
            return StreamingResponse(body, media_type="audio/L16", headers=stream_headers)

        response_format = "pcm" if request.stream else request.response_format
        # Silence, the fixed greeting and short-cache hits need no GPU, so they skip the queue.
        audio = await _run_synthesizer(
            app.state.synthesizer.lookup_ready_audio,
            request.input,
            response_format=response_format,
        )
        if audio is None:
            if app.state.synthesis_lock is None:
                app.state.synthesis_lock = asyncio.Lock()
            async with app.state.synthesis_lock:
                # A barge-in closes the client's stream; do not synthesize for a client that left
                # while queued. A synthesis already running cannot be cancelled.
                if await http_request.is_disconnected():
                    return Response(status_code=499)
                audio = await _run_synthesizer(
                    app.state.synthesizer.synthesize,
                    request.input,
                    response_format=response_format,
                )
        content, media_type, sample_rate = audio
        if request.stream:
            return StreamingResponse(
                iter_pcm_chunks(content, chunk_bytes=resolved_settings.stream_chunk_bytes),
                media_type="audio/L16",
                headers=stream_headers,
            )
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
