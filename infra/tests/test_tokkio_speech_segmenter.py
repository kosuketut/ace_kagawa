from __future__ import annotations

import asyncio
import importlib.util
import random
import re
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def install_stub_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    if "." not in name:
        module.__path__ = []
    sys.modules[name] = module
    return module


def load_tokkio_llm_with_stubs():
    class Logger:
        def debug(self, *_args, **_kwargs):
            pass

        def info(self, *_args, **_kwargs):
            pass

        def warning(self, *_args, **_kwargs):
            pass

        def error(self, *_args, **_kwargs):
            pass

    class Frame:
        def __init__(self, content="", **kwargs):
            self.content = content
            self.__dict__.update(kwargs)

    class LLMFullResponseStartFrame(Frame):
        pass

    class LLMFullResponseEndFrame(Frame):
        pass

    class TextFrame(Frame):
        pass

    class TTSSpeakFrame(Frame):
        pass

    class BaseLLMService:
        pass

    install_stub_module("loguru").logger = Logger()
    install_stub_module("nvidia_pipecat")
    install_stub_module("nvidia_pipecat.services")
    install_stub_module("nvidia_pipecat.services.nvidia_llm").NvidiaLLMService = BaseLLMService
    install_stub_module("pipecat")
    install_stub_module("pipecat.frames")
    frames = install_stub_module("pipecat.frames.frames")
    frames.LLMFullResponseEndFrame = LLMFullResponseEndFrame
    frames.LLMFullResponseStartFrame = LLMFullResponseStartFrame
    frames.TextFrame = TextFrame
    frames.TTSSpeakFrame = TTSSpeakFrame
    install_stub_module("pipecat.processors")
    install_stub_module("pipecat.processors.aggregators")
    install_stub_module("pipecat.processors.aggregators.openai_llm_context").OpenAILLMContext = object
    install_stub_module("pipecat.services")
    install_stub_module("pipecat.services.openai")
    install_stub_module("pipecat.services.openai.llm").OpenAILLMService = BaseLLMService

    path = ROOT / "infra" / "tokkio" / "tokkio_llm.py"
    spec = importlib.util.spec_from_file_location("tokkio_llm_segmenter_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["tokkio_llm_segmenter_test"] = module
    spec.loader.exec_module(module)
    return module


tokkio_llm = load_tokkio_llm_with_stubs()


def word_deltas(text: str) -> list[str]:
    """OpenAI-style tokens: each word carries its leading space."""
    return re.findall(r"\s*\S+", text)


def random_deltas(text: str, seed: int) -> list[str]:
    rnd = random.Random(seed)
    deltas, index = [], 0
    while index < len(text):
        size = rnd.randint(1, 4)
        deltas.append(text[index : index + size])
        index += size
    return deltas


def chunkings(text: str) -> list[tuple[str, list[str]]]:
    return [
        ("words", word_deltas(text)),
        ("chars", list(text)),
        *((f"random{seed}", random_deltas(text, seed)) for seed in range(5)),
    ]


def stream(deltas: list[str], **buffer_kwargs) -> list[str]:
    buffer = tokkio_llm.SpeechSegmentBuffer(**buffer_kwargs)
    segments: list[str] = []
    for delta in deltas:
        segments.extend(buffer.feed(delta))
    segments.extend(buffer.flush())
    return segments


def squash_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def scripted(utterance: str):
    return tokkio_llm.get_scripted_speech_reply([{"role": "user", "content": utterance}])


class SpeechSegmentSpacingTests(unittest.TestCase):
    """pipecat's sentence aggregator joins TTS segments with no separator."""

    def assert_spaces_survive_joining(self, text: str) -> None:
        for label, deltas in chunkings(text):
            with self.subTest(text=text, chunking=label):
                self.assertEqual(squash_spaces("".join(stream(deltas))), text)

    def test_english_reply_keeps_spaces_at_every_segment_boundary(self) -> None:
        for text in (
            "Tokyo University of Technology was founded in 1986 and its main campus is in Hachioji, Tokyo. "
            "I am proud of our students.",
            "We work closely with NVIDIA. Our supercomputer Seiran uses DGX B200 systems, "
            "and Dr. Kagawa leads the project.",
            "Hello, I am Yutaka Kagawa, President of Tokyo University of Technology.",
            "I am Yutaka Kagawa, the President of Tokyo University of Technology, and my research focuses "
            "on ceramic matrix composites and their interfacial mechanics.",
            "We have about 10,000 students across six schools.",
            "Is this NVIDIA? Yes, it is! We love AI; it helps everyone.",
            "Dr. Smith visited the U.S. in 2024 and gave a talk at 3 p.m. on campus.",
            "Our supercomputer, Seiran, uses NVIDIA DGX B200 systems with about 1.5 exaflops of AI performance.",
        ):
            self.assert_spaces_survive_joining(text)

    def test_spaced_ascii_terms_inside_japanese_keep_their_spaces(self) -> None:
        for text in (
            "青嵐ではNVIDIAのスーパーコンピュータNVIDIA DGX B200 Systemsを導入しています。",
            "本学はNVIDIA AI Technology Center Joint Labを設立しました。",
            "現在はIrodori-TTS v4.1-Smallを使っています。",
        ):
            self.assert_spaces_survive_joining(text)

    def test_space_after_a_non_ascii_symbol_survives_before_an_english_word(self) -> None:
        for text in (
            "Hachioji is our main campus — and Kamata is our second campus in Tokyo.",
            "Our new supercomputer is called Seiran — the Blue Gale — and it is very fast.",
            "Seiran is our new supercomputer, and it helps our students’ research a great deal.",
            "Well… I think our students are wonderful, and they work hard every day at Hachioji.",
            "He said “Thank you.” Then we started the symposium together with NVIDIA.",
        ):
            self.assert_spaces_survive_joining(text)

    def test_space_after_a_closing_quote_stays_on_the_quoted_sentence(self) -> None:
        self.assertEqual(
            stream(word_deltas("He said “Thank you.” Then we started.")),
            ["He said “Thank you.” ", "Then we started."],
        )

    def test_space_is_kept_on_the_segment_cut_at_a_space(self) -> None:
        buffer = tokkio_llm.SpeechSegmentBuffer()

        self.assertEqual(buffer.feed("Hello, I am Yutaka Kagawa,"), [])
        self.assertEqual(buffer.feed(" President of Tokyo"), ["Hello, I am Yutaka Kagawa, President of "])
        self.assertEqual(buffer.flush(), ["Tokyo"])

    def test_ascii_comma_at_the_end_of_the_buffer_waits_for_the_next_delta(self) -> None:
        buffer = tokkio_llm.SpeechSegmentBuffer()

        self.assertEqual(buffer.feed("Our joint lab is the NVIDIA-AI-Technology-Center"), ["Our joint lab is the "])
        # Whether a space follows the comma is only known from the next delta.
        self.assertEqual(buffer.feed(","), [])
        self.assertEqual(buffer.feed(" and Seiran is"), ["NVIDIA-AI-Technology-Center, and Seiran "])
        self.assertEqual(buffer.flush(), ["is"])

    def test_japanese_segments_get_no_extra_spaces(self) -> None:
        text = (
            "私の専門は材料強度学で、特にセラミックス複合材料、いわゆるCMCの界面力学特性や非破壊評価を中心に、"
            "耐熱構造材料の研究を長年続けてきました。東京工科大学は1986年に開学しました。"
        )
        for label, deltas in chunkings(text):
            with self.subTest(chunking=label):
                segments = stream(deltas)
                self.assertEqual("".join(segments), text)
                for segment in segments:
                    self.assertEqual(segment, segment.strip())


class SpeechSegmentBoundaryTests(unittest.TestCase):
    def test_closing_bracket_after_sentence_end_stays_with_its_sentence(self) -> None:
        for text in (
            "私は「AIを使いこなす力が大切です。」と学生に伝えています。",
            "学生には「失敗を恐れるな。」と言っています。",
            "（詳しくは大学案内をご覧ください。）次に、学費についてです。",
        ):
            for label, deltas in chunkings(text):
                with self.subTest(text=text, chunking=label):
                    segments = stream(deltas)
                    self.assertEqual("".join(segments), text)
                    for segment in segments:
                        self.assertNotRegex(segment, r"^[」』）)】]")

    def test_sentence_end_at_the_end_of_the_buffer_waits_for_the_next_delta(self) -> None:
        buffer = tokkio_llm.SpeechSegmentBuffer()

        self.assertEqual(buffer.feed("学生には「失敗を恐れるな。"), [])
        self.assertEqual(
            buffer.feed("」と言っています。次"),
            ["学生には「失敗を恐れるな。」", "と言っています。"],
        )
        self.assertEqual(buffer.flush(), ["次"])

    def test_flush_complete_emits_a_held_sentence_and_drops_a_fragment(self) -> None:
        buffer = tokkio_llm.SpeechSegmentBuffer()
        self.assertEqual(buffer.feed("学長は「挑戦を楽しんで。」"), [])
        self.assertEqual(buffer.flush_complete(), ["学長は「挑戦を楽しんで。」"])
        self.assertEqual(buffer.flush(), [])

        buffer = tokkio_llm.SpeechSegmentBuffer()
        self.assertEqual(buffer.feed("東京工科大学は八王子にあります。学部は"), ["東京工科大学は八王子にあります。"])
        self.assertEqual(buffer.flush_complete(), [])
        # The dropped fragment must not leak into a later flush.
        self.assertEqual(buffer.flush(), [])

    def test_ascii_ellipsis_is_one_pause_not_a_sentence_end(self) -> None:
        for label, deltas in chunkings("そうですね...少し考えます。"):
            with self.subTest(chunking=label):
                self.assertEqual(stream(deltas), ["そうですね…少し考えます。"])
        for label, deltas in chunkings("Well... I think so."):
            with self.subTest(chunking=label):
                self.assertEqual(squash_spaces("".join(stream(deltas))), "Well… I think so.")


class NormalizeSpeechSegmentTests(unittest.TestCase):
    def test_meta_note_filter_removes_leaked_prompt_constraints(self) -> None:
        for note in ("（100文字以内、2文）", "（40から60文字）", "（回答は60文字で）", "（2文で説明します）", "(60文字以内)"):
            with self.subTest(note=note):
                self.assertEqual(
                    tokkio_llm.normalize_speech_segment(f"本学は八王子にあります{note}。"),
                    "本学は八王子にあります。",
                )

    def test_meta_note_filter_keeps_factual_parentheticals(self) -> None:
        for text in (
            "最寄り駅から（徒歩10分以内）です。",
            "審査結果は（出願後2週間以内）に通知されます。",
            "学長挨拶（3分以内）をお願いします。",
            "受験料（30日以内に納付）が必要です。",
            "論文は（A4で10枚以内）にまとめてください。",
            "成果は（論文で発表）しています。",
            "パスワードは（大文字と小文字を区別します）。",
            "書類は（全3文書）を提出します。",
            "パスワードは（8文字以上）にしてください。",
        ):
            with self.subTest(text=text):
                self.assertEqual(tokkio_llm.normalize_speech_segment(text), text)

    def test_meta_note_filter_removes_bare_character_count_leaks(self) -> None:
        for note in ("（50文字）", "（約58文字）", "（58文字、1文）", "（全角52文字）", "（40〜60文字程度）", "（60文字程度）"):
            with self.subTest(note=note):
                self.assertEqual(
                    tokkio_llm.normalize_speech_segment(f"本学は八王子にあります{note}。"),
                    "本学は八王子にあります。",
                )

    def test_meta_note_filter_keeps_character_counts_that_are_content(self) -> None:
        for text in (
            "パスワードは（半角英数字8文字）で設定します。",
            "ニックネーム（最大20文字）を登録します。",
            "ID（英字1文字＋数字7桁）を入力します。",
            "TLA（3文字略語）とは何か。",
        ):
            with self.subTest(text=text):
                self.assertEqual(tokkio_llm.normalize_speech_segment(text), text)

    def test_black_triangle_before_a_digit_is_kept_as_a_minus_sign(self) -> None:
        # The TTS server reads "▲5" as マイナスご; a decorative ▲ is still removed.
        for text, expected in (
            ("前年比▲5%の減少です。", "前年比▲5%の減少です。"),
            ("収支は▲120万円です。", "収支は▲120万円です。"),
            ("▲印の項目は必須です。", "印の項目は必須です。"),
            ("▲ 重要なお知らせです。", "重要なお知らせです。"),
            ("△印の項目は任意です。", "印の項目は任意です。"),
        ):
            with self.subTest(text=text):
                self.assertEqual(tokkio_llm.normalize_speech_segment(text), expected)

    def test_signs_and_symbols_keep_their_meaning(self) -> None:
        for text, expected in (
            ("-40℃でも使えます。", "-40℃でも使えます。"),
            ("AI + DXを推進しています。", "AIプラスDXを推進しています。"),
            ("1 + 1 は 2 です。", "1プラス1 は 2 です。"),
            ("A→Bの順です。", "A、Bの順です。"),
            ("八王子 → 蒲田の順です。", "八王子、蒲田の順です。"),
            ("入力⇒出力です。", "入力、出力です。"),
            ("研究室は ○○棟にあります。", "研究室は ○○棟にあります。"),
            ("○×で答えてください。", "○×で答えてください。"),
        ):
            with self.subTest(text=text):
                self.assertEqual(tokkio_llm.normalize_speech_segment(text), expected)

    def test_markdown_bullets_and_emoji_are_still_removed(self) -> None:
        for text, expected in (
            ("- 材料強度学", "材料強度学"),
            ("* 複合材料", "複合材料"),
            ("+ 非破壊評価", "非破壊評価"),
            ("項目\n- 材料強度学\n- 複合材料", "項目。材料強度学。複合材料"),
            ("社会情報専攻:2名 *試験日:9月27日", "社会情報専攻:2名。試験日:9月27日"),
            ("ようこそ😀東京工科大学へ。", "ようこそ東京工科大学へ。"),
        ):
            with self.subTest(text=text):
                self.assertEqual(tokkio_llm.normalize_speech_segment(text), expected)

    def test_ascii_ellipsis_becomes_one_ellipsis(self) -> None:
        self.assertEqual(tokkio_llm.normalize_speech_segment("そうですね...少し考えます。"), "そうですね…少し考えます。")


class ScriptedCueTests(unittest.TestCase):
    def test_documented_symposium_cues_start_the_opening(self) -> None:
        for utterance in (
            "シンポジウムのオープニングをお願いします。",
            "シンポジウムの開会挨拶をお願いします。",
            "シンポジウムの台本を読んでください。",
            "Start.",
            "start",
            "Please read the symposium opening.",
            "Read the symposium opening.",
            "Please start the symposium.",
        ):
            with self.subTest(utterance=utterance):
                self.assertEqual(scripted(utterance), tokkio_llm.SYMPOSIUM_OPENING_SCRIPT)

    def test_questions_about_the_symposium_do_not_start_the_opening(self) -> None:
        for utterance in (
            "What time does the symposium start?",
            "When did the symposium start?",
            "Has the symposium already begun?",
            "Has the symposium already started?",
            "Is the symposium ready?",
            "Where can I read more about the symposium?",
            "シンポジウムの説明をお願いします",
            "シンポジウムの日程を教えてください。お願いします。",
            "シンポジウムについて話してください、お願いします",
            "シンポジウムはいつですか？",
        ):
            with self.subTest(utterance=utterance):
                self.assertIsNone(scripted(utterance))

    def test_documented_open_campus_cues_still_select_their_scripts(self) -> None:
        for utterance in (
            "オープンキャンパスのご挨拶をお願いします。",
            "オープニングをお願いします。",
            "台本を読んでください。",
            "挨拶をお願いします。",
            "学長挨拶をお願いします。",
            "八王子キャンパスのオープニングをお願いします。",
        ):
            with self.subTest(utterance=utterance):
                self.assertEqual(scripted(utterance), tokkio_llm.OPEN_CAMPUS_GREETING_SCRIPT)
        for utterance in (
            "オープンキャンパスの締めの言葉をお願いします。",
            "締めの言葉をお願いします。",
            "クロージングをお願いします。",
            "締めの台本を読んでください。",
        ):
            with self.subTest(utterance=utterance):
                self.assertEqual(scripted(utterance), tokkio_llm.OPEN_CAMPUS_CLOSING_SCRIPT)

    def test_open_campus_questions_do_not_select_a_script(self) -> None:
        for utterance in (
            "オープンキャンパスの日程を教えてください、お願いします",
            "オープンキャンパスはいつですか？",
            "オープンキャンパスについて話してください",
            "八王子キャンパスへの行き方を教えてください、お願いします",
            "締め切りはいつですか、教えてください、お願いします",
        ):
            with self.subTest(utterance=utterance):
                self.assertIsNone(scripted(utterance))

    def test_cues_with_a_lead_in_or_spelling_variant_select_their_script(self) -> None:
        greeting = tokkio_llm.OPEN_CAMPUS_GREETING_SCRIPT
        closing = tokkio_llm.OPEN_CAMPUS_CLOSING_SCRIPT
        symposium = tokkio_llm.SYMPOSIUM_OPENING_SCRIPT
        for utterance, expected in (
            ("オープンキャンパスのごあいさつをお願いします", greeting),
            ("台本を読んで下さい", greeting),
            ("オープンキャンパスのご挨拶をお願い致します", greeting),
            ("オープニングをおねがいします", greeting),
            ("締めの言葉をおねがいします", closing),
            ("それでは、オープンキャンパスのご挨拶をお願いします。", greeting),
            ("香川先生、オープニングをお願いします。", greeting),
            ("最後に締めの言葉をお願いします。", closing),
            ("Start.", symposium),
            ("Please read the symposium opening.", symposium),
            ("オープンキャンパスのご挨拶をお願いします。", greeting),
            ("オープニングをお願いします。", greeting),
            ("台本を読んでください。", greeting),
            ("では台本を読んでください。", greeting),
            ("オープンキャンパスの御挨拶をお願いします。", greeting),
            ("オープンキャンパスのあいさつをお願いします。", greeting),
            ("オープンキャンパスのご挨拶お願いします", greeting),
            ("台本をよんでください", greeting),
            ("学長、ご挨拶をお願いします。", greeting),
            ("では、締めの言葉をお願いします。", closing),
            ("それではシンポジウムのオープニングをお願いします。", symposium),
            ("シンポジウムのオープニングをおねがいします", symposium),
            ("Let's start the symposium.", symposium),
            ("Okay, start the symposium.", symposium),
            ("Now, please read the symposium opening.", symposium),
        ):
            with self.subTest(utterance=utterance):
                self.assertEqual(scripted(utterance), expected)

    def test_trailing_asr_filler_does_not_hide_a_cue(self) -> None:
        for utterance, expected in (
            ("オープンキャンパスのご挨拶をお願いします。えー", tokkio_llm.OPEN_CAMPUS_GREETING_SCRIPT),
            ("締めの言葉をお願いします、えっと", tokkio_llm.OPEN_CAMPUS_CLOSING_SCRIPT),
            ("シンポジウムのオープニングをお願いします。あー、えー。", tokkio_llm.SYMPOSIUM_OPENING_SCRIPT),
            # A leading filler is ignored too.
            ("えー、オープンキャンパスのご挨拶をお願いします。", tokkio_llm.OPEN_CAMPUS_GREETING_SCRIPT),
            ("あのー、それでは締めの言葉をお願いします", tokkio_llm.OPEN_CAMPUS_CLOSING_SCRIPT),
            ("えっと、学長、ご挨拶をお願いします。", tokkio_llm.OPEN_CAMPUS_GREETING_SCRIPT),
        ):
            with self.subTest(utterance=utterance):
                self.assertEqual(scripted(utterance), expected)
        for utterance in ("えー", "オープンキャンパスはいつですか、えー", "えー、オープンキャンパスはいつですか"):
            with self.subTest(utterance=utterance):
                self.assertIsNone(scripted(utterance))

    def test_lead_in_and_spelling_variants_do_not_turn_questions_into_cues(self) -> None:
        for utterance in (
            # 最後に is a closing lead-in only; it must not pick the greeting.
            "最後にご挨拶をお願いします。",
            "締め切りを教えてください",
            "台本はどこですか",
            "挨拶はいつですか",
        ):
            with self.subTest(utterance=utterance):
                self.assertIsNone(scripted(utterance))


class OperatorStopCueTests(unittest.TestCase):
    def test_stop_variants_are_operator_stop_cues(self) -> None:
        for utterance in (
            "Stop.",
            "Please stop.",
            "Stop speaking.",
            "Stop it.",
            "Okay, stop.",
            "Stop, please.",
            "stop stop",
            "ストップ。",
            "ストップ、ストップ",
            "ストップしてください",
            "止めて",
            "止めてください",
            "ちょっと止めて",
            "停止",
        ):
            with self.subTest(utterance=utterance):
                self.assertTrue(tokkio_llm._is_operator_stop_cue(tokkio_llm._normalize_short_utterance(utterance)))

    def test_sentences_that_only_mention_stopping_are_not_stop_cues(self) -> None:
        for utterance in (
            "Has it stopped?",
            "Don't stop.",
            "Bus stop?",
            "Non-stop?",
            "Where is the bus stop?",
            "停止しますか",
        ):
            with self.subTest(utterance=utterance):
                self.assertFalse(tokkio_llm._is_operator_stop_cue(tokkio_llm._normalize_short_utterance(utterance)))

    def test_stop_variant_interrupts_without_reply_or_llm(self) -> None:
        class Harness(tokkio_llm.TokkioLLMServiceMixin):
            async def start_ttfb_metrics(self) -> None:
                pass

            async def stop_ttfb_metrics(self) -> None:
                pass

            async def push_frame(self, _frame) -> None:
                raise AssertionError("stop must not produce speech")

        async def unexpected_stream(_context):
            raise AssertionError("stop must not call the LLM")

        context = types.SimpleNamespace(get_messages=lambda: [{"role": "user", "content": "Okay, stop."}])
        asyncio.run(Harness()._process_context_common(context, unexpected_stream))


class LLMStreamErrorTests(unittest.TestCase):
    """A stream that fails mid-response still speaks the sentence the buffer held."""

    def run_failing_stream(self, deltas: list[str], error: BaseException, *, open_error: bool = False):
        frames: list = []

        class Harness(tokkio_llm.TokkioLLMServiceMixin):
            filler: list[str] = []
            time_delay = 60.0
            first_response_timeout_s = 5.0

            async def start_ttfb_metrics(self) -> None:
                pass

            async def stop_ttfb_metrics(self) -> None:
                pass

            async def push_frame(self, frame) -> None:
                frames.append(frame)

        async def chunks():
            for delta in deltas:
                yield types.SimpleNamespace(content=delta)
            raise error

        async def stream_chat_completions(_context):
            if open_error:
                raise error
            return chunks()

        context = types.SimpleNamespace(
            get_messages=lambda: [{"role": "user", "content": "東京工科大学はどこにありますか？"}]
        )
        try:
            asyncio.run(Harness()._process_context_common(context, stream_chat_completions))
        finally:
            self.frames = [(type(frame).__name__, frame.content) for frame in frames]
        return self.frames

    def test_held_sentence_is_spoken_before_the_error_reply(self) -> None:
        self.assertEqual(
            self.run_failing_stream(["東京工科大学は", "八王子にあります。"], RuntimeError("connection reset by peer")),
            [
                ("LLMFullResponseStartFrame", ""),
                ("TextFrame", "東京工科大学は八王子にあります。"),
                ("TTSSpeakFrame", tokkio_llm._LLM_ERROR_REPLY),
                ("LLMFullResponseEndFrame", ""),
            ],
        )

    def test_held_quoted_sentence_is_spoken_whole(self) -> None:
        frames = self.run_failing_stream(["学長は「挑戦を", "楽しんで。」"], RuntimeError("connection reset by peer"))
        self.assertEqual(frames[1], ("TextFrame", "学長は「挑戦を楽しんで。」"))

    def test_unfinished_fragment_is_not_spoken(self) -> None:
        self.assertEqual(
            self.run_failing_stream(
                ["東京工科大学は", "八王子にあります。", "学部は"], RuntimeError("connection reset by peer")
            ),
            [
                ("LLMFullResponseStartFrame", ""),
                ("TextFrame", "東京工科大学は八王子にあります。"),
                ("TTSSpeakFrame", tokkio_llm._LLM_ERROR_REPLY),
                ("LLMFullResponseEndFrame", ""),
            ],
        )

    def test_failure_to_open_the_stream_only_speaks_the_error_reply(self) -> None:
        self.assertEqual(
            self.run_failing_stream([], RuntimeError("connection refused"), open_error=True),
            [("TTSSpeakFrame", tokkio_llm._LLM_ERROR_REPLY)],
        )

    def test_barge_in_does_not_speak_the_held_sentence(self) -> None:
        with self.assertRaises(asyncio.CancelledError):
            self.run_failing_stream(["東京工科大学は", "八王子にあります。"], asyncio.CancelledError())
        self.assertEqual(self.frames, [("LLMFullResponseStartFrame", ""), ("LLMFullResponseEndFrame", "")])


if __name__ == "__main__":
    unittest.main()
