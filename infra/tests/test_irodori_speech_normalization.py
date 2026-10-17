from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


irodori_server = load_module("irodori_server", ROOT / "infra" / "tts" / "irodori_server.py")


class IrodoriSpeechNormalizationTests(unittest.TestCase):
    def assert_normalized(self, cases: dict[str, str]) -> None:
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(irodori_server.normalize_speech_input(text), expected)

    def test_clock_times_ignore_zero_padding(self) -> None:
        self.assert_normalized(
            {
                "10:05": "じゅうじごふん",
                "09:00": "くじ",
                "07:00": "しちじ",
                "04:00": "よじ",
                "08:30": "はちじさんじゅっぷん",
                "12:06": "じゅうにじろっぷん",
                "19:09": "じゅうくじきゅうふん",
                "9:05〜10:05": "くじごふんからじゅうじごふんまで",
                "09:30〜10:00": "くじさんじゅっぷんからじゅうじまで",
                "9時05分": "くじごふん",
                "9時00分": "くじ",
                "17:43:57": "じゅうしちじよんじゅうさんぷんごじゅうななびょう",
                "05分です。": "ごふんです。",
                # Codes keep the digit-by-digit reading.
                "0120です。": "ぜろいちにぜろです。",
            }
        )

    def test_time_ranges_do_not_double_particles(self) -> None:
        self.assert_normalized(
            {
                "10:00〜12:00まで開催": "じゅうじからじゅうにじまで開催",
                "10:00〜12:00に開催": "じゅうじからじゅうにじに開催",
                "10:00〜12:00の間です。": "じゅうじからじゅうにじまでの間です。",
                "9:30〜からです": "くじさんじゅっぷんからです",
                "10:00–12:00": "じゅうじからじゅうにじまで",
                "10:00—12:00": "じゅうじからじゅうにじまで",
            }
        )

    def test_decimals_and_versions_geminate_before_ten(self) -> None:
        self.assert_normalized(
            {
                "1.5倍": "いってんご倍",
                "8.5点": "はってんご点",
                "10.5": "じゅってんご",
                "11.5": "じゅういってんご",
                "20.5": "にじゅってんご",
                "1.5%": "いってんごぱーせんと",
                "1.5km": "いってんごキロメートル",
                "1.2.3": "いってんにてんさん",
                "CUDA 13.2.1.1": "CUDA じゅうさんてんにてんいってんいち",
                # ろく, ひゃく and れい stay as they are.
                "6.5": "ろくてんご",
                "100.5": "ひゃくてんご",
                "0.5": "れいてんご",
            }
        )

    def test_counters_with_sound_changes(self) -> None:
        self.assert_normalized(
            {
                "21世紀": "にじゅういっせいき",
                "1種類、8種類、10種類": "いっしゅるい、はっしゅるい、じゅっしゅるい",
                "1か国、10か国、6カ国、1ヶ国": "いっかこく、じゅっかこく、ろっかこく、いっかこく",
                "1か国語": "いっかこく語",
                "1軒、3軒、10軒": "いっけん、さんげん、じゅっけん",
                "1軒目": "いっけん目",
                "1頭、6頭、10頭": "いっとう、ろくとう、じゅっとう",
                "1足、3足": "いっそく、さんぞく",
                "1着": "いっちゃく",
                "1歩、3歩、4歩、10歩": "いっぽ、さんぽ、よんほ、じゅっぽ",
                "1曲": "いっきょく",
                "1社、10社": "いっしゃ、じゅっしゃ",
                "1発、3発": "いっぱつ、さんぱつ",
                "1室、1席、1章、1試合、1周、1缶": (
                    "いっしつ、いっせき、いっしょう、いっしあい、いっしゅう、いっかん"
                ),
                "1箱、2通り": "ひとはこ、ふたとおり",
                # Compounds that only start with a counter kanji are left alone.
                "1足す1は2": "いち足すいちはに",
                "3発表": "さん発表",
                "1社会": "いち社会",
            }
        )

    def test_h_row_counters_after_thousands_and_zero(self) -> None:
        self.assert_normalized(
            {
                "1000本": "せんぼん",
                "3000本": "さんぜんぼん",
                "10000本": "いちまんぼん",
                "1000匹": "せんびき",
                "1000杯": "せんばい",
                "0本": "ぜろほん",
                "0分": "ぜろふん",
                "0泊": "ぜろはく",
                "1000分": "せんぷん",
                "100本": "ひゃっぽん",
                "4本": "よんほん",
            }
        )

    def test_trillions_and_kei_geminate(self) -> None:
        self.assert_normalized(
            {
                "1兆": "いっちょう",
                "8兆円": "はっちょう円",
                "10兆円": "じゅっちょう円",
                "100兆": "ひゃくちょう",
                "1兆2000億円": "いっちょうにせん億円",
                "1,000,000,000,000": "いっちょう",
                "8000000000000": "はっちょう",
                "10000000000000000": "いっけい",
                "90京回": "きゅうじゅっけい回",
                "100京": "ひゃっけい",
                "第3京浜": "だいさん京浜",
                "1000万円と1億円です。": "いっせん万円といち億円です。",
            }
        )

    def test_signs_are_spoken(self) -> None:
        self.assert_normalized(
            {
                "-5%": "マイナスごぱーせんと",
                "-1.5": "マイナスいってんご",
                "-5℃まで下がります": "マイナスごどまで下がります",
                "気温は-3℃": "気温はマイナスさんど",
                "−3℃": "マイナスさんど",
                "－5": "マイナスご",
                "▲5": "マイナスご",
                "前年比 +5%": "前年比 プラスごぱーせんと",
                "AI + DX": "AIプラスDX",
                "1 + 1は2": "いちプラスいちはに",
                "C++17": "C++じゅうなな",
                "第3 -5章": "だいさんしょうからごしょう",
                "電話は042-637-2011です。": "電話はぜろよんにのろくさんななのにぜろいちいちです。",
                "- 項目です": "項目です",
            }
        )

    def test_tens_geminate_before_voiceless_units(self) -> None:
        self.assert_normalized(
            {
                "10%": "じゅっぱーせんと",
                "40%": "よんじゅっぱーせんと",
                "10〜20%": "じゅっぱーせんとからにじゅっぱーせんと",
                "10パーセント": "じゅっぱーせんと",
                "10km": "じゅっキロメートル",
                "10cm": "じゅっセンチメートル",
                "10kg": "じゅっキログラム",
                "10TB": "じゅっテラバイト",
                "10PFLOPS": "じゅっペタフロップス",
                "10ページ、10ポイント、10セット": "じゅっページ、じゅっポイント、じゅっセット",
                # Voiced or nasal initials keep じゅう.
                "10GB": "じゅうギガバイト",
                "10m": "じゅうメートル",
                "10ドル": "じゅうドル",
                "100%": "ひゃくぱーせんと",
                # Four-digit numbers are mostly years or model numbers.
                "2010シーズン": "にせんじゅうシーズン",
            }
        )

    def test_numbers_attached_to_letters(self) -> None:
        self.assert_normalized(
            {
                "v4.1": "vよんてんいち",
                "Irodori v4.1-Small": "Irodori vよんてんいち-Small",
                "GPA3.5": "GPAさんてんご",
                "Qwen2.5-7B": "Qwenにてんご-7B",
                "Ver.2.1": "Ver.にてんいち",
                "Web3.0": "Webさんてんぜろ",
                "GPT-5.1": "GPT-ごてんいち",
                "Python 3.12": "Python さんてんいちに",
                # Mixed tokens without a known unit stay intact instead of half-converted.
                "No.1の大学です。": "No.1の大学です。",
                "OpenMPI 4.1.9a1": "OpenMPI 4.1.9a1",
                "15.6inch": "15.6inch",
            }
        )

    def test_iso_dates_postal_codes_addresses_and_hyphen_ranges(self) -> None:
        self.assert_normalized(
            {
                "2026-07-19の東京工科大学": "にせんにじゅうろくねんしちがつじゅうくにちの東京工科大学",
                "2026-07-19時点": "にせんにじゅうろくねんしちがつじゅうくにち時点",
                "〒192-0982": "ゆうびんばんごういちきゅうにのぜろきゅうはちに",
                "〒144-8535": "ゆうびんばんごういちよんよんのはちごさんご",
                "片倉町1404-1": "片倉町せんよんひゃくよんのいち",
                "片倉町1404- 1": "片倉町せんよんひゃくよんのいち",
                "1丁目2-3": "いっちょうめにのさん",
                "2-3人": "ふたりからさんにん",
                "10-20名": "じゅうめいからにじゅうめい",
                "1-3月": "いちがつからさんがつ",
                "第1-3回": "だいいっかいからさんかい",
                "2026-2027年度": "にせんにじゅうろくねんからにせんにじゅうななねんど",
            }
        )
        self.assertNotIn("がつ", irodori_server.normalize_speech_input("2026-07-19T00:00:00"))

    def test_block_numbers_after_a_place_name_are_joined_with_no(self) -> None:
        self.assert_normalized(
            {
                # Both spellings of the Kamata campus address in the RAG corpus.
                "〒144-8535 東京都大田区西蒲田5-23-22": (
                    "ゆうびんばんごういちよんよんのはちごさんご 東京都大田区西蒲田ごのにじゅうさんのにじゅうに"
                ),
                "蒲田駅 〒144-8535 東京都大田区西蒲田 5 -23-22": (
                    "蒲田駅 ゆうびんばんごういちよんよんのはちごさんご 東京都大田区西蒲田ごのにじゅうさんのにじゅうに"
                ),
                "西新宿1-2-3です。": "西新宿いちのにのさんです。",
                # Two numbers after a word are still a range, and phone numbers keep their digits.
                "第3-5章": "だいさんしょうからごしょう",
                "電話03-1234-5678": "電話ぜろさんのいちにさんよんのごろくななはち",
            }
        )

    def test_unit_ranges_with_a_hyphen_are_not_minus(self) -> None:
        self.assert_normalized(
            {
                "受付期間は4月1日-4月30日です。": "受付期間はしがつ ついたちからしがつさんじゅうにちです。",
                "平日9時-17時です。": "平日くじからじゅうしちじです。",
                "9時－17時": "くじからじゅうしちじ",
                "9時半-17時": "くじ半からじゅうしちじ",
                "定員は100名-150名です。": "定員はひゃくめいからひゃくごじゅうめいです。",
                "3人-5人": "さんにんからごにん",
                "1年生-3年生": "いちねん生からさんねん生",
                "3,000円-5,000円": "さんぜんえんからごせんえん",
                "2025年-2026年": "にせんにじゅうごねんからにせんにじゅうろくねん",
                "2026年度-2027年度": "にせんにじゅうろくねんどからにせんにじゅうななねんど",
                "気温は5℃-10℃です。": "気温はごどからじゅうどです。",
                "出願期間は9月1日(火)-9月7日(月)です。": (
                    "出願期間はくがつ ついたち、火曜日、からくがつなのか、月曜日、です。"
                ),
                # A sign after a word is still spoken.
                "前年比-3%": "前年比マイナスさんぱーせんと",
                "前年比▲3%": "前年比マイナスさんぱーせんと",
            }
        )

    def test_numbered_labels_with_a_colon_are_not_clock_times(self) -> None:
        self.assert_normalized(
            {
                "ポイント1：4年間の一貫教育です。": "ポイントいち、よねんかんの一貫教育です。",
                "ポイント1:4年間": "ポイントいち、よねんかん",
                "質問1：10月の説明会について": "質問いち、じゅうがつの説明会について",
                "ステップ2：10分間の休憩": "ステップに、じゅっぷん間の休憩",
                "その1：30名の定員": "そのいち、さんじゅうめいの定員",
                "その1：10〜12月": "そのいち、じゅうがつからじゅうにがつ",
                "その1:00の回": "そのいちじの回",
                "問題1:30分で解く": "問題いち、さんじゅっぷんで解く",
                # Clock times followed by a word or 時 stay times.
                "10：30開始": "じゅうじさんじゅっぷん開始",
                "10:30時点で": "じゅうじさんじゅっぷん時点で",
                "10:30〜12時": "じゅうじさんじゅっぷんからじゅうにじ",
                "10:30日本時間": "じゅうじさんじゅっぷん日本時間",
                "9:00〜17:00月〜金": "くじからじゅうしちじまで月から金",
                "10:30校門前に集合": "じゅうじさんじゅっぷん校門前に集合",
                "16:9": "じゅうろくたいきゅう",
                # A :00/:30 time followed by an ordinary word stays a time.
                "営業時間は9:00〜17:00年中無休です。": "営業時間はくじからじゅうしちじまで年中無休です。",
                "10:00〜17:00月火休み": "じゅうじからじゅうしちじまで月火休み",
                "10:00〜18:00日祝休": "じゅうじからじゅうはちじまで日祝休",
                "平日9:00〜17:00月から金": "平日くじからじゅうしちじまで月から金",
                "受付は10:00〜12:00月水金": "受付はじゅうじからじゅうにじまで月水金",
                "13:00分科会Aが始まります。": "じゅうさんじ分科会Aが始まります。",
                "15:00回収予定": "じゅうごじ回収予定",
                "午前10:00日程説明": "午前じゅうじ日程説明",
                # Only a label word makes the colon a separator.
                "その10:00の回": "そのじゅうじの回",
                "質問1：10時から": "質問いち、じゅうじから",
                # A time that starts a date range (corpus: インターネット出願登録).
                "9月1日(火)10:00〜9月7日(月)15:00": (
                    "くがつ ついたち、火曜日、じゅうじからくがつなのか、月曜日、じゅうごじ"
                ),
            }
        )

    def test_full_width_ascii_is_folded(self) -> None:
        self.assert_normalized(
            {
                "10：30": "じゅうじさんじゅっぷん",
                "１０：００〜１２：００": "じゅうじからじゅうにじまで",
                "２０２６／４／１": "にせんにじゅうろくねんしがつ ついたち",
                "０４２－６３７－２１１１": "ぜろよんにのろくさんななのにいちいちいち",
                "ＴＯＰ５００": "トップごひゃく",
                "ＵＳＪ": "ユーエスジェイ",
                "青嵐（ＳＥＩＲＡＮ）": "せいらん",
                "５Ｇ": "5G",
                "１０ｋｍ": "じゅっキロメートル",
                "10㎞": "じゅっキロメートル",
                # Ellipses and circled numbers are left for Irodori's own normalizer.
                "そうですね……。": "そうですね……。",
                "①工学部": "①工学部",
                # ［］ stay full width, so a bracketed label is not mistaken for a Markdown link.
                "［重要］（必ずお読みください）": "［重要］、必ずお読みください、",
                "[詳細](https://example.com)": "詳細",
            }
        )

    def test_commas_between_small_numbers_are_pauses(self) -> None:
        self.assert_normalized(
            {
                "2，3人": "に、さんにん",
                "2,3人": "に、さんにん",
                "1，2年生": "いち、にねん生",
                "1,2,3": "いち、に、さん",
                "100, 200, 300": "ひゃく、にひゃく、さんびゃく",
                "学生は100，教員は20": "学生はひゃく、教員はにじゅう",
                "100,200": "じゅうまんにひゃく",
                "1,000人": "せんにん",
            }
        )

    def test_spaced_and_slash_dates(self) -> None:
        self.assert_normalized(
            {
                "9 月18日": "くがつじゅうはちにち",
                "4 月 1 日": "しがつ ついたち",
                "7 月": "しちがつ",
                "6/ 7 日": "ろくがつなのか",
                "2027年 3 月31日": "にせんにじゅうななねんさんがつさんじゅういちにち",
                "Windows 11 日本語版": "Windows じゅういち 日本語版",
                "365日": "さんびゃくろくじゅうごにち",
                "6/14日": "ろくがつじゅうよっか",
                "8/2日": "はちがつふつか",
                "試験日:9/27": "試験日は、くがつにじゅうななにち",
                "6/1日曜日に": "ろくがつ ついたち日曜日に",
            }
        )

    def test_nth_weekday_is_not_a_month(self) -> None:
        self.assert_normalized(
            {
                "毎月第1月曜日": "毎月だいいち月曜日",
                "第2月曜日に開催": "だいに月曜日に開催",
                # 第 before a number is spelled だい: the model garbled the mixed
                # 第いち月曜日 (1/3 transcribed correctly vs 3/3; 2026-10-08).
                "第1回目の説明会": "だいいっかい目の説明会",
                "次第3つ目": "次第みっつ目",
                "1月1日": "いちがつ ついたち",
                "10月から": "じゅうがつから",
            }
        )

    def test_day_readings(self) -> None:
        self.assert_normalized(
            {
                "9月19日": "くがつじゅうくにち",
                "2月29日": "にがつにじゅうくにち",
                "19日間": "じゅうくにちかん",
                "17日": "じゅうななにち",
                "私は1952年9月19日生まれです。": "私はせんきゅうひゃくごじゅうにねんくがつじゅうくにち生まれです。",
                "1日（土）と2日（日）": "ついたち、土曜日、とふつか、日曜日、",
                "今月の1日": "今月のついたち",
                "1日に2回": "いちにちににかい",
                "1日〜3日かかります": "いちにちからみっかかかります",
            }
        )

    def test_month_before_tsuitachi_is_spaced(self) -> None:
        # "はちがつついたち" made the model slur the repeated つ (8/12 takes
        # transcribed correctly vs 6/6 with the space; 2026-10-08 benchmark).
        self.assert_normalized(
            {
                "8月1日": "はちがつ ついたち",
                "2026年8月1日（土）": "にせんにじゅうろくねんはちがつ ついたち、土曜日、",
                "2026-08-01の": "にせんにじゅうろくねんはちがつ ついたちの",
                "8/1": "はちがつ ついたち",
                "8月2日": "はちがつふつか",
            }
        )

    def test_ratios_and_colons(self) -> None:
        self.assert_normalized(
            {
                "1:1": "いちたいいち",
                "3:2": "さんたいに",
                "比率は16:9": "比率はじゅうろくたいきゅう",
                "1:10": "いちじじゅっぷん",
                "次のとおりです：": "次のとおりです、",
                "ポイントは：": "ポイントは、",
                "例えば：": "例えば、",
                "2つあります：": "ふたつあります、",
                "日時：10月1日": "日時は、じゅうがつ ついたち",
                "サーバー：": "サーバーは、",
            }
        )

    def test_counter_compounds_and_context(self) -> None:
        self.assert_normalized(
            {
                "Windows 11 本体": "Windows じゅういち 本体",
                "3階層構造": "さん階層構造",
                "100回答": "ひゃく回答",
                "3組織": "さん組織",
                "1年1組": "いちねんいちくみ",
                "3割5分": "さん割ごぶ",
                "3 人": "さんにん",
                "5 名": "ごめい",
            }
        )

    def test_kai_and_soku_counters_leave_verbs_alone(self) -> None:
        self.assert_normalized(
            {
                "もう1回答えてください。": "もういっかい答えてください。",
                "ゴミは週1回収集します。": "ゴミは週いっかい収集します。",
                "1回収録": "いっかい収録",
                "1回転します。": "いっかい転します。",
                "1回路": "いっかい路",
                "100回答": "ひゃく回答",
                "1回収": "いち回収",
                "3に1足して4です。": "さんにいち足してよんです。",
                "1足した": "いち足した",
                "1足せば": "いち足せば",
                "1足さない": "いち足さない",
                "1足そう": "いち足そう",
                "1足、3足": "いっそく、さんぞく",
            }
        )

    def test_chome_geminates(self) -> None:
        self.assert_normalized(
            {
                "西新宿1丁目": "西新宿いっちょうめ",
                "8丁目": "はっちょうめ",
                "10丁目": "じゅっちょうめ",
                "3丁目": "さんちょうめ",
                "1-3丁目": "いっちょうめからさんちょうめ",
            }
        )

    def test_weekday_parentheses(self) -> None:
        self.assert_normalized(
            {
                "説明会（土・日）": "説明会、土曜日、日曜日、",
                "開館（月・祝）": "開館、月曜日、祝日、",
                "休館（祝）": "休館、祝日、",
                "営業(月祝)": "営業、月曜日、祝日、",
                "休み（祝日）": "休み、祝日、",
            }
        )

    def test_urls_are_removed_and_email_is_kept(self) -> None:
        self.assert_normalized(
            {
                # A URL used as a noun becomes ウェブサイト; one in parentheses or on its own is dropped.
                "詳しくはhttps://www.teu.ac.jp/をご覧ください。": "詳しくはウェブサイトをご覧ください。",
                "本学Webサイト（https://www.teu.ac.jp/）をご覧ください。": "本学Webサイトをご覧ください。",
                "サイトはwww.teu.ac.jpです。": "サイトはウェブサイトです。",
                "詳しくは次のページです。https://www.teu.ac.jp/": "詳しくは次のページです。",
                "連絡先はinfo@teu.ac.jpです。": "連絡先はinfo@teu.ac.jpです。",
            }
        )

    def test_parenthesized_readings_are_not_spoken_twice(self) -> None:
        self.assert_normalized(
            {
                "片倉町（かたくらまち）": "かたくらまち",
                "蒲田（かまた）キャンパス": "かまたキャンパス",
                "香川（かがわ）豊（ゆたか）": "かがわゆたか",
                "香川豊（かがわ・ゆたか）": "かがわゆたか",
                "香川豊学長（かがわゆたか）": "香川ゆたか学長",
                "東京都八王子市片倉町（かたくらまち）": "東京都八王子市片倉町",
                "“力(ちから)”": "“ちから”",
                "青嵐（セイラン）": "せいらん",
                "青嵐（SEIRAN、せいらん）": "せいらん",
                "青嵐（読み：せいらん）": "せいらん",
                "本学の蒲田（かまた）キャンパス": "本学のかまたキャンパス",
                "変化に対応できる力(ちから)": "変化に対応できるちから",
                # Not a reading: too short, a common gloss, or spelling the okurigana too.
                "入学金（なし）": "入学金、なし、",
                "受付（なし）": "受付、なし、",
                "定員（なし）": "定員、なし、",
                "昼食（あり）": "昼食、あり、",
                "対象（どなたでも）": "対象、どなたでも、",
                "日時（みてい）": "日時、みてい、",
                "持ち物（なし）": "持ち物、なし、",
                "持ち物（もちもの）": "持ち物、もちもの、",
                "(注)変更の可能性": "注、変更の可能性",
                "大学案内（2027年度版（デジタル））": "大学案内、にせんにじゅうななねんど版、デジタル、",
            }
        )

    def test_pronunciation_hints_run_before_numbers(self) -> None:
        self.assert_normalized(
            {
                "TOP 500": "トップごひゃく",
                "香川豊子さん": "香川豊子さん",
                "香川豊学長": "香川ゆたか学長",
            }
        )

    def test_wave_dash_after_kana_is_a_long_vowel(self) -> None:
        self.assert_normalized(
            {
                "よろしくね〜。": "よろしくねー。",
                "ありがとう～！": "ありがとうー!",
                "受付は10時〜。": "受付はじゅうじから。",
            }
        )

    def test_symbols_are_verbalized(self) -> None:
        self.assert_normalized(
            {
                "学生寮説明&寮見学": "学生寮説明と寮見学",
                "Q&Aコーナー": "キューアンドエーコーナー",
                "MS&T21": "MS&T21",
                "NDR × 2": "NDRかけるに",
                "180GB × 96基": "ひゃくはちじゅうギガバイトかけるきゅうじゅうろっき",
                "※詳細は後日": "詳細は後日",
                "予約は不要です※一部は予約制です。": "予約は不要です、一部は予約制です。",
                "a = b": "a = b",
                "応募→審査→合格": "応募、審査、合格",
            }
        )

    def test_units(self) -> None:
        self.assert_normalized(
            {
                "400Gb/s": "よんひゃくギガビット毎秒",
                "1000GB/s": "せんギガバイト毎秒",
                "8TB/s": "はちテラバイト毎秒",
                "10Gbps": "じゅうギガビット毎秒",
                "16gb": "じゅうろくギガバイト",
                "1000W": "せんワット",
                "10kW": "じゅっキロワット",
                "14.3kW": "じゅうよんてんさんキロワット",
                "1.5kWh": "いってんごキロワットアワー",
                "100V": "ひゃくボルト",
                "50Hz": "ごじゅうヘルツ",
                "60dB": "ろくじゅうデシベル",
                "60fps": "ろくじゅうエフピーエス",
                "-3.5dB": "マイナスさんてんごデシベル",
                "5W1H": "5W1H",
                "2026 W杯": "にせんにじゅうろく W杯",
                "約40万m²": "約よんじゅうまんへいほうメートル",
                "約39万㎡": "約さんじゅうきゅうまんへいほうメートル",
            }
        )


if __name__ == "__main__":
    unittest.main()
