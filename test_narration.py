"""Unit tests for narration.py, capturing how speaker notes are transformed
before being sent to Voicebox.

Regression guard: bracketed cues such as "[laugh]" must SURVIVE narration prep so
Voicebox can interpret them, while parenthesized asides are still stripped.
"""

from __future__ import annotations

import unittest

import narration

MARKED_UP_RAW = "Simone \u2026 [laugh] \u2026 Owt!"  # "Simone … [laugh] … Owt!"


class StripStageDirectionTests(unittest.TestCase):
    def test_removes_parenthesized_cues(self):
        self.assertEqual(narration.strip_stage_directions("(whispering) hi"), "hi")
        self.assertEqual(narration.strip_stage_directions("hi (pause)"), "hi")

    def test_preserves_bracketed_cues_for_tts_engine(self):
        """[laugh] must reach Voicebox, so bracketed cues are kept intact."""
        self.assertEqual(narration.strip_stage_directions("[laugh] hi"), "[laugh] hi")
        self.assertEqual(narration.strip_stage_directions("hi [sigh]"), "hi [sigh]")

    def test_bracketed_laugh_survives_marked_up_notes(self):
        cleaned = narration.strip_stage_directions(MARKED_UP_RAW)
        self.assertIn("[laugh]", cleaned)


class PrepareNarrationTests(unittest.TestCase):
    def test_laugh_marker_reaches_tts_in_both_modes(self):
        """[laugh] now survives narration prep and is sent to Voicebox."""
        for personality in (False, True):
            with self.subTest(personality=personality):
                out = narration.prepare_narration(MARKED_UP_RAW, personality=personality)
                print(f">>> prepare_narration(personality={personality}) = {out!r}")
                self.assertIn("[laugh]", out)
                # "…" is normalized to "...", brackets untouched.
                self.assertEqual(out, "Simone ... [laugh] ... Owt!")

    def test_parenthesized_aside_still_stripped(self):
        text = "First slide (this is just a note) continues."
        out = narration.prepare_narration(text, personality=False)
        self.assertNotIn("just a note", out)
        self.assertEqual(out, "First slide continues.")

    def test_plain_tts_preserves_inline_sub_alias_punctuation(self):
        text = "Install mongosh from the CLI, not the API."
        out = narration.prepare_narration(text, personality=False)
        # Pronunciation aliases survive (used by the TTS engine).
        self.assertIn("mongo-shell", out)
        self.assertIn("C-L-I", out)


class VoiceToneTagParsingTests(unittest.TestCase):
    def test_remove_voice_tone_tags_strips_all_tags(self):
        notes = (
            "[voice: Simone | tone: neutral]\n\n"
            "Welcome everyone.\n\n"
            "[voice: Alex | tone: cheerful]\n\n"
            "And I have some good news."
        )
        stripped = narration._strip_voice_tone_tags(notes)
        self.assertNotIn("[voice:", stripped)
        self.assertNotIn("tone:", stripped)
        self.assertNotIn("Simone", stripped)
        self.assertNotIn("Alex", stripped)
        self.assertIn("Welcome everyone.", stripped)
        self.assertIn("And I have some good news.", stripped)

    def test_parse_blocks_produces_expected_blocks(self):
        notes = (
            "[voice: Simone | tone: neutral]\n\n"
            "Welcome everyone.\n\n"
            "[voice: Alex | tone: cheerful]\n\n"
            "And I have some good news."
        )
        blocks = narration._parse_blocks(notes)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0]["voice"], "Simone")
        self.assertEqual(blocks[0]["tone"], "neutral")
        self.assertEqual(blocks[0]["text"], "Welcome everyone.")
        self.assertEqual(blocks[1]["voice"], "Alex")
        self.assertEqual(blocks[1]["tone"], "cheerful")
        self.assertEqual(blocks[1]["text"], "And I have some good news.")

    def test_tone_only_tag(self):
        """A lone [tone: frustrated] tag (no voice) must set tone and be stripped."""
        notes = (
            "Hi... I'm Simone... the Hatchet assistant.\n"
            "[tone: frustrated]\n"
            "My typical workday begins long... before I even log on."
        )
        blocks = narration._parse_blocks(notes)
        # Leading untagged narration is its own block; the tag starts block 2.
        self.assertEqual(len(blocks), 2)
        self.assertIsNone(blocks[0]["tone"])
        self.assertEqual(blocks[0]["text"], "Hi... I'm Simone... the Hatchet assistant.")
        self.assertEqual(blocks[1]["tone"], "frustrated")
        self.assertEqual(
            blocks[1]["text"], "My typical workday begins long... before I even log on."
        )
        # The tag must be stripped from the narration text sent to TTS.
        stripped = narration.prepare_narration(notes)
        self.assertNotIn("[tone:", stripped)
        self.assertNotIn("frustrated", stripped)
        # And the tone must map to the full instruct.
        self.assertEqual(
            narration._tone_instruct(blocks[1]["tone"]),
            "Frustrated and exasperated, with noticeable impatience.",
        )

    def test_voice_only_tag_keeps_tone_sticky(self):
        """[voice: Alex] alone should keep the previous tone sticky."""
        notes = (
            "[voice: Simone | tone: neutral]\n"
            "First speaker.\n\n"
            "[voice: Alex]\n"
            "Second speaker stays neutral."
        )
        blocks = narration._parse_blocks(notes)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[1]["voice"], "Alex")
        self.assertEqual(blocks[1]["tone"], "neutral")  # tone stuck from block 1

    def test_sticky_voice_tone_across_paragraphs(self):
        notes = (
            "[voice: Simone | tone: neutral]\n\n"
            "Welcome to today's presentation.\n\n"
            "This is an important topic.\n\n"
            "[voice: Alex | tone: excited]\n\n"
            "And here's where things get interesting!"
        )
        blocks = narration._parse_blocks(notes)
        self.assertEqual(len(blocks), 2)
        # Both untagged paragraphs stick to Simone/neutral.
        self.assertIn("Welcome to today's presentation.", blocks[0]["text"])
        self.assertIn("This is an important topic.", blocks[0]["text"])
        self.assertEqual(blocks[0]["voice"], "Simone")
        self.assertEqual(blocks[0]["tone"], "neutral")
        self.assertEqual(blocks[1]["voice"], "Alex")
        self.assertEqual(blocks[1]["tone"], "excited")

    def test_no_tags_backward_compatible(self):
        notes = "Just some plain presentation notes."
        blocks = narration._parse_blocks(notes)
        self.assertEqual(len(blocks), 1)
        self.assertIsNone(blocks[0]["voice"])
        self.assertIsNone(blocks[0]["tone"])
        self.assertEqual(blocks[0]["text"], "Just some plain presentation notes.")
        # prepare_narration is unchanged by the tags feature.
        self.assertEqual(
            narration.prepare_narration(notes), "Just some plain presentation notes."
        )

    def test_tone_instruct_mapping(self):
        self.assertEqual(
            narration._tone_instruct("frustrated"),
            "Frustrated and exasperated, with noticeable impatience.",
        )
        self.assertEqual(
            narration._tone_instruct("sarcastic"),
            "Sarcastic and mocking, with a sharp edge.",
        )
        self.assertIsNone(narration._tone_instruct(None))
        self.assertIsNone(narration._tone_instruct("not-a-real-tone"))

    def test_all_tones_present_and_nonempty(self):
        self.assertEqual(len(narration.TONES), 28)
        for tone, instruction in narration.TONES.items():
            self.assertTrue(tone.strip())
            self.assertTrue(instruction.strip())


ZH_PURE = "宫保鸡丁是一道经典的川菜，麻辣鲜香，下饭又开胃。"
ZH_MIXED = "Kung Pao Chicken 宫保鸡丁是四川名菜，spicy and numbing."
EN_MENTION = "Today we cook Kung Pao Chicken 宫保鸡丁 for dinner."
EN_PURE = "Hello, welcome to this presentation on Sichuan cooking."
JA_PURE = "このプレゼンテーションは、四川料理の紹介から始まります。とても辛いです。"
KO_PURE = "이 프레젠테이션은 사천 요리 소개로 시작합니다. 매우 맵습니다."
RU_PURE = (
    "Эта презентация начинается с краткого введения. "
    "Мы показываем, как готовится это блюдо."
)
DE_PURE = (
    "Der Vortrag beginnt mit einer kurzen Einführung. Wir zeigen, wie das "
    "Gericht zubereitet wird, und warum die Gewürze so wichtig sind."
)
FR_PURE = (
    "Cette présentation commence par une courte introduction. Nous montrons "
    "comment le plat est préparé et pourquoi les épices sont si importantes."
)
ES_PURE = (
    "Esta presentación comienza con una breve introducción. Mostramos cómo "
    "se prepara el plato y por qué las especias son tan importantes."
)
IT_PURE = (
    "Questa presentazione inizia con una breve introduzione. Mostriamo come "
    "si prepara il piatto e perché le spezie sono così importanti."
)
PT_PURE = (
    "Esta apresentação começa com uma breve introdução. Mostramos como o "
    "prato é preparado e porque os temperos são tão importantes."
)


class LanguageDetectionTests(unittest.TestCase):
    def test_pure_mandarin_detects_zh(self):
        self.assertEqual(narration.detect_language(ZH_PURE), "zh")

    def test_mixed_content_detects_zh(self):
        self.assertEqual(narration.detect_language(ZH_MIXED), "zh")

    def test_english_with_cjk_mention_stays_en(self):
        # A mostly-English slide that merely mentions 宫保鸡丁 stays English.
        self.assertEqual(narration.detect_language(EN_MENTION), "en")

    def test_pure_english_detects_en(self):
        self.assertEqual(narration.detect_language(EN_PURE), "en")

    def test_empty_and_neutral_detect_en(self):
        self.assertEqual(narration.detect_language(""), "en")
        self.assertEqual(narration.detect_language("   "), "en")
        self.assertEqual(narration.detect_language("123 ... !!!"), "en")

    def test_non_latin_scripts_detect_their_language(self):
        # Kana / Hangul / Cyrillic each pin down exactly one deck language.
        self.assertEqual(narration.detect_language(JA_PURE), "ja")
        self.assertEqual(narration.detect_language(KO_PURE), "ko")
        self.assertEqual(narration.detect_language(RU_PURE), "ru")

    def test_latin_markers_detect_their_language(self):
        for code, sample in (
            ("de", DE_PURE),
            ("fr", FR_PURE),
            ("es", ES_PURE),
            ("it", IT_PURE),
            ("pt", PT_PURE),
        ):
            self.assertEqual(narration.detect_language(sample), code, msg=code)

    def test_inverted_punctuation_is_spanish(self):
        self.assertEqual(narration.detect_language("¿Qué es esto?"), "es")

    def test_english_with_foreign_words_stays_english(self):
        # Function words alone are not enough — English must win outright.
        self.assertEqual(
            narration.detect_language(
                "This wine list is on the table for our guests."
            ),
            "en",
        )

    def test_override_forces_deck_language(self):
        self.assertEqual(
            narration.resolve_narration_language(ZH_PURE, override="en"), "en"
        )
        self.assertEqual(
            narration.resolve_narration_language(EN_PURE, override="zh"), "zh"
        )
        self.assertEqual(
            narration.resolve_narration_language(ZH_PURE, override="auto"), "zh"
        )
        self.assertEqual(
            narration.resolve_narration_language(EN_PURE, override=None), "en"
        )

    def test_code_mapping(self):
        self.assertEqual(narration.voicebox_language("zh"), "zh")
        self.assertEqual(narration.voicebox_language("en"), "en")
        self.assertEqual(narration.mlx_language("zh"), "chinese")
        self.assertEqual(narration.mlx_language("en"), "english")

    def test_parse_language_override(self):
        self.assertEqual(narration.parse_language_override(None), "auto")
        self.assertEqual(narration.parse_language_override("auto"), "auto")
        self.assertEqual(narration.parse_language_override("zh"), "zh")
        self.assertEqual(narration.parse_language_override("chinese"), "zh")
        self.assertEqual(narration.parse_language_override("EN"), "en")
        self.assertEqual(narration.parse_language_override("bogus"), "auto")


class MultiLanguageTests(unittest.TestCase):
    """All 10 languages Qwen3-TTS speaks must be reachable end to end."""

    def test_supported_codes_match_mlx_names(self):
        self.assertEqual(
            set(narration.SUPPORTED_LANGUAGE_CODES),
            set(narration.MLX_LANGUAGE_NAMES),
        )
        self.assertEqual(
            narration.VOICEBOX_LANGUAGE_CODES, narration.SUPPORTED_LANGUAGE_CODES
        )
        self.assertIn("auto", narration.MLX_LANGUAGE_CODES)

    def test_engine_codes_for_every_language(self):
        for code in narration.SUPPORTED_LANGUAGE_CODES:
            # Voicebox takes the code as-is; one-voice/mlx takes the full name.
            self.assertEqual(narration.voicebox_language(code), code)
            self.assertEqual(
                narration.mlx_language(code), narration.MLX_LANGUAGE_NAMES[code]
            )
        # Unknown / empty input lands on the English default, never raises.
        self.assertEqual(narration.voicebox_language("klingon"), "en")
        self.assertEqual(narration.mlx_language("klingon"), "english")
        self.assertEqual(narration.mlx_language(""), "english")
        self.assertEqual(narration.mlx_language("auto"), "english")
        # Names are accepted wherever codes are.
        self.assertEqual(narration.mlx_language("chinese"), "chinese")
        self.assertEqual(narration.voicebox_language("German"), "de")

    def test_canonical_language_accepts_names_and_region_tags(self):
        cases = {
            "en": "en", "english": "en", "EN": "en", "en-US": "en",
            "zh": "zh", "chinese": "zh", "CN": "zh", "mandarin": "zh",
            "zh-Hans": "zh",
            "ja": "ja", "japanese": "ja", "jp": "ja",
            "ko": "ko", "korean": "ko", "kr": "ko",
            "de": "de", "german": "de",
            "fr": "fr", "french": "fr",
            "ru": "ru", "russian": "ru",
            "pt": "pt", "portuguese": "pt", "pt-BR": "pt", "pt_BR": "pt",
            "es": "es", "spanish": "es",
            "it": "it", "italian": "it",
            "auto": "auto", "detect": "auto",
        }
        for value, expected in cases.items():
            self.assertEqual(
                narration.canonical_language(value), expected, msg=value
            )
        self.assertIsNone(narration.canonical_language(None))
        self.assertIsNone(narration.canonical_language("   "))
        self.assertIsNone(narration.canonical_language("klingon"))

    def test_parse_language_override_accepts_all_codes(self):
        for code in narration.SUPPORTED_LANGUAGE_CODES:
            self.assertEqual(narration.parse_language_override(code), code)
        self.assertEqual(narration.parse_language_override("german"), "de")
        self.assertEqual(narration.parse_language_override("pt-BR"), "pt")
        self.assertEqual(narration.parse_language_override(None), "auto")
        self.assertEqual(narration.parse_language_override("bogus"), "auto")

    def test_resolve_forces_every_language(self):
        # A deck-level override forces every slide, whatever its own script.
        for code in narration.SUPPORTED_LANGUAGE_CODES:
            for sample in (EN_PURE, ZH_PURE):
                self.assertEqual(
                    narration.resolve_narration_language(sample, override=code),
                    code,
                    msg=f"{code} on {sample[:12]!r}",
                )
        # "auto" and language names behave like an override too.
        self.assertEqual(
            narration.resolve_narration_language(EN_PURE, override="auto"), "en"
        )
        self.assertEqual(
            narration.resolve_narration_language(EN_PURE, override="german"), "de"
        )

    def test_japanese_and_mandarin_share_ideographic_full_stop(self):
        # Both use 。as the sentence terminator (deck_to_video relies on this).
        for code in ("zh", "ja"):
            self.assertEqual(narration.voicebox_language(code), code)
        self.assertEqual(narration.mlx_language("ja"), "japanese")


class CjkPunctuationTests(unittest.TestCase):
    def test_cjk_punctuation_survives_normalization(self):
        out = narration._normalize_unicode_punctuation(ZH_PURE)
        for punct in ("，", "。"):
            self.assertIn(punct, out)

    def test_prepare_narration_keeps_mandarin_boundaries(self):
        out = narration.prepare_narration(ZH_PURE, personality=False)
        self.assertIn("，", out)
        self.assertTrue(out.endswith("。"))


if __name__ == "__main__":
    unittest.main(verbosity=2)