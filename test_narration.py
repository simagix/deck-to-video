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


if __name__ == "__main__":
    unittest.main(verbosity=2)