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


if __name__ == "__main__":
    unittest.main(verbosity=2)