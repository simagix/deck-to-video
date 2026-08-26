"""Unit tests for punchline sound effects: [sfx: rimshot] tag parsing in
narration.py and WAV mixing in sfx.py.

Regression guards: SFX tags must be stripped from text sent to Voicebox,
recognized cues must resolve to canonical sample names, and overlay must land
the hit right after end-of-speech while extending (never truncating) the WAV.
"""

from __future__ import annotations

import os
import tempfile
import unittest
import wave

import numpy as np

import narration
import sfx


def _write_tone_wav(path, rate=8000, speech_seconds=0.5, tail_seconds=0.2):
    """Write a fake voiceover: a tone, then trailing silence. Returns speech end (s)."""
    speech_n = int(rate * speech_seconds)
    t = np.arange(speech_n) / rate
    tone = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    silence_n = int(rate * tail_seconds)
    pcm = np.concatenate([tone, np.zeros(silence_n)])
    pcm = np.clip(np.round(pcm * 32767), -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm.tobytes())
    return len(tone) / rate


def _read_wav(path):
    with wave.open(path, "rb") as wf:
        params = wf.getparams()
        frames = wf.readframes(wf.getnframes())
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float64) / 32768.0
    return params, samples.reshape(-1, params.nchannels)


class ParseSfxCuesTests(unittest.TestCase):
    def test_recognizes_sfx_name_tag(self):
        self.assertEqual(narration.parse_sfx_cues("Ha!\n[sfx: rimshot]"), ["rimshot"])

    def test_recognizes_badumtss_aliases(self):
        for alias in ("[badumtss]", "[Ba Dum Tss]", "[ba_dum-tss]", "[rimshot]"):
            with self.subTest(alias=alias):
                self.assertEqual(narration.parse_sfx_cues(f"Joke.{alias}"), ["rimshot"])

    def test_ignores_unknown_sfx_names(self):
        self.assertEqual(narration.parse_sfx_cues("[sfx: applause]"), [])

    def test_dedupes_and_keeps_order(self):
        cues = narration.parse_sfx_cues("[badumtss]\n[sfx: rimshot]")
        self.assertEqual(cues, ["rimshot"])

    def test_no_tags(self):
        self.assertEqual(narration.parse_sfx_cues("plain notes"), [])
        self.assertEqual(narration.parse_sfx_cues(""), [])

    def test_voice_tone_tags_not_confused_with_sfx(self):
        self.assertEqual(narration.parse_sfx_cues("[voice: Simone | tone: witty] hi"), [])


class StripSfxTagsTests(unittest.TestCase):
    def test_tags_stripped_from_tts_text_both_modes(self):
        notes = "That's the last time I trust auto-merge.\n[sfx: rimshot]"
        for personality in (False, True):
            with self.subTest(personality=personality):
                out = narration.prepare_narration(notes, personality=personality)
                self.assertNotIn("sfx", out.lower())
                self.assertNotIn("[", out)
                self.assertIn("auto-merge", out)

    def test_laugh_cue_still_reaches_tts_alongside_sfx_tag(self):
        notes = "Owt! [laugh] [badumtss]"
        out = narration.prepare_narration(notes)
        self.assertIn("[laugh]", out)
        self.assertNotIn("badumtss", out)

    def test_strip_leaves_no_double_spaces(self):
        out = narration._strip_sfx_tags("before [sfx: rimshot] after")
        self.assertEqual(out, "before after")


class OverlaySfxOnWavTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.wav_path = os.path.join(self.tmp.name, "slide_01_voiceover.wav")
        self.rate = 8000
        self.speech_end = _write_tone_wav(self.wav_path, rate=self.rate)

    def tearDown(self):
        self.tmp.cleanup()

    def test_overlay_extends_wav_and_lands_after_speech(self):
        original_params, original = _read_wav(self.wav_path)
        beat = 0.25
        self.assertTrue(
            sfx.overlay_sfx_on_wav(self.wav_path, "rimshot", beat_seconds=beat, tail_seconds=1.0)
        )
        params, mixed = _read_wav(self.wav_path)

        # Same format, strictly longer.
        self.assertEqual(params.framerate, original_params.framerate)
        self.assertEqual(params.sampwidth, original_params.sampwidth)
        self.assertEqual(params.nchannels, original_params.nchannels)
        self.assertGreater(len(mixed), len(original))

        # Speech region untouched.
        speech_samples = int(self.speech_end * self.rate)
        np.testing.assert_allclose(mixed[:speech_samples], original[:speech_samples])

        # Beat gap is silent, then the hit lands right after it.
        hit_start = int(round((self.speech_end + beat) * self.rate))
        gap = mixed[speech_samples:hit_start]
        self.assertAlmostEqual(float(np.abs(gap).max()), 0.0, places=6)
        hit_region = mixed[hit_start : hit_start + self.rate]
        self.assertGreater(float(np.abs(hit_region).max()), 0.05)

        # Contract: near-silence across the final second of the file.
        tail = mixed[-self.rate :]
        self.assertLess(float(np.abs(tail).max()), 0.02)

    def test_unknown_sfx_raises_value_error(self):
        with self.assertRaises(ValueError):
            sfx.overlay_sfx_on_wav(self.wav_path, "applause")

    def test_empty_wav_is_noop(self):
        empty = os.path.join(self.tmp.name, "empty.wav")
        with wave.open(empty, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.rate)
            wf.writeframes(b"")
        self.assertFalse(sfx.overlay_sfx_on_wav(empty, "rimshot"))

    def test_bundled_sample_exists_and_loads_at_other_rates(self):
        path = sfx.resolve_sfx_path("rimshot")
        self.assertTrue(os.path.isfile(path), f"missing {path}; run make_sfx_assets.py")
        resampled = sfx.load_sfx("rimshot", framerate=self.rate, nchannels=1)
        self.assertGreater(len(resampled), int(0.8 * self.rate))  # ≈1.4s of audio

    def test_resolve_prefers_user_sample_over_default(self):
        # assets/ ships ba_dum_tss.wav (canonical/user slot); the synthesized
        # fallback lives at ba_dum_tss_default.wav and must lose to it.
        candidates = list(sfx.SFX_SAMPLES["rimshot"])
        self.assertEqual(sfx.resolve_sfx_path("rimshot"), os.path.join(sfx.ASSETS_DIR, candidates[0]))


class SplitNotesOnSfxTests(unittest.TestCase):
    USER_NOTES = (
        "[tone: angry]\n"
        "Maybe that's the secret.\n"
        "People step up when you trust them.\n"
        "Wally has another approach.\n"
        "He's not a just micromanager.\n"
        "He's an organizational surveillance system… \n"
        "[sfx: rimshot]\n"
        "[tone: dramatic]\n"
        "… Simone ... Owt!"
    )

    def test_mid_slide_tag_splits_into_three_parts(self):
        parts = narration.split_notes_on_sfx(self.USER_NOTES)
        self.assertEqual(
            [p["kind"] for p in parts], ["narration", "sfx", "narration"]
        )
        self.assertEqual(parts[1]["name"], "rimshot")
        self.assertIn("surveillance system", parts[0]["source"])
        self.assertIn("[tone: angry]", parts[0]["source"])
        self.assertNotIn("sfx:", parts[0]["source"])
        # The trailing take keeps its own tone tag and text.
        self.assertIn("[tone: dramatic]", parts[2]["source"])
        self.assertIn("Simone", parts[2]["source"])

    def test_each_take_prepares_cleanly_with_its_own_tone(self):
        parts = narration.split_notes_on_sfx(self.USER_NOTES)
        first = narration.prepare_narration(parts[0]["source"])
        last = narration.prepare_narration(parts[2]["source"])
        for out in (first, last):
            self.assertNotIn("[", out)
            self.assertNotIn("tone:", out)
        self.assertIn("surveillance system", first)
        self.assertEqual(
            narration._first_tone_instruct(parts[2]["source"]),
            narration._tone_instruct("dramatic"),
        )
        self.assertEqual(
            narration._first_tone_instruct(parts[0]["source"]),
            narration._tone_instruct("angry"),
        )

    def test_no_cue_yields_single_part_covering_all_notes(self):
        notes = "[tone: witty]\n\nJust a normal slide."
        parts = narration.split_notes_on_sfx(notes)
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0]["kind"], "narration")
        self.assertEqual(parts[0]["source"], notes)

    def test_unsupported_cue_does_not_split(self):
        parts = narration.split_notes_on_sfx("A.[sfx: applause]B.")
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0]["kind"], "narration")

    def test_supported_set_restricts_splitting(self):
        notes = "A. [badumtss] B."
        self.assertEqual(len(narration.split_notes_on_sfx(notes, supported=set())), 1)
        parts = narration.split_notes_on_sfx(notes, supported={"rimshot"})
        self.assertEqual([p["kind"] for p in parts], ["narration", "sfx", "narration"])

    def test_cues_at_edges_emit_no_empty_takes(self):
        parts = narration.split_notes_on_sfx("[badumtss]")
        self.assertEqual(parts, [{"kind": "sfx", "name": "rimshot"}])
        parts = narration.split_notes_on_sfx("[badumtss]\nAfter.")
        self.assertEqual(
            [p["kind"] for p in parts], ["sfx", "narration"]
        )


class AssembleVoiceoverTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.rate = 8000

    def tearDown(self):
        self.tmp.cleanup()

    def _take(self, name, rate, seconds=0.3):
        path = os.path.join(self.tmp.name, name)
        _write_tone_wav(path, rate=rate, speech_seconds=seconds, tail_seconds=0.0)
        return path

    def test_splices_hit_exactly_between_two_takes(self):
        take1 = self._take("t1.wav", self.rate, 0.30)
        take2 = self._take("t2.wav", 16000, 0.25)  # different rate on purpose
        out = os.path.join(self.tmp.name, "slide_voiceover.wav")
        beat, tail = 0.25, 1.0
        inserted = sfx.assemble_voiceover(
            [("wav", take1), ("sfx", "rimshot"), ("wav", take2)],
            out,
            beat_seconds=beat,
            tail_seconds=tail,
        )
        self.assertEqual(inserted, ["rimshot"])

        params, mixed = _read_wav(out)
        sfx_dur = len(sfx.load_sfx("rimshot", params.framerate, params.nchannels)) / params.framerate
        expected = 0.30 + beat + sfx_dur + 0.25 + tail
        self.assertAlmostEqual(params.nframes / params.framerate, expected, delta=0.02)

        # Take 1 copied verbatim into every output channel (the mix may have
        # been upgraded to stereo because the rimshot sample is stereo).
        _, take1_samples = _read_wav(take1)
        np.testing.assert_allclose(
            mixed[: len(take1_samples)],
            np.tile(take1_samples, (1, params.nchannels)),
        )

        # Beat gap silent; hit lands right after it; then take 2 audible.
        hit_start = int(round((0.30 + beat) * params.framerate))
        gap = mixed[len(take1_samples) : hit_start]
        self.assertLess(float(np.abs(gap).max()), 1e-9)
        self.assertGreater(float(np.abs(mixed[hit_start : hit_start + params.framerate]).max()), 0.05)

        # Final second is near-silence (trailing-tail contract).
        self.assertLess(float(np.abs(mixed[-params.framerate :]).max()), 0.02)

    def test_requires_at_least_one_take(self):
        with self.assertRaises(ValueError):
            sfx.assemble_voiceover([("sfx", "rimshot")], os.path.join(self.tmp.name, "x.wav"))

    def test_output_upgrades_to_sfx_channel_count(self):
        """A wider effect upgrades the mix instead of being folded to mono."""
        native_channels = wave.open(sfx.resolve_sfx_path("rimshot")).getnchannels()
        take1 = self._take("c1.wav", self.rate, 0.30)
        out = os.path.join(self.tmp.name, "channels.wav")
        sfx.assemble_voiceover(
            [("wav", take1), ("sfx", "rimshot")],
            out,
            beat_seconds=0.25,
            tail_seconds=1.0,
        )
        params, mixed = _read_wav(out)
        self.assertEqual(params.nchannels, max(1, native_channels))
        # The voice is duplicated identically across every output channel.
        np.testing.assert_array_equal(mixed[:, 0], mixed[:, -1])


if __name__ == "__main__":
    unittest.main(verbosity=2)