"""Unit tests for background music: [bgm: ...] tag parsing in narration.py,
asset resolution / numpy mixing in bgm.py, and assembly-time layering glue."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
import wave
from unittest import mock

import numpy as np

import bgm
import narration


def _write_pcm_wav(path, rate=8000, seconds=0.5, channels=1):
    """Write a soft sine WAV and return identical float64 frames."""
    n = int(rate * seconds)
    t = np.arange(n) / rate
    tone = (0.4 * np.sin(2 * np.pi * 220.0 * t)).reshape(-1, 1)
    tone = np.repeat(tone, channels, axis=1)
    pcm = np.clip(np.round(tone * 32767), -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm.tobytes())
    return pcm.astype(np.float64) / 32768.0


class ParseBgmCuesTests(unittest.TestCase):
    def test_simple_tag(self):
        self.assertEqual(
            narration.parse_bgm_cues("[bgm: ambient_loop]"),
            [("ambient_loop", None)],
        )

    def test_tag_with_volume_spec(self):
        cues = narration.parse_bgm_cues("Hello\n[bgm: calm.mp3 | volume: 0.2]\nBye")
        self.assertEqual(cues, [("calm.mp3", 0.2)])

    def test_volume_accepts_equals_sign(self):
        self.assertEqual(
            narration.parse_bgm_cues("[BGM: score | volume = 0.3]"), [("score", 0.3)]
        )

    def test_multiple_tags_keep_order_and_duplicates(self):
        cues = narration.parse_bgm_cues("[bgm: a]\n[bgm: b | volume: 0.4]\n[bgm: a]")
        self.assertEqual(cues, [("a", None), ("b", 0.4), ("a", None)])

    def test_unknown_spec_fields_are_ignored(self):
        self.assertEqual(
            narration.parse_bgm_cues("[bgm: x | loop: true | volume: 0.1]"),
            [("x", 0.1)],
        )

    def test_no_tags(self):
        self.assertEqual(narration.parse_bgm_cues("plain notes"), [])

    def test_empty_reference_is_skipped(self):
        self.assertEqual(narration.parse_bgm_cues("[bgm: ]"), [])

    def test_newlines_do_not_fool_the_parser(self):
        self.assertEqual(narration.parse_bgm_cues("[bgm:\nambient_loop]"), [])


class StripBgmTagsTests(unittest.TestCase):
    def test_tag_removed_leaving_prose_intact(self):
        text = "First paragraph.\n[bgm: ambient_loop | volume: 0.2]\nLast paragraph."
        self.assertEqual(
            narration._strip_bgm_tags(text), "First paragraph.\n\nLast paragraph."
        )

    def test_all_tag_kinds_never_reach_tts_text(self):
        text = (
            "[voice: Simone | tone: witty]\n"
            "Joke! [sfx: rimshot]\n"
            "[bgm: ambient_loop | volume: 0.15]"
        )
        cleaned = narration._strip_bgm_tags(
            narration._strip_sfx_tags(narration._strip_voice_tone_tags(text))
        )
        for leak in ("[voice:", "[tone:", "[sfx:", "[bgm:", "ambient_loop"):
            self.assertNotIn(leak, cleaned)

    def test_text_without_tags_is_returned_unchanged(self):
        self.assertEqual(narration._strip_bgm_tags("Just words."), "Just words.")


class ResolveBgmPathTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.assets = tmp.name
        for name in ("loop.mp3", "pad.wav", "both.mp3", "both.wav"):
            with open(os.path.join(self.assets, name), "wb") as fh:
                fh.write(b"fake")

    def _resolve(self, ref):
        with mock.patch.object(bgm, "ASSETS_DIR", self.assets):
            return bgm.resolve_bgm_path(ref)

    def test_bare_name_resolves_inside_assets(self):
        self.assertEqual(os.path.basename(self._resolve("loop")), "loop.mp3")

    def test_bare_name_falls_back_to_wav(self):
        self.assertEqual(os.path.basename(self._resolve("pad")), "pad.wav")

    def test_mp3_preferred_over_wav(self):
        self.assertEqual(os.path.basename(self._resolve("both")), "both.mp3")

    def test_existing_path_used_verbatim_and_absolute(self):
        marker = os.path.join(self.assets, "..", "elsewhere.mp3")
        with open(marker, "wb") as fh:
            fh.write(b"fake")
        resolved = self._resolve(marker)
        self.assertTrue(os.path.isfile(resolved))
        self.assertTrue(os.path.isabs(resolved))

    def test_missing_track_lists_attempts(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            self._resolve("nope")
        self.assertIn("nope", str(ctx.exception))
        self.assertIn(self.assets, str(ctx.exception))

    def test_empty_reference_is_an_error(self):
        with self.assertRaises(ValueError):
            self._resolve("   ")


@unittest.skipUnless(
    os.path.isfile(os.path.join(bgm.ASSETS_DIR, "ambient_loop.mp3")),
    "bundled ambient_loop.mp3 not generated yet",
)
class ShippedAssetTests(unittest.TestCase):
    def test_default_bundle_resolves_by_name_as_mp3(self):
        resolved = bgm.resolve_bgm_path("ambient_loop")
        self.assertTrue(os.path.isfile(resolved))
        self.assertTrue(resolved.endswith(".mp3"))


class FitChannelsTests(unittest.TestCase):
    def test_downfold_stereo_to_mono_averages(self):
        np.testing.assert_array_equal(
            bgm.fit_channels(np.array([[1.0, 2.0], [3.0, 4.0]]), 1), [[1.5], [3.5]]
        )

    def test_upscale_duplicates_channels(self):
        np.testing.assert_array_equal(
            bgm.fit_channels(np.array([[0.5], [-0.5]]), 2),
            [[0.5, 0.5], [-0.5, -0.5]],
        )

    def test_divisible_downfold_groups_columns(self):
        signal = np.array([[1.0, 3.0, 10.0, 20.0]])
        np.testing.assert_array_equal(bgm.fit_channels(signal, 2), [[2.0, 15.0]])

    def test_odd_downfold_falls_back_to_mean_mono_repeat(self):
        signal = np.array([[1.0, 2.0, 3.0]])
        np.testing.assert_array_equal(bgm.fit_channels(signal, 2), [[2.0, 2.0]])

    def test_matching_layout_untouched(self):
        signal = np.zeros((4, 2))
        self.assertIs(signal, bgm.fit_channels(signal, 2))


class TileToSamplesTests(unittest.TestCase):
    def test_exactly_total_rows_are_returned(self):
        signal = np.arange(9.0).reshape(3, 3)
        tiled = bgm.tile_to_samples(signal, 7)
        self.assertEqual(tiled.shape, (7, 3))
        np.testing.assert_array_equal(tiled[:3], signal)

    def test_tiling_restarts_on_track_boundary(self):
        signal = np.arange(6.0).reshape(2, 3)
        tiled = bgm.tile_to_samples(signal, 5)
        np.testing.assert_array_equal(tiled[2], signal[0])

    def test_zero_request_yields_empty_rows(self):
        self.assertEqual(bgm.tile_to_samples(np.ones((2, 2)), 0).shape, (0, 2))


class WriteMusicWavTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.rate = 8000
        self.src_seconds = 0.5
        self.source_wav = os.path.join(tmp.name, "src.wav")
        self.expected = _write_pcm_wav(
            self.source_wav, rate=self.rate, seconds=self.src_seconds
        )
        self.out_wav = os.path.join(tmp.name, "music.wav")

    def _write(self, duration=1.0, volume=1.0):
        return bgm.write_music_wav(
            self.source_wav,
            duration,
            sample_rate=self.rate,
            channels=1,
            volume=volume,
            out_path=self.out_wav,
        )

    def _samples(self):
        with wave.open(self.out_wav, "rb") as wf:
            self.assertEqual(wf.getframerate(), self.rate)
            self.assertEqual(wf.getnchannels(), 1)
            self.assertEqual(wf.getsampwidth(), 2)
            payload = wf.readframes(wf.getnframes())
        return np.frombuffer(payload, dtype="<i2").astype(np.float64) / 32768.0

    def test_gain_applied_at_every_sample(self):
        self._write(duration=0.5, volume=0.5)
        got = self._samples()
        self.assertEqual(len(got), int(0.5 * self.rate))
        # Both sides are quantized to int16, but through different scales
        # (32767 vs 32768); sub-LSB differences are expected.
        np.testing.assert_allclose(got, self.expected[:, 0] * 0.5, atol=2e-5)

    def test_loop_wraps_around_source_end(self):
        # 1.2s output = 9600 samples; sample 4800 is 800 into the 2nd repeat.
        self._write(duration=1.2)
        got = self._samples()
        self.assertEqual(len(got), int(1.2 * self.rate))
        np.testing.assert_allclose(got[4800:4900], self.expected[800:900, 0], atol=1e-6)

    def test_duration_matches_timeline(self):
        self._write(duration=1.05)
        self.assertEqual(len(self._samples()), int(1.05 * self.rate))

    def test_invalid_arguments_raise(self):
        with self.assertRaises(ValueError):
            bgm.write_music_wav(
                self.source_wav, 0, sample_rate=self.rate, channels=1,
                out_path=self.out_wav,
            )
        with self.assertRaises(ValueError):
            bgm.write_music_wav(
                self.source_wav, 1, sample_rate=self.rate, channels=1,
                volume=-0.1, out_path=self.out_wav,
            )
        with self.assertRaises(ValueError):
            bgm.write_music_wav(
                self.source_wav, 1, sample_rate=0, channels=1,
                out_path=self.out_wav,
            )


class MixBackgroundMusicTests(unittest.TestCase):
    """End-to-end: write_music_wav + video_assembly's amix pass on real MP4s."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        self.voice_wav = os.path.join(tmp.name, "slide_01_voiceover.wav")
        _write_pcm_wav(self.voice_wav, rate=8000, seconds=0.5)
        self.music_ref = bgm.resolve_bgm_path("ambient_loop")

        ffmpeg = shutil.which("ffmpeg")
        voiced = os.path.join(tmp.name, "voiced.mp4")
        subprocess.run(
            [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "color=c=black:s=64x36:r=8:d=0.75",
             "-i", self.voice_wav,
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
             "-shortest", voiced],
            check=True,
        )
        silent = os.path.join(tmp.name, "silent.mp4")
        subprocess.run(
            [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "color=c=black:s=64x36:r=8:d=0.75",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", silent],
            check=True,
        )
        self.voiced, self.silent = voiced, silent

    def _score(self, video):
        from video_assembly import _mux_final

        music_wav = os.path.join(self.tmp, "bgm.wav")
        bgm.write_music_wav(
            self.music_ref, 0.75, sample_rate=8000, channels=1,
            volume=0.5, out_path=music_wav,
        )
        output = os.path.join(self.tmp, "scored.mp4")
        if video == self.voiced:
            # Narration WAV + looped music → layered amix.
            _mux_final(video, self.voice_wav, music_wav, output, 0.75)
        else:
            # Fully silent deck: the music is the sole soundtrack.
            _mux_final(video, None, music_wav, output, 0.75)
        return output

    def test_voiced_deck_gets_music_layered_under_narration(self):
        output = self._score(self.voiced)
        samples = bgm._ffmpeg_resampled_frames(output, 8000, 1)
        self.assertGreater(len(samples), 0)
        # Narration (0.4 amplitude tone) survives the layering.
        self.assertGreater(float(np.abs(samples[:4000]).max()), 0.3)

    def test_silent_deck_gets_music_as_sole_soundtrack(self):
        output = self._score(self.silent)
        samples = bgm._ffmpeg_resampled_frames(output, 8000, 1)
        self.assertGreater(len(samples), 0)
        # Music (gained to 0.5 of the track) is the only sound.
        self.assertGreater(float(np.abs(samples).max()), 0.01)


if __name__ == "__main__":
    unittest.main()