"""Unit tests for the FFmpeg assembly in video_assembly.py: the Ken Burns
zoompan expression generator, the dip-window layout behind the dip-black /
dip-white / crossfade transition styles, and end-to-end assembly checked on
real rendered frames."""

from __future__ import annotations

import math
import os
import re
import subprocess
import tempfile
import unittest
import wave

import numpy as np
from PIL import Image

import bgm
from paths import (
    DEFAULT_TRANSITION_SECONDS,
    DEFAULT_TRANSITION_STYLE,
    TARGET_IMAGE_SIZE,
)
from video_assembly import (
    DIP_BLACK,
    DIP_WHITE,
    TRANSITION_STYLES,
    _dip_windows,
    _extract_frame,
    _ffmpeg_exe,
    _ken_burns_filter,
    assemble_presentation_video,
)

SLIDE_RED = (200, 30, 30)
SLIDE_GREEN = (30, 200, 30)
SLIDE_BLUE = (30, 30, 200)


def _write_pcm_wav(path, rate=8000, seconds=0.5):
    """Write a short soft sine WAV (stand-in for a voiceover track)."""
    n = int(rate * seconds)
    t = np.arange(n) / rate
    tone = (0.3 * np.sin(2 * np.pi * 220.0 * t)).reshape(-1, 1)
    pcm = np.clip(np.round(tone * 32767), -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm.tobytes())


def _write_solid_png(path, rgb, size=(64, 36)):
    """Write a solid-color PNG slide."""
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    arr[:, :] = rgb
    Image.fromarray(arr).save(path)


def _extract_frame_t(path, t):
    """Decode one frame at *t* seconds as an RGB numpy array (via ffmpeg)."""
    tmp = f"{path}.frame.png"
    subprocess.run(
        [
            _ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
            "-ss", str(t), "-i", path, "-frames:v", "1", "-update", "1", tmp,
        ],
        check=True,
    )
    frame = np.asarray(Image.open(tmp).convert("RGB"), dtype=int)
    os.remove(tmp)
    return frame


def _last_frame(path, fps=8):
    """Decode the final frame of a rendered video (via video_assembly)."""
    tmp = f"{path}.last.png"
    _extract_frame(path, fps, last=True, out_png=tmp)
    frame = np.asarray(Image.open(tmp).convert("RGB"), dtype=int)
    os.remove(tmp)
    return frame


def _video_duration(path):
    """Duration of a video via ffmpeg null-mux pass (no ffprobe needed)."""
    result = subprocess.run(
        [_ffmpeg_exe(), "-hide_banner", "-i", path, "-map", "0:v", "-f", "null", "-"],
        capture_output=True,
        text=True,
    )
    stamps = re.findall(r"time=(\d+):(\d+):([\d.]+)", result.stderr)
    hours, minutes, seconds = stamps[-1]
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


class TransitionConstantsTests(unittest.TestCase):
    def test_supported_styles_and_default(self):
        self.assertIn("none", TRANSITION_STYLES)
        self.assertIn("dip-black", TRANSITION_STYLES)
        self.assertIn("dip-white", TRANSITION_STYLES)
        self.assertIn("crossfade", TRANSITION_STYLES)
        self.assertEqual(DEFAULT_TRANSITION_STYLE, "dip-black")
        self.assertEqual(DEFAULT_TRANSITION_SECONDS, 0.5)


class KenBurnsFilterTests(unittest.TestCase):
    """The zoompan expression generator encodes the same move as before."""

    def test_zoom_direction_alternates_by_index(self):
        even = _ken_burns_filter(4.0, 24, 1.15, 0, 0.0)
        odd = _ken_burns_filter(4.0, 24, 1.15, 1, 0.0)
        # Zoom-out branch starts at the full zoom factor; zoom-in at 1.0.
        self.assertIn("(1.15-(1.15-1)*", even)
        self.assertIn("(1+(1.15-1)*", odd)

    def test_frame_count_follows_duration_and_fps(self):
        self.assertIn("d=8:", _ken_burns_filter(1.0, 8, 1.15, 0, 0.0))
        self.assertIn("d=264:", _ken_burns_filter(11.0, 24, 1.15, 0, 0.0))

    def test_pan_heading_enters_x_and_y(self):
        angle = 1.234
        chain = _ken_burns_filter(4.0, 24, 1.15, 0, angle)
        self.assertIn(f"{math.cos(angle):.6f}", chain)
        self.assertIn(f"{math.sin(angle):.6f}", chain)

    def test_supersampled_zoompan_scales_to_target(self):
        chain = _ken_burns_filter(4.0, 24, 1.15, 0, 0.0)
        self.assertIn("scale=3648:2048", chain)
        self.assertIn("zoompan=z='(1.15-(1.15-1)*", chain)
        self.assertIn("s=1824x1024:", chain)
        self.assertIn("flags=lanczos", chain)


class DipWindowTests(unittest.TestCase):
    """The closer/boundary dip layout matches the original assembly rules."""

    def test_first_slide_gets_no_head_dip(self):
        dip_in_s, dip_in_c, dip_out_s, dip_out_c = _dip_windows(
            "dip-black", 0.5, True, False
        )
        self.assertEqual(dip_in_s, 0.0)
        self.assertEqual(dip_out_s, 0.25)
        self.assertEqual(dip_out_c, DIP_BLACK)

    def test_last_slide_gets_full_black_closer(self):
        dip_in_s, dip_in_c, dip_out_s, dip_out_c = _dip_windows(
            "dip-black", 0.5, False, True
        )
        self.assertEqual(dip_in_s, 0.25)
        self.assertEqual(dip_out_s, 0.5)
        self.assertEqual(dip_out_c, DIP_BLACK)

    def test_dip_white_boundaries_are_white(self):
        _, dip_in_c, _, dip_out_c = _dip_windows("dip-white", 0.5, False, False)
        self.assertEqual(dip_in_c, DIP_WHITE)
        self.assertEqual(dip_out_c, DIP_WHITE)

    def test_none_style_has_no_dips(self):
        self.assertEqual(
            _dip_windows("none", 0.5, False, False),
            (0.0, DIP_BLACK, 0.0, DIP_BLACK),
        )

    def test_zero_duration_disables_everything(self):
        self.assertEqual(
            _dip_windows("dip-black", 0.0, False, True),
            (0.0, DIP_BLACK, 0.0, DIP_BLACK),
        )


class AssembleWithTransitionsTests(unittest.TestCase):
    """End-to-end renders of a tiny three-slide deck (0.5s of audio each)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.pngs = []
        self.wavs = []
        for idx, rgb in enumerate((SLIDE_RED, SLIDE_GREEN, SLIDE_BLUE), start=1):
            png = os.path.join(self._tmp.name, f"slide_{idx:02d}.png")
            _write_solid_png(png, rgb)
            self.pngs.append(png)
            wav = os.path.join(self._tmp.name, f"slide_{idx:02d}_voiceover.wav")
            _write_pcm_wav(wav, seconds=0.5)
            self.wavs.append(wav)

    def _assemble(self, **kwargs):
        output = os.path.join(self._tmp.name, "out.mp4")
        assemble_presentation_video(self.pngs, self.wavs, output, fps=8, **kwargs)
        return output

    def test_none_hard_cuts(self):
        output = self._assemble(transition="none")
        self.assertAlmostEqual(_video_duration(output), 1.5, delta=0.25)
        frame = _extract_frame_t(output, 0.0)
        self.assertTrue((np.abs(frame - SLIDE_RED) <= 8).all())

    def test_dip_white_keeps_runtime_and_fades(self):
        output = self._assemble(transition="dip-white")
        self.assertAlmostEqual(_video_duration(output), 1.5, delta=0.25)
        # No fade-in: the first slide is fully visible from frame 0.
        self.assertTrue(
            (np.abs(_extract_frame_t(output, 0.0) - SLIDE_RED) <= 8).all()
        )
        # Boundary at t=0.5: the incoming slide's head dip → (almost) white.
        self.assertGreaterEqual(int(_extract_frame_t(output, 0.5).min()), 230)
        # Closer: the last frame fades out to (almost) pure black.
        self.assertLessEqual(int(_last_frame(output).max()), 200)

    def test_default_dip_black_keeps_runtime_and_fades(self):
        # No transition kwarg → the default style ('dip-black') is used.
        output = self._assemble()
        self.assertAlmostEqual(_video_duration(output), 1.5, delta=0.25)
        # No fade-in: the first slide is fully visible from frame 0, so
        # thumbnails and previews show the slide content instead of black.
        self.assertTrue(
            (np.abs(_extract_frame_t(output, 0.0) - SLIDE_RED) <= 8).all()
        )
        # Boundary at t=0.5: the incoming slide's head dip → (almost) black.
        self.assertLessEqual(int(_extract_frame_t(output, 0.5).max()), 16)
        # Mid-slide content is untouched (slide 2 is fully visible at 0.75).
        self.assertGreaterEqual(int(_extract_frame_t(output, 0.75).max()), 150)
        # Closer: the last frame fades out to (almost) pure black.
        self.assertLessEqual(int(_last_frame(output).max()), 200)

    def test_crossfade_extends_runtime_and_blends(self):
        output = self._assemble(transition="crossfade")
        # 3 × 0.5s slides + 2 × 0.5s dissolves = 2.5s.
        self.assertAlmostEqual(_video_duration(output), 2.5, delta=0.3)
        # First dissolve spans [0.5, 1.0]; its midpoint is a 50/50 blend.
        mid = _extract_frame_t(output, 0.75)
        expected = (np.asarray(SLIDE_RED) + np.asarray(SLIDE_GREEN)) // 2
        self.assertTrue((np.abs(mid - expected) <= 40).all())

    def test_ken_burns_output_is_target_size_and_duration(self):
        output = self._assemble(transition="none", ken_burns_zoom=1.15)
        self.assertAlmostEqual(_video_duration(output), 1.5, delta=0.25)
        frame = _extract_frame_t(output, 0.1)
        self.assertEqual(
            frame.shape[:2], (TARGET_IMAGE_SIZE[1], TARGET_IMAGE_SIZE[0])
        )

    def test_invalid_style_raises_value_error(self):
        with self.assertRaises(ValueError):
            self._assemble(transition="sparkle")


class BackgroundMusicMixTests(unittest.TestCase):
    """End-to-end BGM layering on voiced and fully silent decks."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.pngs = []
        for idx, rgb in enumerate((SLIDE_RED, SLIDE_GREEN), start=1):
            png = os.path.join(self._tmp.name, f"slide_{idx:02d}.png")
            _write_solid_png(png, rgb)
            self.pngs.append(png)
        self.music = bgm.resolve_bgm_path("ambient_loop")

    def _decode_audio(self, path, seconds=6.0):
        samples = bgm._ffmpeg_resampled_frames(path, 8000, 1)
        self.assertGreater(len(samples), 0)
        return samples[: int(seconds * 8000)]

    def test_voiced_deck_gets_music_under_narration(self):
        wavs = []
        for idx in (1, 2):
            wav = os.path.join(self._tmp.name, f"slide_{idx:02d}_voiceover.wav")
            _write_pcm_wav(wav, seconds=0.5)
            wavs.append(wav)
        output = os.path.join(self._tmp.name, "voiced_bgm.mp4")
        assemble_presentation_video(
            self.pngs, wavs, output, fps=8, transition="none",
            background_music_path=self.music, bg_music_volume=0.5,
        )
        self.assertAlmostEqual(_video_duration(output), 1.0, delta=0.3)
        # Music is audible, not just narration silence.
        self.assertGreater(float(np.abs(self._decode_audio(output)).max()), 0.01)

    def test_silent_deck_gets_music_as_sole_soundtrack(self):
        output = os.path.join(self._tmp.name, "silent_bgm.mp4")
        assemble_presentation_video(
            self.pngs, [None, None], output, fps=8, transition="none",
            background_music_path=self.music, bg_music_volume=0.5,
        )
        # 2 silent slides (3s each) + 1 inter-slide pause (1s) = 7s.
        self.assertAlmostEqual(_video_duration(output), 7.0, delta=0.5)
        self.assertGreater(float(np.abs(self._decode_audio(output)).max()), 0.01)


if __name__ == "__main__":
    unittest.main()
