"""Unit tests for slide-change transitions in video_assembly.py: dip-to-color
frame blending, still-dissolve crossfade boundary clips, and end-to-end
assembly with transitions enabled."""

from __future__ import annotations

import os
import tempfile
import unittest
import wave

import numpy as np

from paths import DEFAULT_TRANSITION_SECONDS, DEFAULT_TRANSITION_STYLE
from video_assembly import (
    DIP_BLACK,
    DIP_WHITE,
    TRANSITION_STYLES,
    _blend_toward_color,
    _build_crossfade_clip,
    _clip_set_duration,
    _import_moviepy,
    assemble_presentation_video,
    build_slide_clip,
)

try:
    from moviepy.editor import VideoFileClip  # type: ignore[import-untyped]
except ImportError:
    from moviepy import VideoFileClip  # type: ignore[import-untyped,no-redef]


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
    from PIL import Image

    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    arr[:, :] = rgb
    Image.fromarray(arr).save(path)


def _solid_image_clip(rgb, size=(64, 36), duration=1.0):
    """A solid-color ImageClip of the given duration."""
    _, ImageClip, _, _, _ = _import_moviepy()
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    arr[:, :] = rgb
    return _clip_set_duration(ImageClip(arr), duration)


class TransitionConstantsTests(unittest.TestCase):
    def test_supported_styles_and_default(self):
        self.assertIn("none", TRANSITION_STYLES)
        self.assertIn("dip-black", TRANSITION_STYLES)
        self.assertIn("dip-white", TRANSITION_STYLES)
        self.assertIn("crossfade", TRANSITION_STYLES)
        self.assertEqual(DEFAULT_TRANSITION_STYLE, "dip-black")
        self.assertEqual(DEFAULT_TRANSITION_SECONDS, 0.5)


class BlendTowardColorTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((4, 6, 3), dtype=np.uint8)
        self.frame[:, :] = SLIDE_RED

    def test_zero_factor_returns_frame_unchanged(self):
        blended = _blend_toward_color(self.frame, 0.0, DIP_WHITE)
        self.assertTrue((blended == self.frame).all())

    def test_full_factor_returns_solid_color(self):
        blended = _blend_toward_color(self.frame, 1.0, DIP_WHITE)
        self.assertTrue((blended == 255).all())

    def test_half_factor_blends_halfway(self):
        blended = _blend_toward_color(self.frame, 0.5, DIP_WHITE)
        expected = np.asarray(SLIDE_RED) * 0.5 + 255 * 0.5
        self.assertTrue((np.abs(blended.astype(int) - expected) <= 2).all())


class DipEffectTests(unittest.TestCase):
    """build_slide_clip with dip parameters, on a silent slide (no WAV)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.png = os.path.join(self._tmp.name, "slide_01.png")
        _write_solid_png(self.png, SLIDE_RED)

    def _clip(self, **kwargs):
        clip, audio = build_slide_clip(
            self.png, None, fps=8, silent_seconds=1.0, **kwargs
        )
        self.assertIsNone(audio)
        self.addCleanup(clip.close)
        return clip

    def test_no_dips_leaves_frames_untouched(self):
        clip = self._clip()
        for t in (0.0, 0.5, 0.999):
            self.assertTrue((clip.get_frame(t) == SLIDE_RED).all(), f"t={t}")

    def test_head_dip_fades_from_white(self):
        clip = self._clip(dip_in_seconds=0.5, dip_in_color=DIP_WHITE)
        self.assertTrue((clip.get_frame(0.0) == 255).all())
        # smoothstep(0.5) == 0.5, so a quarter second in the blend is half done.
        quarter = clip.get_frame(0.25)
        expected = np.asarray(SLIDE_RED) * 0.5 + 255 * 0.5
        self.assertTrue((np.abs(quarter.astype(int) - expected) <= 2).all())
        self.assertTrue((clip.get_frame(0.75) == SLIDE_RED).all())

    def test_tail_dip_fades_to_white(self):
        clip = self._clip(dip_out_seconds=0.5, dip_out_color=DIP_WHITE)
        self.assertTrue((clip.get_frame(0.0) == SLIDE_RED).all())
        # The tail window is [0.5, 1.0], so 0.4 is still untouched...
        self.assertTrue((clip.get_frame(0.4) == SLIDE_RED).all())
        # ...0.75 is the halfway blend (smoothstep(0.5) == 0.5)...
        mid = clip.get_frame(0.75)
        expected = np.asarray(SLIDE_RED) * 0.5 + 255 * 0.5
        self.assertTrue((np.abs(mid.astype(int) - expected) <= 2).all())
        # ...and the very end is fully white.
        self.assertTrue((clip.get_frame(1.0) == 255).all())

    def test_head_dip_fades_from_black(self):
        # Mechanism test: a head dip toward black ramps color → frame (used by
        # boundary head dips); the first slide of an assembly gets none.
        clip = self._clip(dip_in_seconds=0.5, dip_in_color=DIP_BLACK)
        self.assertTrue((clip.get_frame(0.0) == 0).all())
        self.assertTrue((clip.get_frame(0.75) == SLIDE_RED).all())

    def test_default_dip_color_is_black(self):
        # No color passed → the default dip color is black (the 'dip-black'
        # style's look, and the opener/closer look in every style).
        clip = self._clip(dip_in_seconds=0.5)
        self.assertTrue((clip.get_frame(0.0) == 0).all())
        self.assertTrue((clip.get_frame(0.75) == SLIDE_RED).all())

    def test_oversized_dips_are_clamped_to_duration(self):
        clip = self._clip(
            dip_in_seconds=5.0,
            dip_out_seconds=5.0,
            dip_in_color=DIP_WHITE,
            dip_out_color=DIP_WHITE,
        )
        # 10s of dips on a 1s clip: both windows shrink to 0.5s and meet red
        # exactly at the midpoint.
        self.assertTrue((clip.get_frame(0.0) == 255).all())
        self.assertTrue((clip.get_frame(0.5) == SLIDE_RED).all())
        self.assertTrue((clip.get_frame(1.0) == 255).all())

    def test_dip_applies_after_ken_burns(self):
        clip = self._clip(
            ken_burns_zoom=1.15, index=0, dip_in_seconds=0.25, dip_in_color=DIP_WHITE
        )
        self.assertTrue((clip.get_frame(0.0) == 255).all())
        self.assertFalse((clip.get_frame(0.9) == 255).all())


class CrossfadeClipTests(unittest.TestCase):
    def test_dissolves_from_outgoing_to_incoming(self):
        outgoing = _solid_image_clip(SLIDE_RED)
        incoming = _solid_image_clip(SLIDE_GREEN)
        self.addCleanup(outgoing.close)
        self.addCleanup(incoming.close)

        crossfade = _build_crossfade_clip(outgoing, incoming, 0.5, fps=8)
        self.assertIsNotNone(crossfade)
        self.addCleanup(crossfade.close)
        self.assertAlmostEqual(crossfade.duration, 0.5)

        # smoothstep(0.5) == 0.5, so the midpoint frame is a 50/50 blend.
        self.assertTrue((crossfade.get_frame(0.0) == SLIDE_RED).all())
        mid = crossfade.get_frame(0.25)
        expected = (np.asarray(SLIDE_RED) + np.asarray(SLIDE_GREEN)) // 2
        self.assertTrue((np.abs(mid.astype(int) - expected) <= 2).all())
        self.assertTrue((crossfade.get_frame(0.5) == SLIDE_GREEN).all())

    def test_mismatched_sizes_raise(self):
        small = _solid_image_clip(SLIDE_RED, size=(32, 18))
        large = _solid_image_clip(SLIDE_GREEN, size=(64, 36))
        self.addCleanup(small.close)
        self.addCleanup(large.close)
        with self.assertRaises(ValueError):
            _build_crossfade_clip(small, large, 0.5, fps=8)

    def test_zero_duration_returns_none(self):
        outgoing = _solid_image_clip(SLIDE_RED)
        incoming = _solid_image_clip(SLIDE_GREEN)
        self.addCleanup(outgoing.close)
        self.addCleanup(incoming.close)
        self.assertIsNone(_build_crossfade_clip(outgoing, incoming, 0.0, fps=8))


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

    def _video(self, path):
        video = VideoFileClip(path)
        self.addCleanup(video.close)
        return video

    def test_none_hard_cuts(self):
        output = self._assemble(transition="none")
        with self._video(output) as video:
            self.assertAlmostEqual(video.duration, 1.5, delta=0.25)
            self.assertTrue(
                (np.abs(video.get_frame(0.0).astype(int) - SLIDE_RED) <= 8).all()
            )

    def test_dip_white_keeps_runtime_and_fades(self):
        output = self._assemble(transition="dip-white")
        with self._video(output) as video:
            self.assertAlmostEqual(video.duration, 1.5, delta=0.25)
            # No fade-in: the first slide is fully visible from frame 0.
            self.assertTrue(
                (np.abs(video.get_frame(0.0).astype(int) - SLIDE_RED) <= 8).all()
            )
            # Boundary at t=0.5: the incoming slide's head dip → (almost) white.
            self.assertGreaterEqual(int(video.get_frame(0.5).min()), 230)
            # Closer: the last frame fades out to (almost) pure black.
            self.assertLessEqual(int(video.get_frame(video.duration - 0.01).max()), 200)

    def test_default_dip_black_keeps_runtime_and_fades(self):
        # No transition kwarg → the default style ('dip-black') is used.
        output = self._assemble()
        with self._video(output) as video:
            self.assertAlmostEqual(video.duration, 1.5, delta=0.25)
            # No fade-in: the first slide is fully visible from frame 0, so
            # thumbnails and previews show slide content instead of black.
            self.assertTrue(
                (np.abs(video.get_frame(0.0).astype(int) - SLIDE_RED) <= 8).all()
            )
            # Boundary at t=0.5: the incoming slide's head dip → (almost) black.
            self.assertLessEqual(int(video.get_frame(0.5).max()), 16)
            # Mid-slide content is untouched (slide 2 is fully visible at 0.75).
            self.assertGreaterEqual(int(video.get_frame(0.75).max()), 150)
            # Closer: the last frame fades out to (almost) pure black.
            self.assertLessEqual(int(video.get_frame(video.duration - 0.01).max()), 200)

    def test_crossfade_extends_runtime_and_blends(self):
        output = self._assemble(transition="crossfade")
        with self._video(output) as video:
            # 3 × 0.5s slides + 2 × 0.5s dissolves = 2.5s.
            self.assertAlmostEqual(video.duration, 2.5, delta=0.3)
            # First dissolve spans [0.5, 1.0]; its midpoint is a 50/50 blend.
            mid = video.get_frame(0.75).astype(int)
            expected = (np.asarray(SLIDE_RED) + np.asarray(SLIDE_GREEN)) // 2
            self.assertTrue((np.abs(mid - expected) <= 40).all())

    def test_invalid_style_raises_value_error(self):
        with self.assertRaises(ValueError):
            self._assemble(transition="sparkle")


if __name__ == "__main__":
    unittest.main()
