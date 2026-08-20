"""MoviePy helpers to assemble slide images and voiceover audio into MP4."""

from __future__ import annotations

import math
import os
import random
from typing import List, Optional, Tuple

from paths import (
    DEFAULT_FPS,
    DEFAULT_INTER_SLIDE_PAUSE_SECONDS,
    DEFAULT_SILENT_SLIDE_SECONDS,
    KEN_BURNS_IMAGE_SIZE,
    TARGET_IMAGE_SIZE,
)

AUDIO_CROSSFADE_SECONDS = 0.1


def _import_moviepy():
    try:
        from moviepy.editor import (  # type: ignore[import-untyped]
            AudioFileClip,
            CompositeVideoClip,
            ImageClip,
            concatenate_videoclips,
        )
    except ImportError:
        from moviepy import (  # type: ignore[import-untyped,no-redef]
            AudioFileClip,
            CompositeVideoClip,
            ImageClip,
            concatenate_videoclips,
        )
    return AudioFileClip, ImageClip, concatenate_videoclips, CompositeVideoClip


def _clip_set_duration(clip, duration: float):
    if hasattr(clip, "with_duration"):
        return clip.with_duration(duration)
    return clip.set_duration(duration)


def _clip_set_audio(clip, audio):
    if hasattr(clip, "with_audio"):
        return clip.with_audio(audio)
    return clip.set_audio(audio)


def _apply_audio_crossfade(audio_clip, fade_seconds: float = AUDIO_CROSSFADE_SECONDS):
    if fade_seconds <= 0:
        return audio_clip
    try:
        return audio_clip.audio_fadein(fade_seconds).audio_fadeout(fade_seconds)
    except AttributeError:
        pass
    try:
        from moviepy.audio.fx import AudioFadeIn, AudioFadeOut  # type: ignore[import-untyped]

        return audio_clip.with_effects([AudioFadeIn(fade_seconds), AudioFadeOut(fade_seconds)])
    except (ImportError, AttributeError):
        print("   ⚠️  MoviePy audio fade unavailable; transitions may click slightly.")
        return audio_clip


def _close_clip(clip) -> None:
    if clip is None:
        return
    close = getattr(clip, "close", None)
    if callable(close):
        close()


def _smoothstep(t: float) -> float:
    """Classic ease-in/out 'smoothstep', remapping [0,1] -> [0,1].

    Gently lifts (zero velocity) at t=0 and settles (zero velocity) at t=1,
    with continuous speed in between.  This is the curve iMovie's Ken Burns
    pans are built on; a plain linear ramp starts/stops abruptly and reads as
    twitchy / 'shaky'.
    """
    t = max(0.0, min(t, 1.0))
    return t * t * (3.0 - 2.0 * t)


def _apply_ken_burns(clip, target_size: Tuple[int, int], zoom: float, pan_ratio: float = 0.5, index: int = 0):
    """Apply an iMovie-style smooth Ken Burns zoom + pan, cropped to target size.

    This models a single coherent camera move: the window slides along one
    constant direction while zooming, and BOTH the zoom and the pan are driven
    by the SAME smoothstep easing curve. A shared eased parameter keeps the
    motion coherent and free of the old 'drift out and back' taper
    (``sin(pi/2*t)*t*(1-t)``) that made slides rock/sway instead of panning.
    """
    duration = clip.duration or 0.0
    target_w, target_h = target_size

    # Alternate zoom-in / zoom-out per slide for variety (as before).
    zoom_out = index % 2 == 0
    start_scale = zoom if zoom_out else 1.0
    end_scale = 1.0 if zoom_out else zoom

    # One pan heading per slide; both axes follow it (no mid-pan reversal).
    pan_angle = random.uniform(0, 2 * math.pi)

    def _progress(t: float) -> float:
        if duration <= 0:
            return 1.0
        return min(max(t / duration, 0.0), 1.0)

    def _zoom_at(t: float) -> float:
        e = _smoothstep(_progress(t))
        return start_scale + (end_scale - start_scale) * e

    def _pos_at(t: float) -> Tuple[float, float]:
        e = _smoothstep(_progress(t))
        scale = start_scale + (end_scale - start_scale) * e
        img_w = target_w * scale
        img_h = target_h * scale
        margin_x = (img_w - target_w) / 2.0
        margin_y = (img_h - target_h) / 2.0
        # Monotonic pan along a single heading, eased with the same curve as
        # the zoom. `2*e - 1` sweeps from -1 (start) to +1 (end), so the window
        # translates from one edge toward the other — no 'there and back' sway.
        pan = 2.0 * e - 1.0
        dx = margin_x * pan_ratio * pan * math.cos(pan_angle)
        dy = margin_y * pan_ratio * pan * math.sin(pan_angle)
        return ((target_w - img_w) / 2.0 + dx, (target_h - img_h) / 2.0 + dy)

    _, _, _, CompositeVideoClip = _import_moviepy()

    if hasattr(clip, "resized"):
        clip = clip.resized(lambda t: _zoom_at(t))
    else:
        clip = clip.resize(lambda t: _zoom_at(t))

    if hasattr(clip, "with_position"):
        clip = clip.with_position(lambda t: _pos_at(t))
    else:
        clip = clip.set_position(lambda t: _pos_at(t))

    composite = CompositeVideoClip([clip], size=(int(target_w), int(target_h)))
    return _clip_set_duration(composite, duration)


def build_slide_clip(
    png_path: str,
    wav_path: Optional[str],
    fps: int = DEFAULT_FPS,
    silent_seconds: float = DEFAULT_SILENT_SLIDE_SECONDS,
    trailing_pause_seconds: float = 0.0,
    ken_burns_zoom: float = 0.0,
    index: int = 0
):
    """Create one slide sub-clip: image (optionally with Ken Burns) + voiceover audio."""
    AudioFileClip, ImageClip, _, _ = _import_moviepy()

    audio_track = None
    video_track = None
    slide_sub_clip = None

    try:
        if wav_path and os.path.isfile(wav_path):
            audio_track = AudioFileClip(wav_path)
            audio_track = _apply_audio_crossfade(audio_track)
            # Voiceover WAVs end with a built-in 1s trailing silence, which
            # serves as the natural gap before the next slide — don't add a
            # separate inter-slide pause on top of it.
            duration = audio_track.duration
        else:
            duration = silent_seconds + max(0.0, trailing_pause_seconds)

        video_track = ImageClip(png_path)
        video_track = _clip_set_duration(video_track, duration)
        if ken_burns_zoom > 0:
            video_track = _apply_ken_burns(
                video_track, KEN_BURNS_IMAGE_SIZE, ken_burns_zoom, index=index
            )
        if audio_track is not None:
            slide_sub_clip = _clip_set_audio(video_track, audio_track)
            if slide_sub_clip is not video_track:
                _close_clip(video_track)
                video_track = None
        else:
            slide_sub_clip = video_track
            video_track = None

        if hasattr(slide_sub_clip, "with_fps"):
            slide_sub_clip = slide_sub_clip.with_fps(fps)
        elif hasattr(slide_sub_clip, "set_fps"):
            slide_sub_clip = slide_sub_clip.set_fps(fps)

        return slide_sub_clip, audio_track
    except Exception:
        _close_clip(slide_sub_clip)
        _close_clip(video_track)
        _close_clip(audio_track)
        raise


def assemble_presentation_video(
    png_paths: List[str],
    wav_paths: List[Optional[str]],
    output_mp4: str,
    fps: int = DEFAULT_FPS,
    inter_slide_pause_seconds: float = DEFAULT_INTER_SLIDE_PAUSE_SECONDS,
    ken_burns_zoom: float = 0.0,
) -> str:
    """Stitch per-slide clips into one MP4, optionally with Ken Burns zoom/pan."""
    if not png_paths:
        raise ValueError("No slide images to assemble")

    _, _, concatenate_videoclips, _ = _import_moviepy()

    slide_clips = []
    audio_handles = []

    try:
        total_slides = len(png_paths)
        for idx, (png_path, wav_path) in enumerate(zip(png_paths, wav_paths), start=1):
            trailing_pause = inter_slide_pause_seconds if idx < total_slides else 0.0
            print(f"   🎬 Building clip for slide {idx}...")
            slide_clip, audio_track = build_slide_clip(
                png_path,
                wav_path,
                fps=fps,
                trailing_pause_seconds=trailing_pause,
                ken_burns_zoom=ken_burns_zoom,
                index = idx,
            )
            slide_clips.append(slide_clip)
            if audio_track is not None:
                audio_handles.append(audio_track)

        final_movie = concatenate_videoclips(slide_clips, method="compose")
        print(f"\n📼 Rendering {output_mp4} ...")
        final_movie.write_videofile(
            output_mp4,
            fps=fps,
            codec="libx264",
            audio_codec="aac",
            logger=None,
        )
        _close_clip(final_movie)
        return output_mp4
    finally:
        for clip in slide_clips:
            _close_clip(clip)
        for audio in audio_handles:
            _close_clip(audio)
