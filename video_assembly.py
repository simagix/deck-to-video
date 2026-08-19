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


def _apply_ken_burns(clip, target_size: Tuple[int, int], zoom: float, pan_ratio: float = 0.5, index: int = 0):
    """Apply a gentle Ken Burns zoom-in + diagonal pan, cropped to target size."""
    duration = clip.duration or 0.0
    target_w, target_h = target_size
    zoom_out = False
    if index % 2 == 0:
        zoom_out = True
    _, _, _, CompositeVideoClip = _import_moviepy()

    def _progress(t: float) -> float:
        if duration <= 0:
            return 1.0
        return min(max(t / duration, 0.0), 1.0)

    # zoom in
    def _zoom_in_at(t: float) -> float:
        return 1.0 + (zoom - 1.0) * _progress(t)

    # zoom out (for testing)
    def _zoom_out_at(t: float) -> float:
        return zoom - (zoom - 1.0) * _progress(t)

    def _pos_at(t: float) -> Tuple[float, float]:
        progress = _progress(t)
        scale = 1.0 + (zoom - 1.0) * progress
        img_w = target_w * scale
        img_h = target_h * scale
        margin_x = (img_w - target_w) / 2.0
        margin_y = (img_h - target_h) / 2.0
        # Base random direction for this clip (stored per-slide for variety),
        # but the pan offsets are computed so the image remains centered
        # at start (progress=0) and end (progress=1).
        base_angle = getattr(_pos_at, "base_angle", None)
        if base_angle is None:
            base_angle = random.uniform(0, 2 * math.pi)
            _pos_at.base_angle = base_angle
        # Tapering factor: zero at progress=0 and progress=1,
        # peaking in the middle. This keeps the image centered throughout.
        t = progress
        taper = math.sin(2 * math.pi * 0.25 * t) * t * (1 - t)
        # Pan offsets vanish at t=0 and t=1, keeping image centered.
        dx = margin_x * pan_ratio * taper * math.cos(base_angle)
        dy = margin_y * pan_ratio * taper * math.sin(base_angle)
        return ((target_w - img_w) / 2.0 + dx, (target_h - img_h) / 2.0 + dy)

    if hasattr(clip, "resized"):
        if zoom_out:
            clip = clip.resized(lambda t: _zoom_out_at(t))
        else:
            clip = clip.resized(lambda t: _zoom_in_at(t))
    else:
        if zoom_out:   
            clip = clip.resize(lambda t: _zoom_out_at(t))
        else:
            clip = clip.resize(lambda t: _zoom_in_at(t))

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
