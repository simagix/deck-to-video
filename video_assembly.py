"""MoviePy helpers to assemble slide images and voiceover audio into MP4."""

from __future__ import annotations

import os
from typing import List, Optional

from paths import DEFAULT_FPS, DEFAULT_INTER_SLIDE_PAUSE_SECONDS, DEFAULT_SILENT_SLIDE_SECONDS

AUDIO_CROSSFADE_SECONDS = 0.1


def _import_moviepy():
    try:
        from moviepy.editor import (  # type: ignore[import-untyped]
            AudioFileClip,
            ImageClip,
            concatenate_videoclips,
        )
    except ImportError:
        from moviepy import (  # type: ignore[import-untyped,no-redef]
            AudioFileClip,
            ImageClip,
            concatenate_videoclips,
        )
    return AudioFileClip, ImageClip, concatenate_videoclips


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


def build_slide_clip(
    png_path: str,
    wav_path: Optional[str],
    fps: int = DEFAULT_FPS,
    silent_seconds: float = DEFAULT_SILENT_SLIDE_SECONDS,
    trailing_pause_seconds: float = 0.0,
):
    """Create one slide sub-clip: static image + optional voiceover audio."""
    AudioFileClip, ImageClip, _ = _import_moviepy()

    audio_track = None
    video_track = None
    slide_sub_clip = None

    try:
        if wav_path and os.path.isfile(wav_path):
            audio_track = AudioFileClip(wav_path)
            audio_track = _apply_audio_crossfade(audio_track)
            duration = audio_track.duration + max(0.0, trailing_pause_seconds)
        else:
            duration = silent_seconds + max(0.0, trailing_pause_seconds)

        video_track = ImageClip(png_path)
        video_track = _clip_set_duration(video_track, duration)
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
) -> str:
    """Stitch per-slide clips into one MP4."""
    if not png_paths:
        raise ValueError("No slide images to assemble")

    _, _, concatenate_videoclips = _import_moviepy()

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
