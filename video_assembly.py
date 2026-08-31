"""MoviePy helpers to assemble slide images and voiceover audio into MP4."""

from __future__ import annotations

import math
import os
import random
from typing import List, Optional, Tuple

import numpy as np

import bgm
from paths import (
    DEFAULT_BG_MUSIC_VOLUME,
    DEFAULT_FPS,
    DEFAULT_INTER_SLIDE_PAUSE_SECONDS,
    DEFAULT_SILENT_SLIDE_SECONDS,
    DEFAULT_TRANSITION_SECONDS,
    DEFAULT_TRANSITION_STYLE,
    KEN_BURNS_IMAGE_SIZE,
    TARGET_IMAGE_SIZE,
)

AUDIO_CROSSFADE_SECONDS = 0.1

#: Solid colors the transitions dip through. The default 'dip-black' style —
#: and the video's closing fade in every style — fades through black like any
#: film or show would; the optional 'dip-white' style flashes toward white.
DIP_WHITE = (255, 255, 255)
DIP_BLACK = (0, 0, 0)

#: Supported slide-change transition styles (``--transition``).
TRANSITION_STYLES = ("none", "dip-black", "dip-white", "crossfade")


def _import_moviepy():
    try:
        from moviepy.editor import (  # type: ignore[import-untyped]
            AudioFileClip,
            CompositeVideoClip,
            ImageClip,
            VideoClip,
            concatenate_videoclips,
        )
    except ImportError:
        from moviepy import (  # type: ignore[import-untyped,no-redef]
            AudioFileClip,
            CompositeVideoClip,
            ImageClip,
            VideoClip,
            concatenate_videoclips,
        )
    return (
        AudioFileClip,
        ImageClip,
        concatenate_videoclips,
        CompositeVideoClip,
        VideoClip,
    )


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


def _apply_frame_effect(clip, func):
    """Apply a frame-level effect ``func(get_frame, t) -> frame`` (v1/v2 safe).

    MoviePy 2.x renamed ``Clip.fl`` to ``Clip.transform`` with the same
    calling convention, so use whichever the installed version provides.
    """
    if hasattr(clip, "transform"):
        return clip.transform(func)
    return clip.fl(func)


def _blend_toward_color(frame, factor: float, color: Tuple[int, int, int]):
    """Blend a uint8 RGB frame toward ``color`` by ``factor`` (0 = unchanged)."""
    if factor <= 0.0:
        return frame
    color_layer = np.asarray(color, dtype=np.float32)
    blended = frame.astype(np.float32) * (1.0 - factor) + color_layer * factor
    return blended.astype(frame.dtype)


def _apply_dip(
    clip,
    in_seconds: float,
    in_color: Tuple[int, int, int],
    out_seconds: float,
    out_color: Tuple[int, int, int],
):
    """Dip a clip's video toward a solid color at its head and/or tail.

    The head ramps ``color -> frame`` and the tail ramps ``frame -> color``,
    both eased with the same smoothstep curve as the Ken Burns moves. Frames
    outside the dip windows pass through untouched, and the audio track is
    never modified: boundary dips are sized to fit inside each slide's
    built-in trailing silence, so narration is never covered.
    """
    duration = clip.duration or 0.0
    in_seconds = max(0.0, float(in_seconds))
    out_seconds = max(0.0, float(out_seconds))
    total = in_seconds + out_seconds
    if duration <= 0.0 or total <= 0.0:
        return clip
    if total > duration:
        # Degenerate very-short clips: shrink both windows proportionally so
        # the head and tail dips never overlap.
        scale = duration / total
        in_seconds *= scale
        out_seconds *= scale

    def effect(get_frame, t):
        in_factor = (
            1.0 - _smoothstep(t / in_seconds)
            if in_seconds > 0.0 and t < in_seconds
            else 0.0
        )
        out_factor = (
            _smoothstep((t - (duration - out_seconds)) / out_seconds)
            if out_seconds > 0.0 and t > duration - out_seconds
            else 0.0
        )
        factor, color = (
            (in_factor, in_color) if in_factor >= out_factor else (out_factor, out_color)
        )
        if factor <= 0.0:
            return get_frame(t)
        return _blend_toward_color(get_frame(t), factor, color)

    return _apply_frame_effect(clip, effect)


def _build_crossfade_clip(outgoing_clip, incoming_clip, duration: float, fps: int):
    """Build a silent still-dissolve clip bridging two consecutive slides.

    Freezes the outgoing slide's final rendered frame and dissolves it into
    the incoming slide's first frame — the same still-to-still dissolve
    Keynote/PowerPoint use between slides. Freezing (instead of overlapping
    the moving clips) keeps Ken Burns motion continuous — each slide's
    zoom/pan plays exactly once, with no rewind stutter — and the incoming
    slide's narration starts unclipped after the dissolve. The clip carries
    no audio; ``concatenate_videoclips`` treats that as silence and offsets
    the following slide's audio by the boundary duration.
    """
    if duration <= 0.0:
        return None
    last_frame_t = max(0.0, (outgoing_clip.duration or 0.0) - 1.0 / max(int(fps), 1))
    frame_a = outgoing_clip.get_frame(last_frame_t)
    frame_b = incoming_clip.get_frame(0.0)
    if frame_a.shape != frame_b.shape:
        raise ValueError(
            "Cannot crossfade slides rendered at different sizes: "
            f"{frame_a.shape} vs {frame_b.shape}"
        )

    def make_frame(t):
        eased = _smoothstep(t / duration)
        blended = frame_a.astype(np.float32) * (1.0 - eased) + frame_b.astype(np.float32) * eased
        return blended.astype(np.uint8)

    _, _, _, _, VideoClip = _import_moviepy()
    return VideoClip(make_frame, duration=duration)


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

    _, _, _, CompositeVideoClip, _ = _import_moviepy()

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
    index: int = 0,
    dip_in_seconds: float = 0.0,
    dip_in_color: Tuple[int, int, int] = DIP_BLACK,
    dip_out_seconds: float = 0.0,
    dip_out_color: Tuple[int, int, int] = DIP_BLACK,
):
    """Create one slide sub-clip: image (optionally with Ken Burns) + voiceover audio.

    ``dip_in_seconds`` / ``dip_out_seconds`` optionally fade the clip's first
    / last frames toward solid colors (slide-change transitions). Dips apply
    to the final composited frames, after any Ken Burns motion, and never
    touch the audio track.
    """
    AudioFileClip, ImageClip, _, _, _ = _import_moviepy()

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
        if dip_in_seconds > 0 or dip_out_seconds > 0:
            video_track = _apply_dip(
                video_track,
                in_seconds=dip_in_seconds,
                in_color=dip_in_color,
                out_seconds=dip_out_seconds,
                out_color=dip_out_color,
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
    background_music_path: Optional[str] = None,
    bg_music_volume: float = DEFAULT_BG_MUSIC_VOLUME,
    transition: str = DEFAULT_TRANSITION_STYLE,
    transition_seconds: float = DEFAULT_TRANSITION_SECONDS,
) -> str:
    """Stitch per-slide clips into one MP4, with Ken Burns and slide transitions.

    Transition styles:

    - ``dip-black`` — boundary dips: the outgoing slide fades to black over
      ``transition_seconds / 2`` and the incoming slide fades back in over the
      next half, so the whole blackout lasts about ``transition_seconds``.
      The dips fit inside each slide's built-in trailing silence, so total
      runtime and narration timing stay unchanged.
    - ``dip-white`` — the same boundary dips, flashing through white instead.
    - ``crossfade`` — a silent still-dissolve clip bridging consecutive slides
      (adds ``transition_seconds`` per boundary to the runtime; narration is
      never covered).
    - ``none`` — hard cuts.

    Every style also fades the video out to black at the end over
    ``transition_seconds``. The first slide is always fully visible from
    frame 0 (no fade-in), so file-browser thumbnails and players show the
    slide content instead of a black frame.
    """
    if not png_paths:
        raise ValueError("No slide images to assemble")

    transition = (transition or "none").strip().lower()
    if transition not in TRANSITION_STYLES:
        raise ValueError(
            f"Unknown transition style {transition!r}; expected one of "
            f"{', '.join(TRANSITION_STYLES)}"
        )
    transition_seconds = max(0.0, float(transition_seconds))
    use_boundary_dips = (
        transition in ("dip-black", "dip-white") and transition_seconds > 0
    )
    use_crossfades = transition == "crossfade" and transition_seconds > 0
    use_closer = use_boundary_dips or use_crossfades

    _, _, concatenate_videoclips, _, _ = _import_moviepy()

    slide_clips = []
    boundary_clips = []
    audio_handles = []

    try:
        total_slides = len(png_paths)
        if use_closer:
            print(
                f"\n✨ Slide transitions: {transition} "
                f"({transition_seconds:.2f}s per change)"
            )
        for idx, (png_path, wav_path) in enumerate(zip(png_paths, wav_paths), start=1):
            trailing_pause = inter_slide_pause_seconds if idx < total_slides else 0.0
            is_first = idx == 1
            is_last = idx == total_slides

            # The closer fade runs through black at the full configured
            # length; boundary dips split it in half across the cut (outgoing
            # tail + incoming head) so the flash reads as one
            # ~transition_seconds-long effect. The first slide deliberately
            # gets NO head dip: it must be fully visible from frame 0 so
            # file-browser thumbnails, players, and previews show the slide
            # content instead of black. Crossfade slides only get the closer
            # dip — boundaries are separate dissolve clips.
            dip_in_seconds = 0.0
            dip_out_seconds = 0.0
            boundary_color = DIP_BLACK if transition == "dip-black" else DIP_WHITE
            dip_in_color = boundary_color
            dip_out_color = boundary_color
            if use_closer and is_last:
                dip_out_seconds = transition_seconds
                dip_out_color = DIP_BLACK
            if use_boundary_dips and not is_first:
                dip_in_seconds = transition_seconds / 2.0
            if use_boundary_dips and not is_last:
                dip_out_seconds = transition_seconds / 2.0

            print(f"   🎬 Building clip for slide {idx}...")
            slide_clip, audio_track = build_slide_clip(
                png_path,
                wav_path,
                fps=fps,
                trailing_pause_seconds=trailing_pause,
                ken_burns_zoom=ken_burns_zoom,
                index=idx,
                dip_in_seconds=dip_in_seconds,
                dip_in_color=dip_in_color,
                dip_out_seconds=dip_out_seconds,
                dip_out_color=dip_out_color,
            )
            slide_clips.append(slide_clip)
            if audio_track is not None:
                audio_handles.append(audio_track)

        # Crossfades are inserted between whole slide clips as silent
        # still-dissolve boundary clips, so 'chain' concatenation still works
        # (see the comment below for why 'compose' must stay off the table).
        clips_to_concat = slide_clips
        if use_crossfades:
            clips_to_concat = []
            for position, slide_clip in enumerate(slide_clips):
                clips_to_concat.append(slide_clip)
                if position < len(slide_clips) - 1:
                    print(f"   ✨ Crossfading slides {position + 1} → {position + 2}...")
                    boundary_clip = _build_crossfade_clip(
                        slide_clip,
                        slide_clips[position + 1],
                        transition_seconds,
                        fps,
                    )
                    if boundary_clip is not None:
                        clips_to_concat.append(boundary_clip)
                        boundary_clips.append(boundary_clip)

        # Every slide clip is rendered at the same fixed size (either the target
        # size for static slides or KEN_BURNS_IMAGE_SIZE for Ken Burns slides),
        # so 'chain' concatenation is safe: it passes each clip's frame straight
        # through. We must NOT use method='compose' here: compose re-centers each
        # already-cropped clip in a new composite with with_position('center'),
        # and that Pillow re-composite corrupts the most-zoomed-in frames of the
        # Ken Burns clips, exposing a black band on the pan side (observed on the
        # last zoom-in frame of slide 5).
        final_movie = concatenate_videoclips(clips_to_concat, method="chain")

        # Background music ([bgm: track]) spans slide boundaries, so unlike SFX
        # it cannot be baked into per-slide WAVs: layer it under the whole
        # assembled soundtrack here, right before rendering.
        if background_music_path:
            print(
                f"\n🎵 Mixing background music "
                f"{os.path.basename(background_music_path)!r} "
                f"(volume {bg_music_volume}) ..."
            )
            final_movie = bgm.attach_background_music(
                final_movie,
                background_music_path,
                volume=bg_music_volume,
                wav_paths=[p for p in wav_paths if p],
            )

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
        for clip in boundary_clips:
            _close_clip(clip)
        for audio in audio_handles:
            _close_clip(audio)
