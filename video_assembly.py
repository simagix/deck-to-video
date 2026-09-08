"""FFmpeg helpers to assemble slide images and voiceover audio into MP4.

The assembly pipeline is pure FFmpeg (no MoviePy):

1. Each slide becomes one independently encoded, VIDEO-ONLY segment
   (``libx264``). Plain slides are scaled to the target size; slides with
   Ken Burns get a ``zoompan`` zoom+pan computed from the same smoothstep
   easing and pan geometry the original MoviePy implementation used
   (supersampled 2x so the integer-quantized ``zoompan`` window still moves
   in half-pixel steps at output resolution).
2. Slide-change transitions are baked into the segments as head/tail ``fade``
   dips (``dip-black`` / ``dip-white``), or rendered as separate
   still-dissolve boundary segments (``crossfade``).
3. Segments are joined with the concat demuxer and ``-c copy`` — a lossless
   stream copy with no generation loss.
4. The narration track is built once from the voiceover WAVs (concat filter,
   plain PCM), optional background music is decoded/looped/gained by
   ``bgm.write_music_wav``, and both are layered with a single ``amix`` pass
   in the final mux — the only audio encode in the whole pipeline.

Keeping segments video-only keeps every concat boundary on an exact frame
edge (AAC priming inside per-segment MP4s would otherwise drift the timeline)
and lets segments render as independent parallel FFmpeg processes: the stitch
step is roughly an order of magnitude faster than the previous MoviePy
per-frame Python loop, at equivalent quality.
"""

from __future__ import annotations

import math
import os
import random
import shutil
import subprocess
import tempfile
import wave
from typing import List, Optional, Tuple

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

#: Supported slide-change transition styles (``--transition``).
TRANSITION_STYLES = ("none", "dip-black", "dip-white", "crossfade")

#: Solid colors the transitions dip through (FFmpeg ``fade`` color names).
#: The default 'dip-black' style — and the video's closing fade in every dip
#: or crossfade style — fades through black like any film or show would; the
#: optional 'dip-white' style flashes toward white.
DIP_WHITE = "white"
DIP_BLACK = "black"

#: Supersampling factor for the Ken Burns ``zoompan`` crop. ``zoompan``
#: quantizes its window position to whole input pixels; rendering from a 2x
#: input halves the quantization step to half an output pixel — visually
#: equivalent to the old float-accurate MoviePy path, at negligible cost.
KEN_BURNS_SUPERSAMPLE = 2

#: A/V encode settings shared by every segment (and the crossfade boundary
#: segments), so the concat demuxer can stream-copy them together. These
#: match MoviePy's former defaults (libx264, preset medium, crf 23).
X264_ARGS = (
    "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p",
)
AAC_ARGS = ("-c:a", "aac", "-b:a", "192k")


def _ffmpeg_exe() -> str:
    """Locate an FFmpeg binary: PATH first, then the imageio-ffmpeg fallback."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg  # optional fallback, see requirements.txt

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    raise RuntimeError(
        "ffmpeg not found on PATH and imageio-ffmpeg is not installed. "
        "Install FFmpeg (e.g. `brew install ffmpeg`) or `pip install imageio-ffmpeg`."
    )


def _run_ffmpeg(*args: str) -> None:
    """Run one FFmpeg invocation, failing loudly on a non-zero exit."""
    subprocess.run(
        [_ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", *args],
        check=True,
    )


def _wav_duration(path: str) -> float:
    """Duration of a PCM WAV in seconds (the voiceover clip format)."""
    with wave.open(path, "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def _probe_voice_format(wav_paths) -> Optional[Tuple[int, int]]:
    """(sample_rate, channels) of the first available voiceover WAV, or None.

    None means a fully silent deck (no WAVs at all): segments are produced
    without audio tracks and background music becomes the sole soundtrack.
    """
    for path in wav_paths or []:
        if path and os.path.isfile(path):
            try:
                with wave.open(path, "rb") as wf:
                    return wf.getframerate(), wf.getnchannels()
            except wave.Error:
                continue  # non-PCM narration; treat as absent for format probing
    return None


def _ken_burns_filter(
    duration: float,
    fps: int,
    zoom: float,
    index: int,
    pan_angle: float,
    pan_ratio: float = 0.5,
) -> str:
    """``zoompan`` filter chain reproducing the original MoviePy Ken Burns move.

    Both the zoom and the pan are driven by ONE smoothstep-eased parameter
    ``e`` (``t²·(3-2t)``); zoom alternates in/out per slide (even index zooms
    out); the pan window sweeps monotonically along a single random heading
    ``pan_angle`` (``pan_ratio`` bounds how far it travels). The crop window
    starts at :data:`KEN_BURNS_IMAGE_SIZE` — slightly smaller than the target
    so the zoomed image always covers the frame — exactly like the former
    resize+composite path, and is finished with a lanczos scale to
    :data:`TARGET_IMAGE_SIZE`.

    Filter-graph details: with a single looped input image, ``on`` counts the
    output frames, so ``e`` is a function of ``on/(frames-1)``;
    ``zoompan``'s ``x``/``y`` are the crop window's top-left corner in
    (supersampled) source pixels — the negative of the old composite
    position, divided by the zoom factor; ``zoom`` itself is scaled by
    ``iw / KEN_BURNS_WIDTH`` because MoviePy cropped a KEN_BURNS_IMAGE_SIZE
    window out of a TARGET_IMAGE_SIZE image.
    """
    frames = max(int(round(duration * fps)), 1)
    u = f"(on/{max(frames - 1, 1)})"
    e = f"(({u})*({u})*(3-2*({u})))"
    if index % 2 == 0:  # zoom out (matches the old index % 2 == 0 branch)
        z = f"({zoom}-({zoom}-1)*{e})"
    else:  # zoom in
        z = f"(1+({zoom}-1)*{e})"
    cos_a, sin_a = math.cos(pan_angle), math.sin(pan_angle)
    ss = KEN_BURNS_SUPERSAMPLE
    kb_w, kb_h = KEN_BURNS_IMAGE_SIZE
    out_w, out_h = TARGET_IMAGE_SIZE
    x = f"{ss}*{kb_w / 2}*({z}-1)*(1-{pan_ratio}*(2*{e}-1)*{cos_a:.6f})/({z})"
    y = f"{ss}*{kb_h / 2}*({z}-1)*(1-{pan_ratio}*(2*{e}-1)*{sin_a:.6f})/({z})"
    return (
        f"scale={kb_w * ss}:{kb_h * ss},"
        f"zoompan=z='{z}':x='{x}':y='{y}':"
        f"d={frames}:s={kb_w}x{kb_h}:fps={fps},"
        f"scale={out_w}:{out_h}:flags=lanczos"
    )


def _dip_windows(
    transition: str,
    transition_seconds: float,
    is_first: bool,
    is_last: bool,
) -> Tuple[float, str, float, str]:
    """(in_seconds, in_color, out_seconds, out_color) for one slide segment.

    The closer fade runs through black at the full configured length;
    boundary dips split it in half across the cut (outgoing tail + incoming
    head) so the flash reads as one ~transition_seconds-long effect. The
    first slide deliberately gets NO head dip: it must be fully visible from
    frame 0 so file-browser thumbnails, players, and previews show the slide
    content instead of black. Crossfaded slides only get the closer dip —
    boundaries are separate dissolve segments.
    """
    dip_in_s = dip_out_s = 0.0
    dip_in_color = dip_out_color = DIP_BLACK
    boundary_style = transition in ("dip-black", "dip-white")
    use_closer = (boundary_style or transition == "crossfade") and transition_seconds > 0
    if not use_closer:
        return dip_in_s, dip_in_color, dip_out_s, dip_out_color
    boundary = DIP_BLACK if transition == "dip-black" else DIP_WHITE
    if is_last:
        dip_out_s = transition_seconds
        dip_out_color = DIP_BLACK
    if boundary_style and not is_first:
        dip_in_s = transition_seconds / 2.0
        dip_in_color = boundary
    if boundary_style and not is_last:
        dip_out_s = transition_seconds / 2.0
        dip_out_color = boundary
    return dip_in_s, dip_in_color, dip_out_s, dip_out_color


def _fade_filters(
    dip_in_seconds: float,
    dip_in_color: str,
    dip_out_seconds: float,
    dip_out_color: str,
    duration: float,
) -> str:
    """Head/tail ``fade`` filters for one segment ('' when there are none)."""
    fades = ""
    if dip_in_seconds > 0:
        fades += f"fade=t=in:st=0:d={dip_in_seconds}:color={dip_in_color},"
    if dip_out_seconds > 0:
        start = max(duration - dip_out_seconds, 0.0)
        fades += f"fade=t=out:st={start:.6f}:d={dip_out_seconds}:color={dip_out_color},"
    return fades


def _build_segment(
    png_path: str,
    out_path: str,
    duration: float,
    fps: int,
    ken_burns_zoom: float,
    index: int,
    pan_angle: float,
    dip_in_seconds: float,
    dip_in_color: str,
    dip_out_seconds: float,
    dip_out_color: str,
) -> None:
    """Encode one VIDEO-ONLY slide segment (image + optional Ken Burns).

    Segments carry no audio: joining video-only streams keeps every boundary
    on an exact frame edge, and the whole narration is built and muxed once
    in the final pass (see :func:`_build_narration_track` / :func:`_mux_final`)
    instead of being re-timestamped per segment by the concat demuxer.
    """
    out_w, out_h = TARGET_IMAGE_SIZE
    fades = _fade_filters(
        dip_in_seconds, dip_in_color, dip_out_seconds, dip_out_color, duration
    )
    if ken_burns_zoom > 0:
        framerate = "1"
        video = _ken_burns_filter(duration, fps, ken_burns_zoom, index, pan_angle) + "," + fades
    else:
        framerate = str(fps)
        video = f"scale={out_w}:{out_h}:flags=lanczos," + fades
    video += "format=yuv420p[v]"

    _run_ffmpeg(
        "-loop", "1", "-framerate", framerate, "-i", png_path,
        "-filter_complex", f"[0:v]{video}",
        "-map", "[v]",
        "-t", f"{duration:.6f}",
        *X264_ARGS,
        out_path,
    )



def _extract_frame(segment: str, fps: int, last: bool, out_png: str) -> None:
    """Dump one frame of an encoded segment (last, or first) as a PNG."""
    if last:
        # The last frame STARTS (1/fps)s before the segment ends; aiming
        # halfway between the second-to-last and last frame start times
        # captures exactly the final frame.
        seek = 1.5 / fps
        _run_ffmpeg(
            "-sseof", f"-{seek:.6f}", "-i", segment,
            "-frames:v", "1", "-update", "1", out_png,
        )
    else:
        _run_ffmpeg("-i", segment, "-frames:v", "1", "-update", "1", out_png)


def _build_crossfade_segment(
    outgoing_segment: str,
    incoming_segment: str,
    out_path: str,
    duration: float,
    fps: int,
) -> None:
    """Silent VIDEO-ONLY still-dissolve segment bridging two slide segments.

    Freezes the outgoing segment's final rendered frame and dissolves it into
    the incoming segment's first frame (alpha fade-in over the base frame) —
    the same still-to-still dissolve Keynote/PowerPoint use between slides.
    Freezing keeps Ken Burns motion continuous — each slide's zoom/pan plays
    exactly once, with no rewind stutter — and the incoming slide's
    narration starts unclipped after the dissolve (the narration track gets
    matching silence for this boundary).
    """
    with tempfile.TemporaryDirectory(prefix="d2v_xfade_") as tmp:
        frame_a = os.path.join(tmp, "last.png")
        frame_b = os.path.join(tmp, "first.png")
        _extract_frame(outgoing_segment, fps, last=True, out_png=frame_a)
        _extract_frame(incoming_segment, fps, last=False, out_png=frame_b)
        _run_ffmpeg(
            "-loop", "1", "-framerate", str(fps), "-i", frame_a,
            "-loop", "1", "-framerate", str(fps), "-i", frame_b,
            "-filter_complex",
            "[1:v]format=rgba,fade=t=in:st=0:d={:.6f}:alpha=1[b];"
            "[0:v][b]overlay=format=auto,format=yuv420p[v]".format(duration),
            "-map", "[v]",
            "-t", f"{duration:.6f}",
            *X264_ARGS,
            out_path,
        )


def _concat_segments(segment_paths: List[str], output_path: str, workdir: str) -> None:
    """Stream-copy every segment into one MP4 via the concat demuxer.

    All segments share codec, size, pixel format and audio layout, so
    ``-c copy`` joins them losslessly — no re-encode, no generation loss.
    """
    list_path = os.path.join(workdir, "concat.txt")
    with open(list_path, "w", encoding="utf-8") as fh:
        for segment in segment_paths:
            fh.write(f"file '{segment}'\n")
    _run_ffmpeg("-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", output_path)


def _build_narration_track(
    audio_entries: List[Tuple[str, str]],
    audio_format: Tuple[int, int],
    workdir: str,
) -> Optional[str]:
    """Concatenate voiceover WAVs (and silences) into ONE narration track.

    *audio_entries* is an ordered list of ``("wav", path)`` and
    ``("silence", seconds)`` tuples covering the whole timeline — one entry
    per slide, plus an explicit silence for every crossfade boundary.

    The result stays plain PCM (16-bit WAV at the voiceover's rate/channels);
    AAC encoding happens exactly once, in :func:`_mux_final`. Returns the
    track path, or None for a fully silent deck (no WAV entries at all).
    """
    if not any(kind == "wav" for kind, _ in audio_entries):
        return None
    rate, channels = audio_format
    layout = "mono" if channels == 1 else "stereo"
    inputs: List[str] = []
    labels: List[str] = []
    for kind, value in audio_entries:
        if kind == "wav":
            inputs += ["-i", value]
        else:
            inputs += [
                "-f", "lavfi", "-t", f"{float(value):.6f}",
                "-i", f"anullsrc=r={rate}:cl={layout}",
            ]
        labels.append(f"[{len(labels)}:a]")
    track = os.path.join(workdir, "narration.wav")
    graph = "".join(labels) + f"concat=n={len(labels)}:v=0:a=1[a]"
    _run_ffmpeg(*inputs, "-filter_complex", graph, "-map", "[a]", track)
    return track


def _mux_final(
    video_path: str,
    narration_wav: Optional[str],
    music_wav: Optional[str],
    output_path: str,
    duration: float,
) -> None:
    """Mux the assembled video with its soundtrack(s) — the only audio encode.

    The narration (already in timeline order) and the looped/gained music
    are layered with ``amix`` in ``normalize=0`` mode: the music's volume was
    applied during decode (``bgm.write_music_wav``), so the narration keeps
    its original level. Video is stream-copied untouched.
    """
    args = ["-i", video_path]
    maps = ["-map", "0:v"]
    graph = ""
    if narration_wav:
        args += ["-i", narration_wav]
        if music_wav:
            args += ["-i", music_wav]
            graph = "[1:a][2:a]amix=inputs=2:duration=first:normalize=0[a]"
            maps += ["-map", "[a]"]
        else:
            maps += ["-map", "1:a"]
    else:
        args += ["-i", music_wav]
        maps += ["-map", "1:a"]
    if graph:
        args += ["-filter_complex", graph]
    args += [
        *maps,
        "-c:v", "copy", *AAC_ARGS,
        "-t", f"{duration:.6f}",
        output_path,
    ]
    _run_ffmpeg(*args)


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
    """Stitch per-slide segments into one MP4, with Ken Burns and transitions.

    Transition styles:

    - ``dip-black`` — boundary dips: the outgoing slide fades to black over
      ``transition_seconds / 2`` and the incoming slide fades back in over
      the next half, so the whole blackout lasts about ``transition_seconds``.
      The dips fit inside each slide's built-in trailing silence, so total
      runtime and narration timing stay unchanged.
    - ``dip-white`` — the same boundary dips, flashing through white instead.
    - ``crossfade`` — a silent still-dissolve segment bridging consecutive
      slides (adds ``transition_seconds`` per boundary to the runtime;
      narration is never covered).
    - ``none`` — hard cuts.

    Every dip/crossfade style also fades the video out to black at the end
    over ``transition_seconds``. The first slide is always fully visible from
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

    audio_format = _probe_voice_format(wav_paths)
    total_slides = len(png_paths)
    if use_boundary_dips or use_crossfades:
        print(
            f"\n✨ Slide transitions: {transition} "
            f"({transition_seconds:.2f}s per change)"
        )

    with tempfile.TemporaryDirectory(prefix="d2v_assemble_") as workdir:
        segments: List[str] = []
        audio_entries: List[Tuple[str, str]] = []
        total_duration = 0.0
        for idx, (png_path, wav_path) in enumerate(zip(png_paths, wav_paths), start=1):
            if wav_path and os.path.isfile(wav_path):
                # Voiceover WAVs end with a built-in 1s trailing silence, which
                # serves as the natural gap before the next slide — don't add
                # a separate inter-slide pause on top of it.
                duration = _wav_duration(wav_path)
                audio_entries.append(("wav", wav_path))
            else:
                pause = inter_slide_pause_seconds if idx < total_slides else 0.0
                duration = DEFAULT_SILENT_SLIDE_SECONDS + max(0.0, pause)
                audio_entries.append(("silence", duration))
            is_first, is_last = idx == 1, idx == total_slides
            dip_in_s, dip_in_color, dip_out_s, dip_out_color = _dip_windows(
                transition, transition_seconds, is_first, is_last
            )
            pan_angle = random.uniform(0, 2 * math.pi)
            segment = os.path.join(workdir, f"seg_{idx:02d}.mp4")
            print(f"   🎬 Building clip for slide {idx}...")
            _build_segment(
                png_path, segment, duration, fps, ken_burns_zoom, idx,
                pan_angle, dip_in_s, dip_in_color, dip_out_s, dip_out_color,
            )
            segments.append(segment)
            total_duration += duration
            if use_crossfades and idx < total_slides:
                # The dissolve bridges silently; the narration track gets the
                # matching silence so no voiceover is clipped or delayed.
                audio_entries.append(("silence", transition_seconds))

        if use_crossfades:
            bridged: List[str] = []
            for position, segment in enumerate(segments):
                bridged.append(segment)
                if position < len(segments) - 1:
                    print(
                        f"   ✨ Crossfading slides {position + 1} → {position + 2}..."
                    )
                    boundary = os.path.join(workdir, f"xfade_{position + 1:02d}.mp4")
                    _build_crossfade_segment(
                        segment, segments[position + 1], boundary,
                        transition_seconds, fps,
                    )
                    bridged.append(boundary)
                    total_duration += transition_seconds
            segments = bridged

        # Background music ([bgm: track]) spans slide boundaries, so unlike
        # SFX it cannot be baked into per-slide WAVs: layer it under the
        # whole assembled soundtrack here, after the lossless concat.
        narration = _build_narration_track(audio_entries, audio_format, workdir) \
            if audio_format else None
        music = None
        if background_music_path:
            print(
                f"\n🎵 Mixing background music "
                f"{os.path.basename(background_music_path)!r} "
                f"(volume {bg_music_volume}) ..."
            )
            music = os.path.join(workdir, "bgm.wav")
            sample_rate, channels = audio_format or (bgm.VOICE_FALLBACK_RATE, 1)
            bgm.write_music_wav(
                background_music_path,
                total_duration,
                sample_rate=sample_rate,
                channels=channels,
                volume=bg_music_volume,
                out_path=music,
            )

        if narration or music:
            assembled = os.path.join(workdir, "assembled_video.mp4")
            _concat_segments(segments, assembled, workdir)
            _mux_final(assembled, narration, music, output_mp4, total_duration)
        else:
            _concat_segments(segments, output_mp4, workdir)

        # Output is always TARGET_IMAGE_SIZE: plain slides are scaled to it,
        # and Ken Burns crops (a KEN_BURNS_IMAGE_SIZE window) are finished
        # with a lanczos scale — the frame geometry the old MoviePy pipeline
        # intended, without its accidental first-clip-size output.
        print(f"\n📼 Wrote {output_mp4}")
        return output_mp4

