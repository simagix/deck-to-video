"""Background-music support for ``[bgm: ...]`` speaker-note tags.

The tag itself is parsed in ``narration.py`` (alongside ``[sfx: ...]``); the
tag text never reaches the TTS. This module owns everything else:

- :func:`resolve_bgm_path` maps a tag reference to a real audio file — either
  an explicit path supplied in the note, or a bare filename looked up under
  ``assets/`` (SFX samples live there too; background music is just another
  bundled asset, shipped as ``.mp3``).

Unlike punchline SFX, background music is NOT baked into the per-slide WAVs:
it must play continuously across slide boundaries, so it is layered under the
assembled narration at video-assembly time (``bgm.attach_background_music``,
invoked from ``video_assembly``), looping when the track is shorter than the
timeline.
"""

from __future__ import annotations

import math
import os
import wave

import numpy as np

from typing import Tuple

from paths import DEFAULT_BG_MUSIC_VOLUME, SYNTH_DIR

ASSETS_DIR = os.path.join(SYNTH_DIR, "assets")

#: Extension order tried for a bare (extension-less) tag reference. ``.mp3``
#: is the conventional background-music format here; ``.wav`` stays accepted
#: for parity with the SFX samples in the same directory.
BGM_EXTENSIONS = (".mp3", ".wav")


def resolve_bgm_path(ref: str) -> str:
    """Resolve a ``[bgm: ...]`` tag reference to an existing audio file.

    Resolution order:

    1. The reference as an explicit filesystem path (tilde-expanded),
       absolute or relative to the current working directory.
    2. As a filename directly under ``assets/`` (exact match, e.g. a full
       ``ambient.mp3`` name).
    3. As an extension-less asset name, trying :data:`BGM_EXTENSIONS` in
       order (``ambient`` -> ``assets/ambient.mp3``, then ``.wav``).

    Raises ``FileNotFoundError`` listing every location attempted.
    """
    ref = (ref or "").strip()
    if not ref:
        raise ValueError("[bgm: ...] tag needs a track name or path")

    expanded = os.path.expanduser(ref)
    if os.path.isfile(expanded):
        return os.path.abspath(expanded)

    attempts: list[str] = []
    has_extension = bool(os.path.splitext(expanded)[1])
    names = [expanded] if has_extension else [expanded + ext for ext in BGM_EXTENSIONS]
    for name in names:
        candidate = os.path.join(ASSETS_DIR, name)
        attempts.append(candidate)
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        f"Background music {ref!r} not found. "
        f"Looked for: {', '.join(attempts)}. Drop the track under {ASSETS_DIR} "
        "or pass a path to the file."
    )


# ============================================================================
# Assembly-time mixing (MoviePy front-end, numpy mixing core)
# ============================================================================
# Philosophy mirrors sfx.py: every composite layer must share one sample
# format, otherwise MoviePy's mixer misaligns layers. The voiceover WAVs fix
# that format, so the track is decoded ONCE via FFmpeg straight into the
# narration's rate/channel layout; looping, trimming and gain are then plain
# numpy ops before wrapping the result in an AudioClip.

#: Assumed voiceover format when probing is impossible (no cached WAVs, e.g.
#: a fully silent deck scored only by background music).
VOICE_FALLBACK_RATE = 44100


def _clip_set_audio(clip, audio):
    """MoviePy 1.x/2.x compatible with_audio/set_audio."""
    if hasattr(clip, "with_audio"):
        return clip.with_audio(audio)
    return clip.set_audio(audio)


def probe_voice_format(wav_paths) -> Tuple[int, int]:
    """First available voiceover WAV decides the mix's ``(sample_rate, channels)``."""
    for path in wav_paths or []:
        if path and os.path.isfile(path):
            with wave.open(path, "rb") as wf:
                return wf.getframerate(), wf.getnchannels()
    return VOICE_FALLBACK_RATE, 1


def fit_channels(signal: np.ndarray, channels: int) -> np.ndarray:
    """Fold or duplicate *signal* (shape ``(n, ch)``) to exactly *channels*.

    Down-folding averages grouped columns when divisible (stereo->mono),
    falling back to mean-mono duplication for odd layouts — matching sfx.py's
    dual-mono philosophy where up-scaling duplicates identical channels so a
    wide image survives without inventing stereo content.
    """
    src = signal.shape[1]
    if src == channels:
        return signal
    if src > channels:
        if src % channels == 0:
            grouped = signal.reshape(len(signal), channels, src // channels)
            return grouped.mean(axis=2)
        mono = signal.mean(axis=1, keepdims=True)
        return np.repeat(mono, channels, axis=1)
    return np.repeat(signal, channels, axis=1)


def tile_to_samples(signal: np.ndarray, total_samples: int) -> np.ndarray:
    """Whole-number tiling trimmed to exactly *total_samples* rows.

    Repeats always start on the track's own boundary rather than cross-fading
    a mid-phrase cut; the final trim lands wherever the timeline ends.
    """
    n_rows = signal.shape[0]
    if total_samples <= 0 or n_rows == 0:
        return np.zeros((max(total_samples, 0), signal.shape[1]), dtype=np.float64)
    repeats = int(math.ceil(total_samples / n_rows))
    tiled = np.tile(signal, (repeats, 1))
    return tiled[:total_samples]


def _read_pcm_wav_frames(path):
    """Decode an uncompressed PCM WAV via stdlib to float64 ``(n, channels)``.

    Values land in [-1, 1]; the native sample rate is returned alongside so
    callers can resample separately.
    """
    import wave as _wave

    with _wave.open(path, "rb") as wf:
        sampwidth, channels = wf.getsampwidth(), wf.getnchannels()
        rate = wf.getframerate()
        payload = wf.readframes(wf.getnframes())
    if channels <= 0:
        raise ValueError(f"{path} reports zero channels")
    if sampwidth == 2:
        data = np.frombuffer(payload, dtype="<i2").astype(np.float64) / 32768.0
    elif sampwidth == 4:
        data = np.frombuffer(payload, dtype="<i4").astype(np.float64) / 2147483648.0
    elif sampwidth == 1:
        data = (np.frombuffer(payload, dtype="u1").astype(np.float64) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported PCM width ({sampwidth} bytes) in {path}")
    rows = data.size // channels
    return data[: rows * channels].reshape(rows, channels), rate


def _resample_rows(signal: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear-interpolation resample along time (per channel)."""
    if src_rate == dst_rate or signal.shape[0] == 0:
        return signal
    src_n = signal.shape[0]
    dst_n = max(int(round(src_n * float(dst_rate) / float(src_rate))), 1)
    positions = np.linspace(0.0, src_n - 1.0, num=dst_n)
    columns = [
        np.interp(positions, np.arange(src_n), signal[:, ch])
        for ch in range(signal.shape[1])
    ]
    return np.stack(columns, axis=1)


def _ffmpeg_resampled_frames(path: str, sample_rate: int, channels: int) -> np.ndarray:
    """Decode any supported input straight to ``(n, channels)`` float32 PCM.

    One FFmpeg subprocess emits raw little-endian samples already conforming
    to the requested layout — sidestepping MoviePy's chunked audio reader,
    whose end-of-file handling varies across versions.
    """
    import shutil as _shutil
    import subprocess as _subprocess

    ffmpeg = _shutil.which("ffmpeg")
    if not ffmpeg:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    proc = _subprocess.run(
        [
            ffmpeg, "-v", "error", "-i", path, "-vn", "-sn", "-dn",
            "-f", "f32le", "-acodec", "pcm_f32le",
            "-ar", str(int(sample_rate)), "-ac", str(int(channels)),
            "-",
        ],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout:
        detail = proc.stderr.decode("utf-8", "replace").strip()[-400:]
        raise ValueError(f"FFmpeg could not decode {path!r}: {detail}")
    flat = np.frombuffer(proc.stdout, dtype="<f4").astype(np.float64)
    rows = flat.size // int(channels)
    if rows == 0:
        raise ValueError(f"{path} decoded to zero audio frames")
    return flat[: rows * int(channels)].reshape(rows, int(channels))


def decode_music_array(music_path: str, sample_rate: int, channels: int) -> np.ndarray:
    """Decode the full track to float64 frames shaped ``(n, channels)`` at *rate*.

    Routing keeps the core pure numpy wherever possible: an uncompressed PCM
    WAV goes through the stdlib ``wave`` module plus linear resampling, while
    every other container (MP3/M4A/OGG…) is converted by ONE FFmpeg process
    directly into the target rate/channel layout. Either way, looping,
    trimming and gain downstream are plain array math, and MoviePy never
    appears near the audio reader (whose boundary behavior differs between
    releases).

    Memory note: peak footprint is roughly duration_seconds * sample_rate *
    channels * 8 bytes (a 3-minute stereo 44.1kHz track is ~76 MB) — fine for
    presentation-length scoring; pruned segment decoding is future work.
    """
    if str(music_path).lower().endswith(".wav"):
        import wave as _wave

        try:
            signal, wav_rate = _read_pcm_wav_frames(music_path)
            signal = _resample_rows(signal, wav_rate, sample_rate)
            return fit_channels(signal, int(channels))
        except (_wave.Error, EOFError, ValueError):
            pass  # mislabeled container or exotic codec — defer to FFmpeg
    return _ffmpeg_resampled_frames(music_path, sample_rate, channels)


def build_music_clip(
    music_path: str,
    duration: float,
    *,
    sample_rate: int,
    channels: int,
    volume: float = DEFAULT_BG_MUSIC_VOLUME,
):
    """Precomputed, looped, gain-applied soundtrack wrapped as an AudioClip."""
    if duration <= 0:
        raise ValueError("duration must be positive")
    if volume < 0:
        raise ValueError("volume cannot be negative")

    signal = decode_music_array(music_path, sample_rate, channels)
    if signal.size == 0:
        raise ValueError(f"{music_path} contains no decodable audio")
    signal = signal * volume

    total_samples = int(round(duration * sample_rate))
    looped = tile_to_samples(signal, total_samples)

    from moviepy.audio.AudioClip import AudioClip

    def make_frame(t):
        scalar = np.isscalar(t)
        times = np.atleast_1d(np.asarray(t, dtype=np.float64))
        idx = np.clip((times * sample_rate).astype(np.int64), 0, len(looped) - 1)
        frame = looped[idx]
        return frame[0] if scalar else frame

    return AudioClip(
        frame_function=make_frame,
        duration=float(duration),
        fps=int(sample_rate),
    )


def attach_background_music(
    final_movie,
    music_path: str,
    *,
    volume: float = DEFAULT_BG_MUSIC_VOLUME,
    wav_paths=None,
):
    """Layer looped background music underneath *final_movie*'s soundtrack.

    Probing order: existing voiceover WAVs fix the mix format; otherwise the
    narration clip's own fps is honored; a bare deck (no WAVs at all) falls
    back to VOICE_FALLBACK_RATE. Works both under voiced decks — where
    CompositeAudioClip puts the music behind the concatenated narration+SFX
    track — and silent-only decks with no narration audio at all.

    The returned clip shares readers with *final_movie*: close ONLY the
    returned clip afterwards.
    """
    from moviepy.audio.AudioClip import CompositeAudioClip

    duration = getattr(final_movie, "duration", None)
    if not duration:
        raise ValueError("Cannot score a clip without a known duration")

    sample_rate, channels = probe_voice_format(wav_paths)
    narr_audio = final_movie.audio
    narr_fps = getattr(narr_audio, "fps", None)
    if narr_fps:
        sample_rate = int(narr_fps)

    music_clip = build_music_clip(
        music_path,
        duration,
        sample_rate=sample_rate,
        channels=channels,
        volume=volume,
    )

    if narr_audio is None:
        return _clip_set_audio(final_movie, music_clip)

    combined = CompositeAudioClip([music_clip, narr_audio])
    return _clip_set_audio(final_movie, combined)
