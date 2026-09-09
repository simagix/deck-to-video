"""Sound-effect mixing for punchline cues (e.g. a rimshot after a joke).

A slide's speaker notes may carry a bracketed ``[sfx: rimshot]`` tag (see
``narration.parse_sfx_cues``). When a tagged slide's voiceover WAV is
generated, :func:`overlay_sfx_on_wav` mixes the sample directly into the
WAV right after the last spoken word, so each slide remains one
self-contained audio file and ``video_assembly`` needs no changes.
"""

from __future__ import annotations

import os
import wave

import numpy as np

from paths import (
    DEFAULT_SFX_BEAT_SECONDS,
    DEFAULT_SFX_GAIN,
    DEFAULT_VOICEOVER_TRAIL_SILENCE_SECONDS,
    SYNTH_DIR,
)

ASSETS_DIR = os.path.join(SYNTH_DIR, "assets")

#: Supported sound effects -> sample file candidates under ``ASSETS_DIR``.
#: The first existing file wins, so dropping e.g. ``ba_dum_tss.wav`` (a
#: licensed recording converted to WAV) overrides the synthesized default,
#: which ``make_sfx_assets.py`` keeps in ``ba_dum_tss_default.wav``.
SFX_SAMPLES = {
    "rimshot": ("ba_dum_tss.wav", "ba_dum_tss_default.wav"),
    # User-supplied recordings only; no synthesized fallbacks exist for these.
    "sad_trombone": ("sad_trombone.wav",),
    "drum_roll": ("drum_roll.wav",),
    "cha_ching": ("cha-ching.wav",),
    "door_slam": ("door_slam.wav",),
}


def resolve_sfx_path(name: str) -> str:
    """Return the path of the best available sample for *name*."""
    candidates = SFX_SAMPLES.get(name)
    if not candidates:
        raise ValueError(
            f"Unknown sound effect {name!r}. Supported: {', '.join(SFX_SAMPLES)}."
        )
    for filename in candidates:
        path = os.path.join(ASSETS_DIR, filename)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"No sample found for {name!r} in {ASSETS_DIR} "
        f"(looked for: {', '.join(candidates)}). "
        "Run make_sfx_assets.py to generate the default."
    )

#: Amplitude threshold (fraction of full scale, ≈ -40 dBFS) below which a
#: voiceover sample counts as trailing silence when locating end-of-speech.
SPEECH_END_THRESHOLD = 0.01

_DTYPE_BY_WIDTH = {1: np.int8, 2: np.int16, 4: np.int32}


def _wav_to_float_array(frames: bytes, sampwidth: int, nchannels: int) -> np.ndarray:
    """Convert raw WAV frames to a float64 array in [-1, 1], shape (n, channels)."""
    if sampwidth == 3:
        raise NotImplementedError(
            "24-bit WAV input is not supported for SFX mixing; "
            "re-generate the voiceover as 8/16/32-bit PCM."
        )
    try:
        dtype = _DTYPE_BY_WIDTH[sampwidth]
    except KeyError:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes") from None
    data = np.frombuffer(frames, dtype=dtype).astype(np.float64)
    data /= float(2 ** (8 * sampwidth - 1))
    return data.reshape(-1, nchannels)


def _float_array_to_wav(samples: np.ndarray, sampwidth: int) -> bytes:
    """Clamp a float64 array in [-1, 1] back to raw WAV frames of *sampwidth*."""
    dtype = _DTYPE_BY_WIDTH[sampwidth]
    scale = float(2 ** (8 * sampwidth - 1))
    clipped = np.clip(np.round(samples * scale), -scale, scale - 1)
    return clipped.astype(dtype).tobytes()


def _read_wav_float(path: str):
    """Read a WAV file; return (wave params, float64 array shaped (n, channels))."""
    with wave.open(path, "rb") as wf:
        params = wf.getparams()
        frames = wf.readframes(wf.getnframes())
    return params, _wav_to_float_array(frames, params.sampwidth, params.nchannels)


def _fit_to_format(
    samples: np.ndarray,
    src_rate: int,
    src_channels: int,
    dst_rate: int,
    dst_channels: int,
) -> np.ndarray:
    """Fold to mono, resample to *dst_rate*, and fan out to *dst_channels*."""
    mono = samples.reshape(-1, src_channels).mean(axis=1)
    target_n = max(1, int(round(len(mono) / src_rate * dst_rate)))
    mono = np.interp(
        np.linspace(0.0, len(mono) - 1, target_n),
        np.arange(len(mono), dtype=np.float64),
        mono,
    )
    if dst_channels > 1:
        return np.tile(mono[:, None], (1, dst_channels))
    return mono[:, None]


def load_sfx(
    name: str,
    framerate: int,
    nchannels: int,
    *,
    keep_source_channels: bool = False,
) -> np.ndarray:
    """Load an SFX sample resampled to match (framerate, nchannels).

    Returns float64 samples in [-1, 1] shaped ``(num_frames, nchannels)``.
    With ``keep_source_channels=True`` the sample keeps its own channel count
    instead of being folded to *nchannels* (used to preserve genuine stereo
    images; callers must fan the result out themselves if wider output is
    desired).
    """
    path = resolve_sfx_path(name)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"SFX sample missing: {path}. Run make_sfx_assets.py to generate it."
        )
    params, signal = _read_wav_float(path)
    dst_channels = params.nchannels if keep_source_channels else nchannels
    return _fit_to_format(signal, params.framerate, params.nchannels, framerate, dst_channels)


def find_speech_end_seconds(samples: np.ndarray, framerate: int) -> float:
    """Return the time (seconds) just past the last non-silent sample."""
    active = np.abs(samples).max(axis=1) > SPEECH_END_THRESHOLD
    if not active.any():
        return 0.0
    return (int(np.max(np.nonzero(active))) + 1) / framerate


def assemble_voiceover(
    parts,
    out_path: str,
    *,
    beat_seconds: float = DEFAULT_SFX_BEAT_SECONDS,
    gain: float = DEFAULT_SFX_GAIN,
    tail_seconds: float = DEFAULT_VOICEOVER_TRAIL_SILENCE_SECONDS,
) -> List[str]:
    """Stitch narration takes and SFX hits into one slide voiceover WAV.

    *parts* is an ordered sequence of ``("wav", path)`` narration takes and
    ``("sfx", name)`` effects. Each SFX is preceded by *beat_seconds* of
    silence (the comedic pause after the preceding take), so the hit lands
    exactly where the tag sat in the notes — no end-of-speech guessing.
    Takes are resampled to the first take's format if Voicebox ever returns
    mixed formats, and *tail_seconds* of silence closes the file.

    Channel handling: if any effect sample has more channels than the
    narration takes (e.g. a true-stereo rimshot on a mono voiceover), the
    output is upgraded to that channel count and the voice is duplicated
    across channels, so a wide stereo image survives the splice. Dual-mono
    samples stay untouched (folding them loses nothing).

    Returns the names of the SFX that were inserted (in order).
    """
    wav_parts = [path for kind, path in parts if kind == "wav"]
    if not wav_parts:
        raise ValueError("assemble_voiceover requires at least one narration take")
    params, _ = _read_wav_float(wav_parts[0])
    rate, channels = params.framerate, params.nchannels

    # Upgrade the output format when an effect is wider than the narration.
    for kind, value in parts:
        if kind == "sfx":
            sfx_params, _ = _read_wav_float(resolve_sfx_path(value))
            channels = max(channels, sfx_params.nchannels)

    chunks = []
    inserted: List[str] = []
    for kind, value in parts:
        if kind == "wav":
            src_params, signal = _read_wav_float(value)
            chunks.append(
                _fit_to_format(
                    signal,
                    src_params.framerate,
                    src_params.nchannels,
                    rate,
                    channels,
                )
            )
        elif kind == "sfx":
            chunks.append(np.zeros((int(round(beat_seconds * rate)), channels)))
            signal = load_sfx(value, rate, channels, keep_source_channels=True)
            if signal.shape[1] != channels:
                signal = _fit_to_format(signal, rate, signal.shape[1], rate, channels)
            chunks.append(gain * signal)
            inserted.append(value)
        else:
            raise ValueError(f"Unknown voiceover part kind: {kind!r}")
    chunks.append(np.zeros((int(round(tail_seconds * rate)), channels)))

    mixed = np.concatenate(chunks)
    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(params.sampwidth)
        wf.setframerate(rate)
        wf.writeframes(_float_array_to_wav(mixed, params.sampwidth))
    return inserted


def overlay_sfx_on_wav(
    wav_path: str,
    name: str = "rimshot",
    *,
    beat_seconds: float = DEFAULT_SFX_BEAT_SECONDS,
    gain: float = DEFAULT_SFX_GAIN,
    tail_seconds: float = DEFAULT_VOICEOVER_TRAIL_SILENCE_SECONDS,
) -> bool:
    """Mix an SFX into *wav_path* right after the final spoken word.

    The sample starts *beat_seconds* after end-of-speech (the comedic pause),
    and the WAV is extended so at least *tail_seconds* of silence follows the
    hit — preserving the built-in trailing-silence contract used by video
    assembly. Returns True when the overlay was applied.

    Only the first recognized cue per slide is honored; callers should not
    apply multiple effects sequentially (each call re-detects end-of-speech,
    which would stack them back-to-back).
    """
    if name not in SFX_SAMPLES:
        raise ValueError(
            f"Unknown sound effect {name!r}. Supported: {', '.join(SFX_SAMPLES)}."
        )
    with wave.open(wav_path, "rb") as wf:
        params = wf.getparams()
        frames = wf.readframes(wf.getnframes())

    nchannels, sampwidth, framerate = params[:3]
    voice = _wav_to_float_array(frames, sampwidth, nchannels)
    if voice.shape[0] == 0:
        return False

    sfx = load_sfx(name, framerate, nchannels)
    start = int(round((find_speech_end_seconds(voice, framerate) + beat_seconds) * framerate))
    total = start + sfx.shape[0] + int(round(tail_seconds * framerate))

    mixed = np.zeros((total, nchannels), dtype=np.float64)
    mixed[: voice.shape[0]] = voice
    mixed[start : start + sfx.shape[0]] += gain * sfx[: total - start]

    with wave.open(wav_path, "wb") as wf:
        wf.setparams(params)
        wf.writeframes(_float_array_to_wav(mixed, sampwidth))
    return True