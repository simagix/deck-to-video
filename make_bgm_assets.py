"""Generate the bundled background-music loop: ``assets/ambient_loop.mp3``.

Mirrors ``make_sfx_assets.py``: synthesizes a CC0-equivalent original pad
(a four-chord ambient progression with slow tremolo), writes a throwaway WAV,
and encodes the shipped MP3 via FFmpeg (libmp3lame). Background music rides
under the voiceover at DEFAULT_BG_MUSIC_VOLUME, so the source stays quiet and
mid-forward rather than bright or punchy.

Usage:  python make_bgm_assets.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import wave

import numpy as np

from paths import SYNTH_DIR

RATE = 44100
CHORD_SECONDS = 4.0
PEAK = 0.32

# Gentle low/mid voicings (Hz). Ending on the opening chord keeps whole-number
# tiling at the loop boundary consonant instead of clashing.
CHORDS = (
    ("Am", [(110.00, 0.9), (220.00, 0.7), (261.63, 0.55), (329.63, 0.45), (392.00, 0.3)]),
    ("F",  [(87.31, 0.9), (174.61, 0.7), (220.00, 0.55), (261.63, 0.45), (329.63, 0.3)]),
    ("C",  [(130.81, 0.85), (196.00, 0.65), (261.63, 0.5), (329.63, 0.42)]),
    ("G",  [(98.00, 0.9), (146.83, 0.65), (246.94, 0.5), (293.66, 0.4)]),
)


def _chord_audio(notes, n_samples):
    """Render one sustained chord: slightly detuned sine pairs + slow tremolo."""
    t = np.arange(n_samples) / RATE
    signal = np.zeros(n_samples)
    for freq, amp in notes:
        signal += amp * np.sin(2 * np.pi * freq * t)
        signal += amp * np.sin(2 * np.pi * freq * 1.003 * t)
    signal /= max(len(notes) * 2, 1)
    signal *= 1.0 - 0.12 * np.sin(2 * np.pi * 0.15 * t)  # breathing tremolo
    return signal


def _synthesize_loop():
    """Four-chord cycle overlapped with 0.8 s hann crossfades between chords."""
    total_n = int(RATE * CHORD_SECONDS * len(CHORDS))
    bed = np.zeros(total_n)
    span = int(RATE * CHORD_SECONDS)
    fade_n = min(int(0.8 * RATE), span // 4)

    for idx, (_, notes) in enumerate(CHORDS):
        start = idx * span
        is_last = idx == len(CHORDS) - 1
        # Non-final chords overhang into the next chord's attack region: that
        # overhang meets the next iteration's own hann-faded head, forming the
        # crossfade. The final chord stops flush so tiling restarts consonantly.
        end = min(start + (0 if is_last else fade_n) + span, total_n)
        chunk = _chord_audio(notes, end - start)
        window = np.ones(end - start)
        if fade_n:
            window[:fade_n] *= np.hanning(2 * fade_n)[:fade_n]
        bed[start:end] += chunk * window

    # Kill startup/teardown clicks so tiled repeats never pop.
    edge = min(int(0.03 * RATE), len(bed) // 8)
    bed[:edge] *= np.linspace(0.0, 1.0, edge)
    bed[-edge:] *= np.linspace(1.0, 0.0, edge)

    bed -= bed.mean()
    bed *= PEAK / max(np.max(np.abs(bed)), 1e-9)
    return bed


def _ffmpeg_exe() -> str:
    """Prefer a system FFmpeg; fall back to the imageio-ffmpeg bundled one."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    import imageio_ffmpeg  # type: ignore[import-untyped]

    return imageio_ffmpeg.get_ffmpeg_exe()


def main() -> int:
    out_mp3 = os.path.join(SYNTH_DIR, "assets", "ambient_loop.mp3")
    os.makedirs(os.path.dirname(out_mp3), exist_ok=True)

    loop = _synthesize_loop()
    pcm = np.clip(np.round(loop * 32767), -32768, 32767).astype(np.int16)
    # Dual-mono stereo: identical channels preserve a wide-but-safe image and
    # degrade cleanly when a mono-narrated deck folds everything down.
    pcm = np.repeat(pcm.reshape(-1, 1), 2, axis=1)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
    try:
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(RATE)
            wf.writeframes(pcm.tobytes())

        cmd = [
            _ffmpeg_exe(),
            "-y",
            "-loglevel", "error",
            "-i", wav_path,
            "-codec:a", "libmp3lame",
            "-b:a", "128k",
            out_mp3,
        ]
        print(f"🎼 Encoding {out_mp3} ...")
        subprocess.run(cmd, check=True)
    finally:
        os.unlink(wav_path)

    print(f"✅ Wrote {out_mp3} ({os.path.getsize(out_mp3)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())