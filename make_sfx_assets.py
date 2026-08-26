#!/usr/bin/env python3
"""Generate the default SFX sample in assets/ (ba_dum_tss_default.wav).

The rimshot is synthesized from scratch so the repo stays free of binary
licensing questions and the sample stays reproducible. It is only a fallback:
drop your own ``assets/ba_dum_tss.wav`` (e.g. converted from a licensed
recording) to take precedence — this script never overwrites that file.

  "ba" / "dum" — two low tom/snare hits (pitch-swept sine thump + noise snap)
  "tss"        — a decaying high-passed noise crash

Usage:
    python make_sfx_assets.py
"""

from __future__ import annotations

import os
import wave

import numpy as np

RATE = 44100
PEAK = 0.85


def _envelope(n: int, decay_tau: float) -> np.ndarray:
    return np.exp(-np.arange(n) / (decay_tau * RATE))


def _thump(duration: float, start_hz: float, end_hz: float) -> np.ndarray:
    """A drum-like pitch-swept sine thump."""
    n = int(duration * RATE)
    t = np.arange(n) / RATE
    freq = np.geomspace(start_hz, end_hz, n)
    phase = 2.0 * np.pi * np.cumsum(freq) / RATE
    attack = np.minimum(t / 0.004, 1.0)  # avoid a click at sample 0
    return np.sin(phase) * _envelope(n, duration * 0.38) * attack


def _snap(duration: float, rng: np.random.Generator) -> np.ndarray:
    """A bright stick/noise transient for the snare rattle."""
    n = int(duration * RATE)
    noise = rng.standard_normal(n)
    return np.diff(noise, prepend=0.0) * _envelope(n, duration * 0.22)


def _crash(duration: float, rng: np.random.Generator) -> np.ndarray:
    """A cymbal-ish wash: high-passed noise with a long decay."""
    n = int(duration * RATE)
    noise = rng.standard_normal(n)
    # One-pole high-pass keeps the shimmer, drops the rumble.
    hp = np.zeros(n)
    prev_in = 0.0
    for i in range(1, n):
        hp[i] = 0.94 * (hp[i - 1] + noise[i] - prev_in)
        prev_in = noise[i]
    wobble = 1.0 + 0.08 * np.sin(2.0 * np.pi * 7.0 * np.arange(n) / RATE)
    return hp * _envelope(n, duration * 0.24) * wobble


def make_ba_dum_tss() -> np.ndarray:
    rng = np.random.default_rng(1965)
    total = np.zeros(int(1.4 * RATE))

    def add(signal: np.ndarray, at_seconds: float) -> None:
        start = int(at_seconds * RATE)
        end = min(total.size, start + signal.size)
        total[start:end] += signal[: end - start]

    add(_thump(0.13, 175.0, 85.0), 0.000)   # "ba"
    add(_snap(0.05, rng), 0.000)
    add(_thump(0.15, 150.0, 70.0), 0.118)   # "dum"
    add(_snap(0.05, rng), 0.118)
    add(_crash(1.05, rng), 0.238)           # "tss"

    total /= max(np.abs(total).max(), 1e-9)
    return PEAK * total


def write_wav(path: str, samples: np.ndarray, rate: int = RATE) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pcm = np.clip(np.round(samples * 32767.0), -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm.tobytes())
    print(f"✅ wrote {path} ({len(samples) / rate:.2f}s @ {rate} Hz mono 16-bit)")


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    # Deliberately NOT ba_dum_tss.wav — a user-provided recording at the
    # canonical name always wins and must never be clobbered by a rebuild.
    write_wav(
        os.path.join(here, "assets", "ba_dum_tss_default.wav"), make_ba_dum_tss()
    )


if __name__ == "__main__":
    main()