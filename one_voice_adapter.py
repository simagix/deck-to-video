"""OneVoice adapter for deck-to-video — drop-in replacement for voicebox_client.

Provides the same interface as voicebox_client.generate_voicebox_audio() but
uses the one-voice local TTS engine instead of Voicebox HTTP API.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# one-voice is imported lazily so --help stays fast and the dependency is
# only needed when actually generating audio.
_one_voice_imported = False
_parse_script = None
_profile_reference = None
_synthesize = None
_resolve_default_voice = None


def _ensure_one_voice():
    """Import one-voice, raising a helpful error if not installed."""
    global _one_voice_imported, _parse_script, _profile_reference
    global _synthesize, _resolve_default_voice
    if _one_voice_imported:
        return
    try:
        from one_voice import (
            parse_script as _parse_script,
            _profile_reference as _profile_reference,
            _synthesize as _synthesize,
            resolve_default_voice as _resolve_default_voice,
        )
    except ImportError:
        raise RuntimeError(
            "one-voice is not installed. Install it with:\n"
            "  pip install one-voice@git+https://github.com/simagix/one-voice.git\n"
            "or use --voicebox to fall back to the Voicebox API."
        )
    _one_voice_imported = True


def get_one_voice_config(
    profile_id_override: Optional[str] = None,
    engine_override: Optional[str] = None,
) -> tuple[str, None]:
    """Resolve the voice name and engine for one-voice.

    ``profile_id_override`` is the voice name (or mapped from UUID via
    ``VOICE_PROFILE_MAP`` env var). ``engine_override`` is ignored —
    one-voice only uses Qwen3-TTS.

    Returns ``(voice_name, None)`` — engine is always None for one-voice.
    """
    voice = profile_id_override or os.environ.get("ONE_VOICE")
    if not voice:
        # Fall back to VOICEBOX_PROFILE_ID for backward compat, then default
        voice = os.environ.get("VOICEBOX_PROFILE_ID")
    if not voice:
        _ensure_one_voice()
        voice = _resolve_default_voice()
    return voice, None


def generate_one_voice_audio(
    text: str,
    profile_id: str,
    output_wav: str,
    api_base: str = "",
    timeout: float = 600,
    personality: bool = False,
    engine: Optional[str] = None,
    instruct: Optional[str] = None,
) -> str:
    """Generate narration audio via one-voice and save a .wav file.

    Mirrors the voicebox_client.generate_voicebox_audio() signature so it
    can be used as a drop-in replacement. Differences:

    - ``api_base`` and ``timeout`` are ignored (local generation, no HTTP).
    - ``personality`` is ignored (one-voice has no LLM rewrite).
    - ``engine`` is ignored (one-voice only uses Qwen3-TTS).
    - ``instruct`` is prepended to the text as a tone direction.
    """
    if not text or not text.strip():
        raise ValueError("Cannot generate audio from empty text")

    _ensure_one_voice()

    # Resolve voice name and reference audio
    voice_name = profile_id or _resolve_default_voice()
    ref_audio = _profile_reference(voice_name)

    # Build the text: prepend tone instruction if provided
    # one-voice uses in-band tone directions spoken by the model
    if instruct:
        # The instruct is a full sentence like "Speak in a warm, friendly way."
        # Prepend it so the model reads it as a direction
        text = f"{instruct} {text}"

    output_path = Path(output_wav)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"   🎙️  one-voice ({voice_name}): {len(text)} chars → {os.path.basename(output_wav)}")

    # one-voice generates the WAV file directly
    _synthesize(
        text=text,
        model="mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
        output=output_path,
        ref_audio=ref_audio,
        ref_text=None,
    )

    if not output_path.exists():
        raise RuntimeError(f"one-voice failed to produce audio: {output_wav}")

    return output_wav