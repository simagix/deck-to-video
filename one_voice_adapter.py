"""OneVoice adapter for deck-to-video — drop-in replacement for voicebox_client.

Provides the same interface as voicebox_client.generate_voicebox_audio() but
uses the one-voice local TTS engine instead of Voicebox HTTP API.

Voice search path:
- deck-to-video's own voices/ directory (for project-specific voices like Simone)
- one-voice's voices/ directory (for base voices like Golding and Neufeld)

Set ONE_VOICE_PATH before import to customize, or let _ensure_one_voice() configure
it automatically based on this file's location.
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
_trim_tone_leak = None

# This file's directory — deck-to-video's root
_DECK_ROOT = Path(__file__).resolve().parent
# deck-to-video's custom voices (checked first)
_CUSTOM_VOICES = _DECK_ROOT / "voices"


def _setup_voice_search_path() -> None:
    """Configure ONE_VOICE_PATH to include both custom and one-voice voices.

    Priority (leftmost wins):
    1. deck-to-video/voices/ — project-specific voices (e.g. Simone)
    2. one-voice/voices/ — base voices (Golding, Neufeld)

    Only sets the path if ONE_VOICE_PATH is not already configured by the user.
    """
    if os.environ.get("ONE_VOICE_PATH"):
        return  # User has configured it explicitly

    # Find one-voice's voices directory
    # Try: pip-installed package, then common development locations
    one_voice_voices = None

    # Check if one_voice is installed as a package
    try:
        import one_voice

        one_voice_root = Path(one_voice.__file__).resolve().parent
        candidate = one_voice_root / "voices"
        if candidate.is_dir():
            one_voice_voices = candidate
    except ImportError:
        pass

    # Fallback: development location (sibling directory)
    if one_voice_voices is None:
        candidate = _DECK_ROOT.parent / "one-voice" / "voices"
        if candidate.is_dir():
            one_voice_voices = candidate

    # Build the search path
    paths = []
    if _CUSTOM_VOICES.is_dir():
        paths.append(str(_CUSTOM_VOICES))
    if one_voice_voices is not None:
        paths.append(str(one_voice_voices))

    if paths:
        os.environ["ONE_VOICE_PATH"] = os.pathsep.join(paths)


def _ensure_one_voice():
    """Import one-voice, raising a helpful error if not installed.

    Configures the voice search path before importing so one-voice can find
    voices from both deck-to-video and its own installation.
    """
    global _one_voice_imported, _parse_script, _profile_reference
    global _synthesize, _resolve_default_voice, _trim_tone_leak
    if _one_voice_imported:
        return

    # Configure search path BEFORE importing one-voice
    _setup_voice_search_path()

    try:
        from one_voice import (
            parse_script as _parse_script,
            _profile_reference as _profile_reference,
            _synthesize as _synthesize,
            resolve_default_voice as _resolve_default_voice,
            trim_tone_leak as _trim_tone_leak,
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
    language: str = "en",
) -> str:
    """Generate narration audio via one-voice and save a .wav file.

    Mirrors the voicebox_client.generate_voicebox_audio() signature so it
    can be used as a drop-in replacement. Differences:

    - ``api_base`` and ``timeout`` are ignored (local generation, no HTTP).
    - ``personality`` is ignored (one-voice has no LLM rewrite).
    - ``engine`` is ignored (one-voice only uses Qwen3-TTS).
    - ``instruct`` is prepended to the text as a tone direction.
    - ``language`` is ``"en"``/``"zh"`` and maps to mlx-audio
      ``lang_code="english"``/``"chinese"`` for the Qwen3-TTS voice clone.
    """
    if not text or not text.strip():
        raise ValueError("Cannot generate audio from empty text")

    _ensure_one_voice()

    # Resolve voice name and reference audio
    voice_name = profile_id or _resolve_default_voice()
    ref_audio = _profile_reference(voice_name)

    # Explicit per-slide language beats mlx-audio's "auto" for Mandarin:
    # verified that lang_code="chinese" pronounces 宫保鸡丁 correctly while
    # "auto"/"english" both render 公保鸡丁.
    from narration import mlx_language

    lang_code = mlx_language(language)

    # Build the text: prepend tone instruction if provided
    # one-voice uses in-band tone directions spoken by the model
    has_instruct = instruct is not None
    narration_text = text  # save original narration for trim target
    if has_instruct:
        # The instruct is a full sentence like "Speak in a warm, friendly way."
        # Prepend it so the model reads it as a direction
        text = f"{instruct} {text}"

    output_path = Path(output_wav)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"   🎙️  one-voice ({voice_name} [{lang_code}]): {len(text)} chars → {os.path.basename(output_wav)}")

    if has_instruct:
        # Generate to a temp file, then trim the leaked tone prompt
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            raw_path = Path(tmp.name)
        try:
            _synthesize(
                text=text,
                model="mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
                output=raw_path,
                ref_audio=ref_audio,
                ref_text=None,
                lang_code=lang_code,
            )
            # Trim using the narration text (without instruct) as the target
            _trim_tone_leak(raw_path, narration_text, output_path)
        finally:
            if raw_path.exists():
                raw_path.unlink()
    else:
        # No tone instruct — generate directly
        _synthesize(
            text=text,
            model="mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
            output=output_path,
            ref_audio=ref_audio,
            ref_text=None,
            lang_code=lang_code,
        )

    if not output_path.exists():
        raise RuntimeError(f"one-voice failed to produce audio: {output_wav}")

    return output_wav