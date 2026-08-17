"""Local Voicebox API client for narration audio generation."""

from __future__ import annotations

import base64
import json
import os
import shutil
import time
from typing import Any, List, Optional

import requests  # type: ignore[import-untyped]

from paths import DEFAULT_VOICEBOX_URL

_PERSONALITY_FALLBACK_WARNED = False
_PERSONALITY_LLM_WARMED_API_BASE: Optional[str] = None

# TTS engines Voicebox accepts for the /generate `engine` field. This is the
# *speech synthesizer* (the profile's "Default Engine"), NOT the refinement
# LLM used for personality text rewriting.
SUPPORTED_ENGINES: tuple[str, ...] = (
    "qwen",              # Qwen3-TTS
    "qwen_custom_voice", # Qwen CustomVoice (presets + instruct)
    "luxtts",            # LuxTTS
    "chatterbox",        # Chatterbox Multilingual
    "chatterbox_turbo",  # Chatterbox Turbo
    "tada",              # HumeAI TADA
    "kokoro",            # Kokoro
)
_SUPPORTED_ENGINE_SET = frozenset(SUPPORTED_ENGINES)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def personality_enabled_from_env(default: bool = False) -> bool:
    return _env_bool("VOICEBOX_PERSONALITY", default)


def get_voicebox_config(
    profile_id_override: Optional[str] = None,
    engine_override: Optional[str] = None,
) -> tuple[str, str, Optional[str]]:
    """Return (api_base_url, profile_id, engine) from env / CLI.

    ``engine`` is ``None`` when neither the CLI nor ``VOICEBOX_ENGINE`` is
    set, in which case Voicebox uses the profile's own "Default Engine".
    """
    api_base = (
        os.environ.get("VOICEBOX_API_URL")
        or os.environ.get("VOICEBOX_URL")
        or DEFAULT_VOICEBOX_URL
    ).rstrip("/")
    profile_id = (profile_id_override or os.environ.get("VOICEBOX_PROFILE_ID") or "").strip()
    if not profile_id:
        raise RuntimeError(
            "VOICEBOX_PROFILE_ID is not set. Add it to synth/.env or pass --profile-id. "
            "List profiles with: curl http://127.0.0.1:17493/profiles"
        )
    engine = (engine_override or os.environ.get("VOICEBOX_ENGINE") or "").strip() or None
    return api_base, profile_id, engine


def _voicebox_refinement_llm_model_size(api_base: str) -> str:
    try:
        resp = requests.get(f"{api_base.rstrip('/')}/settings/captures", timeout=10)
        resp.raise_for_status()
        return (resp.json().get("llm_model") or "0.6B").upper()
    except (requests.RequestException, ValueError, AttributeError):
        return "0.6B"


def _voicebox_refinement_llm_model(api_base: str) -> str:
    return f"qwen3-{_voicebox_refinement_llm_model_size(api_base).lower()}"


def _voicebox_personality_llm_status(api_base: str) -> tuple[bool, bool, str]:
    model_name = _voicebox_refinement_llm_model(api_base)
    try:
        resp = requests.get(f"{api_base.rstrip('/')}/models/status", timeout=10)
        resp.raise_for_status()
        for model in resp.json().get("models", []):
            if model.get("model_name") == model_name:
                return (
                    bool(model.get("downloaded")),
                    bool(model.get("loaded")),
                    model_name,
                )
    except (requests.RequestException, ValueError, AttributeError):
        pass
    return False, False, model_name


def _voicebox_personality_llm_ready(api_base: str) -> tuple[bool, str]:
    downloaded, _, model_name = _voicebox_personality_llm_status(api_base)
    return downloaded, model_name


def _profile_has_personality(api_base: str, profile_id: str) -> bool:
    try:
        resp = requests.get(f"{api_base.rstrip('/')}/profiles/{profile_id}", timeout=10)
        resp.raise_for_status()
        personality = resp.json().get("personality")
        return bool(isinstance(personality, str) and personality.strip())
    except (requests.RequestException, ValueError, AttributeError):
        return False


def _warmup_personality_llm(api_base: str) -> None:
    global _PERSONALITY_LLM_WARMED_API_BASE
    normalized_api_base = api_base.rstrip("/")
    if _PERSONALITY_LLM_WARMED_API_BASE == normalized_api_base:
        return

    downloaded, loaded, model_name = _voicebox_personality_llm_status(api_base)
    if not downloaded:
        raise RuntimeError(
            f"Voicebox personality LLM '{model_name}' is not downloaded. "
            "Download it in Voicebox → Settings → Captures → Refinement model."
        )
    if loaded:
        _PERSONALITY_LLM_WARMED_API_BASE = normalized_api_base
        return

    llm_size = _voicebox_refinement_llm_model_size(api_base)
    print(f"   … Loading Voicebox personality LLM ({model_name})...")
    resp = requests.post(
        f"{normalized_api_base}/llm/generate",
        json={"prompt": "Hi", "max_tokens": 1, "model_size": llm_size},
        timeout=180,
    )
    if not resp.ok:
        detail = resp.text[:500]
        raise RuntimeError(
            f"Voicebox personality LLM warmup failed ({resp.status_code}): {detail}"
        )
    _PERSONALITY_LLM_WARMED_API_BASE = normalized_api_base


def _warn_personality_fallback(llm_model: str) -> None:
    global _PERSONALITY_FALLBACK_WARNED
    if _PERSONALITY_FALLBACK_WARNED:
        return
    _PERSONALITY_FALLBACK_WARNED = True
    print("   ⚠️  Voicebox personality LLM is not downloaded; using plain TTS for now.")
    print(
        f"      To enable personality rewrite, download '{llm_model}' in Voicebox "
        "→ Settings → Captures → Refinement model."
    )


def _parse_sse_data_lines(body: str) -> List[dict[str, Any]]:
    events: List[dict[str, Any]] = []
    for line in body.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload:
            continue
        try:
            parsed = json.loads(payload)
            if isinstance(parsed, dict):
                events.append(parsed)
        except json.JSONDecodeError:
            continue
    return events


def _wait_for_voicebox_generation(
    api_base: str,
    generation_id: str,
    timeout: float = 600,
    poll_interval: float = 1.5,
) -> dict[str, Any]:
    status_url = f"{api_base.rstrip('/')}/generate/{generation_id}/status"
    deadline = time.monotonic() + timeout
    last_status = ""

    while time.monotonic() < deadline:
        resp = requests.get(status_url, timeout=60)
        resp.raise_for_status()
        events = _parse_sse_data_lines(resp.text)
        if not events:
            try:
                payload = resp.json()
                if isinstance(payload, dict):
                    events = [payload]
            except ValueError:
                pass

        for event in reversed(events):
            status = (event.get("status") or "").lower()
            if status in ("failed", "error"):
                raise RuntimeError(f"Voicebox generation failed: {event.get('error') or event}")
            if status == "completed" or (event.get("duration") or 0) > 0:
                return event
            last_status = status or last_status

        if last_status:
            print(f"      … Voicebox status: {last_status}")
        time.sleep(poll_interval)

    raise RuntimeError(
        f"Voicebox generation timed out after {timeout:.0f}s "
        f"(last status: {last_status or 'unknown'})"
    )


def _save_voicebox_audio_from_response(
    data: dict[str, Any],
    output_wav: str,
    api_base: str,
    timeout: float,
) -> str:
    for key in ("audio_base64", "audio", "wav_base64", "data"):
        encoded = data.get(key)
        if isinstance(encoded, str) and encoded.strip():
            raw = base64.b64decode(encoded)
            with open(output_wav, "wb") as wav_file:
                wav_file.write(raw)
            return output_wav

    audio_path = data.get("audio_path")
    if isinstance(audio_path, str) and audio_path and os.path.isfile(audio_path):
        shutil.copy2(audio_path, output_wav)
        return output_wav

    generation_id = data.get("id")
    if generation_id:
        _wait_for_voicebox_generation(api_base, generation_id, timeout=timeout)
        audio_url = f"{api_base.rstrip('/')}/audio/{generation_id}"
        audio_resp = requests.get(audio_url, timeout=timeout)
        audio_resp.raise_for_status()
        with open(output_wav, "wb") as wav_file:
            wav_file.write(audio_resp.content)
        return output_wav

    raise RuntimeError(f"Voicebox response did not include audio: {data}")


def generate_voicebox_audio(
    text: str,
    profile_id: str,
    output_wav: str,
    api_base: str = DEFAULT_VOICEBOX_URL,
    timeout: float = 600,
    personality: bool = False,
    engine: Optional[str] = None,
) -> str:
    """POST narration text to Voicebox /generate and save a .wav file.

    ``engine`` optionally overrides the profile's "Default Engine" (the TTS
    synthesizer). When ``None``, Voicebox uses the profile's default engine.
    """
    if not text or not text.strip():
        raise ValueError("Cannot generate audio from empty text")

    url = f"{api_base.rstrip('/')}/generate"
    payload: dict[str, Any] = {
        "text": text,
        "profile_id": profile_id,
        "language": "en",
    }

    if engine:
        if engine not in _SUPPORTED_ENGINE_SET:
            raise ValueError(
                f"Unknown Voicebox engine {engine!r}. Supported engines: "
                f"{', '.join(SUPPORTED_ENGINES)}."
            )
        payload["engine"] = engine

    use_personality = personality
    if use_personality:
        if not _profile_has_personality(api_base, profile_id):
            print(
                "   ⚠️  Voicebox profile has no personality prompt; using plain TTS."
            )
            use_personality = False
        else:
            downloaded, llm_model = _voicebox_personality_llm_ready(api_base)
            if not downloaded:
                _warn_personality_fallback(llm_model)
                use_personality = False
            else:
                _warmup_personality_llm(api_base)
                payload["personality"] = True

    mode = "personality" if use_personality else "plain TTS"
    print(f"   🎙️  Voicebox ({mode}): {len(text)} chars → {os.path.basename(output_wav)}")
    resp = requests.post(url, json=payload, timeout=timeout)
    if use_personality and resp.status_code >= 500:
        print("   ⚠️  Voicebox personality rewrite failed; retrying after LLM warmup...")
        _warmup_personality_llm(api_base)
        resp = requests.post(url, json=payload, timeout=timeout)
    if use_personality and resp.status_code >= 500:
        detail = resp.text[:500]
        raise RuntimeError(
            "Voicebox personality rewrite failed. "
            "Open the Voicebox app and try 'Speak in character' once to load the "
            f"refinement model, then retry. Server said: {detail}"
        )
    if not resp.ok:
        detail = resp.text[:500]
        raise RuntimeError(f"Voicebox /generate failed ({resp.status_code}): {detail}")

    data: dict[str, Any] = resp.json()
    return _save_voicebox_audio_from_response(data, output_wav, api_base, timeout)
