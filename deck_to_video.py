#!/usr/bin/env python3
"""
Export a deck (Google Slides or local PPTX) to PNG + speaker notes, generate
voiceover via local Voicebox, and assemble MP4 video(s) with FFmpeg.

Requires Voicebox running locally only when using --gen-voiceover
(default API: http://127.0.0.1:17493).

SETUP:
======
1. Python dependencies:
       cd synth
       pip install -r requirements.txt

2. Google Slides only — API credentials in synth/:
   - credentials.json and token.json

3. PPTX only — LibreOffice for slide rendering:
   - Install LibreOffice and ensure `soffice` is on PATH

4. Voicebox:
   - Start the Voicebox desktop app (API on port 17493)
   - Add to synth/.env:
       VOICEBOX_PROFILE_ID=your-profile-uuid

USAGE:
======
    cd synth
    python deck_to_video.py <SLIDES_ID_OR_URL>
    python deck_to_video.py deck.pptx --gen-voiceover
    python deck_to_video.py deck.pptx   # reuses existing slide_XX_voiceover.wav
    python deck_to_video.py <id> --only-slide 3
    python deck_to_video.py <id> --split-at 10,20
    python deck_to_video.py <id> -o my_deck.mp4
    python deck_to_video.py <id> --export-only
    python deck_to_video.py <id> --ken-burns     # add Ken Burns zoom/pan
    python deck_to_video.py deck.pptx --gen-voiceover --language zh
                                                 # force en/zh/ja/ko/de/fr/ru/pt/es/it
                                                 # (default: auto-detect per slide)

OUTPUT:
=======
synth/out/<sanitized_title>/
    slide_01.png, slide_02.png, ...
    slide_01_notes.txt, ...
    slide_01_voiceover.wav, ...
    <sanitized_title>.mp4  (unless --export-only or --split-at)
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import wave
from typing import List, Optional, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)


def _read_version() -> str:
    """Read the version string from the VERSION file next to this script."""
    version_path = os.path.join(_SCRIPT_DIR, "VERSION")
    try:
        with open(version_path, "r", encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return "unknown"


__version__ = _read_version()

try:
    import dotenv  # type: ignore[import-untyped]
    import requests  # type: ignore[import-untyped]

    from bgm import resolve_bgm_path
    from google_slides import (
        check_document_type,
        export_slides_to_png,
        export_speaker_notes,
        extract_presentation_id,
        get_presentation_title,
        get_skipped_slide_indices,
    )
    from narration import (
        _first_tone_instruct,
        _parse_blocks,
        _tone_instruct,
        SUPPORTED_LANGUAGE_CODES,
        canonical_language,
        parse_bgm_cues,
        prepare_narration,
        split_notes_on_sfx,
        split_notes_on_tone,
    )
    from paths import (
        DEFAULT_BG_MUSIC_VOLUME,
        DEFAULT_FPS,
        DEFAULT_INTER_SLIDE_PAUSE_SECONDS,
        DEFAULT_KEN_BURNS_ZOOM,
        DEFAULT_SILENT_SLIDE_SECONDS,
        DEFAULT_TRANSITION_SECONDS,
        DEFAULT_TRANSITION_STYLE,
        DEFAULT_VOICEOVER_TRAIL_SILENCE_SECONDS,
        ENV_PATH,
        OUT_BASE_DIR,
    )
    from pptx_source import (
        export_pptx_slides_to_png,
        export_pptx_speaker_notes,
        get_pptx_title,
        is_pptx_path,
    )
    from split_ranges import compute_slide_ranges, parse_split_at, video_label_for_range
    from sfx import SFX_SAMPLES, assemble_voiceover
    from video_assembly import TRANSITION_STYLES, assemble_presentation_video

    # Narration engine: one-voice (default) or Voicebox (--voicebox flag)
    # one-voice is imported lazily; voicebox_client is kept for --voicebox mode
    from voicebox_client import (
        SUPPORTED_ENGINES,
        generate_voicebox_audio,
        get_voicebox_config,
        personality_enabled_from_env,
    )
    try:
        from one_voice_adapter import generate_one_voice_audio, get_one_voice_config
        _one_voice_available = True
    except ImportError:
        _one_voice_available = False
except ImportError as exc:
    sys.stderr.write(
        f"\n❌ Missing Python dependency ({exc}).\n"
        "\n"
        "This usually means 'python' is NOT the project virtualenv.\n"
        "Run one of:\n"
        "\n"
        "    source .venv/bin/activate\n"
        "    python deck_to_video.py ...\n"
        "\n"
        "or invoke the venv interpreter directly:\n"
        "\n"
        "    .venv/bin/python deck_to_video.py ...\n"
        "\n"
        "If dependencies are genuinely missing from the active environment:\n"
        "\n"
        "    pip install -r requirements.txt\n"
        "\n"
    )
    sys.exit(1)

dotenv.load_dotenv(ENV_PATH)


def _sorted_slide_assets(output_dir: str, suffix: str) -> List[str]:
    paths = []
    for name in os.listdir(output_dir):
        match = re.match(rf"^slide_(\d{{2}}){re.escape(suffix)}$", name)
        if match:
            paths.append((int(match.group(1)), os.path.join(output_dir, name)))
    paths.sort(key=lambda item: item[0])
    return [path for _, path in paths]


def _slide_number_from_name(path: str, suffix: str) -> Optional[int]:
    """Extract the 1-based slide number embedded in an exported asset name."""
    match = re.match(rf"^slide_(\d{{2}}){re.escape(suffix)}$", os.path.basename(path))
    return int(match.group(1)) if match else None


def _sanitize_title_for_path(title: str) -> str:
    sanitized = title.replace(" ", "_").replace("/", "_").replace("\\", "_")
    return "".join(char for char in sanitized if char.isalnum() or char in ("_", "-", "."))


def _voiceover_wav_path(output_dir: str, slide_idx: int) -> str:
    return os.path.join(output_dir, f"slide_{slide_idx:02d}_voiceover.wav")


def _append_silence_to_wav(wav_path: str, duration: float = 3.0) -> None:
    """Append silence to a WAV file to prevent the last word from cutting off."""
    with wave.open(wav_path, "rb") as wf:
        params = wf.getparams()
        frames = wf.readframes(wf.getnframes())

    nchannels, sampwidth, framerate = params[:3]
    silence_frame = b"\x00" * sampwidth * nchannels
    silence_frames = silence_frame * int(framerate * duration)

    with wave.open(wav_path, "wb") as wf:
        wf.setparams(params)
        wf.writeframes(frames + silence_frames)


def _missing_voiceover_slide_numbers(
    output_dir: str,
    notes_per_slide: List[str],
    *,
    personality: bool,
    slide_numbers: Optional[List[int]] = None,
) -> List[int]:
    missing: List[int] = []
    for idx, notes_text in enumerate(notes_per_slide, start=1):
        # slide_numbers lets the caller keep real slide filenames (e.g.
        # slide_04_voiceover.wav for --only-slide 4) even when the asset
        # lists contain a subset of the deck.
        slide_idx = slide_numbers[idx - 1] if slide_numbers else idx
        if not prepare_narration(notes_text, personality=personality):
            continue
        if not os.path.isfile(_voiceover_wav_path(output_dir, slide_idx)):
            missing.append(slide_idx)
    return missing


def _ensure_terminal_punctuation(text: str, language: Optional[str] = None) -> str:
    """End narration with a sentence terminator so TTS closes its prosody.

    CJK languages (Mandarin, Japanese) get the ideographic full stop ``。``;
    every other language gets ``.``. Text already ending in a terminator is
    left alone. *language* is the resolved deck-level code; when omitted it is
    auto-detected from *text*.
    """
    from narration import detect_language

    if not text:
        return text
    if text[-1] in (".", "!", "?", "。", "！", "？"):
        return text
    code = language or detect_language(text)
    terminator = "。" if code in ("zh", "ja") else "."
    return text.rstrip() + terminator


def _resolve_take_language(source: str, language_override: str) -> str:
    """Resolve the per-take language code from the deck-level override."""
    from narration import resolve_narration_language

    return resolve_narration_language(source, override=language_override)


def _cli_language(value: str) -> str:
    """argparse ``type`` for ``--language`` — accepts codes or language names.

    Rejects unknown values outright (unlike the env-var path, which degrades
    to ``"auto"``), so a typo on the command line fails fast.
    """
    canonical = canonical_language(value)
    if canonical is None:
        raise argparse.ArgumentTypeError(
            f"unknown language {value!r}. Use 'auto', a code "
            f"({', '.join(SUPPORTED_LANGUAGE_CODES)}), or a language name "
            "such as 'chinese' or 'pt-BR'."
        )
    return canonical


def _generate_voiceover_take(
    source: str,
    output_wav: str,
    api_base: str,
    profile_id: str,
    personality: bool,
    engine: Optional[str],
    use_voicebox: bool = False,
    language_override: str = "auto",
) -> bool:
    """Synthesize one narration take. Uses Voicebox or one-voice based on use_voicebox.

    ``language_override`` is the deck-level ``--language`` value (any
    canonical code, or ``"auto"``); ``"auto"`` auto-detects per take.
    """
    from narration import voicebox_language

    narration_text = prepare_narration(source, personality=personality)
    if not narration_text.strip():
        return False
    instruct = _first_tone_instruct(source)
    language = _resolve_take_language(narration_text, language_override)
    text = _ensure_terminal_punctuation(narration_text, language=language)
    voicebox_lang = voicebox_language(language)
    print(f"      🌐 language: {language} (voicebox={voicebox_lang})")

    if use_voicebox:
        generate_voicebox_audio(
            text,
            profile_id=profile_id,
            output_wav=output_wav,
            api_base=api_base,
            personality=personality,
            engine=engine,
            instruct=instruct,
            language=voicebox_lang,
        )
    else:
        generate_one_voice_audio(
            text,
            profile_id=profile_id,
            output_wav=output_wav,
            instruct=instruct,
            language=language,
        )
    return True


def _generate_voiceover_take_with_voice(
    source: str,
    output_wav: str,
    api_base: str,
    voice_name: str,
    personality: bool,
    engine: Optional[str],
    use_voicebox: bool = False,
    language_override: str = "auto",
) -> bool:
    """Synthesize one narration take with a specific voice.

    Like _generate_voiceover_take() but accepts an explicit voice name
    instead of using the default profile_id.
    """
    from narration import voicebox_language

    narration_text = prepare_narration(source, personality=personality)
    if not narration_text.strip():
        return False
    instruct = _first_tone_instruct(source)
    language = _resolve_take_language(narration_text, language_override)
    text = _ensure_terminal_punctuation(narration_text, language=language)
    voicebox_lang = voicebox_language(language)
    print(f"      🌐 language: {language} (voicebox={voicebox_lang})")

    if use_voicebox:
        generate_voicebox_audio(
            text,
            profile_id=voice_name,
            output_wav=output_wav,
            api_base=api_base,
            personality=personality,
            engine=engine,
            instruct=instruct,
            language=voicebox_lang,
        )
    else:
        generate_one_voice_audio(
            text,
            profile_id=voice_name,
            output_wav=output_wav,
            instruct=instruct,
            language=language,
        )
    return True


def _generate_voiceover_for_slide(
    output_dir: str,
    slide_idx: int,
    notes_text: str,
    api_base: str,
    profile_id: str,
    personality: bool,
    engine: Optional[str] = None,
    use_voicebox: bool = False,
    language_override: str = "auto",
) -> Optional[str]:
    if not prepare_narration(notes_text, personality=personality):
        print(
            f"   ⏭️  Slide {slide_idx}: no narration "
            f"(will use {DEFAULT_SILENT_SLIDE_SECONDS}s silent)"
        )
        return None

    wav_path = _voiceover_wav_path(output_dir, slide_idx)
    parts = split_notes_on_sfx(notes_text, supported=set(SFX_SAMPLES))

    if not any(part["kind"] in ("sfx", "pause") for part in parts):
        # Single-take path (no sound-effect cues): check for multiple voices/tones
        blocks = _parse_blocks(notes_text)
        if len(blocks) <= 1:
            # Zero or one block — single take. Honor an explicit [voice: ...]
            # tag even when it is the only block; otherwise a single-tag
            # slide (e.g. "[voice: yuki] ...") would silently fall back to
            # the default profile voice.
            block_voice = blocks[0].get("voice") if blocks else None
            voice_to_use = block_voice or profile_id
            if not _generate_voiceover_take_with_voice(
                notes_text, wav_path, api_base, voice_to_use, personality, engine, use_voicebox,
                language_override,
            ):
                print(f"   ⏭️  Slide {slide_idx}: no narration after prep")
                return None
        else:
            # Multiple blocks — generate separate takes with per-block voice/tone
            voice_name = profile_id  # default voice
            print(f"   🎭 Slide {slide_idx}: {len(blocks)} narration blocks detected")
            pieces: List[Tuple[str, str]] = []
            with tempfile.TemporaryDirectory(prefix=f"slide_{slide_idx:02d}_voice_") as tmp_dir:
                for i, block in enumerate(blocks, start=1):
                    # Use block's voice if specified, otherwise stick with previous
                    if block.get("voice"):
                        voice_name = block["voice"]
                    block_text = block["text"]
                    if not block_text.strip():
                        continue
                    take_path = os.path.join(tmp_dir, f"block_{i:02d}.wav")
                    print(f"      🎙️ Block {i}: voice={voice_name}, text={block_text[:50]!r}...")
                    if _generate_voiceover_take_with_voice(
                        block_text,
                        take_path,
                        api_base,
                        voice_name,
                        personality,
                        engine,
                        use_voicebox,
                        language_override,
                    ):
                        pieces.append(("wav", take_path))
                        print(f"      ✅ Block {i}: generated {take_path}")
                    else:
                        print(f"      ❌ Block {i}: generation failed")
                if pieces:
                    assemble_voiceover(pieces, wav_path)
                    print(f"   🔗 Slide {slide_idx}: assembled {len(pieces)} clips")
                else:
                    print(f"   ❌ Slide {slide_idx}: no clips generated")
        # Append a short tail of silence so the final word is never truncated
        # and voiced slides flow into the next with a natural 1s gap.
        _append_silence_to_wav(
            wav_path, duration=DEFAULT_VOICEOVER_TRAIL_SILENCE_SECONDS
        )
        print(f"   ✅ Slide {slide_idx}: saved {wav_path}")
        return wav_path

    # Multi-take path: each recognized "[sfx: ...]" / "[pause ...]" tag splits
    # the notes into separate takes. Within each narration part, voice/tone
    # changes are also split into separate takes so multi-voice slides sound
    # right. Pause tags become exact silence stitched between takes.
    pieces: List[Tuple[str, str]] = []
    inserted: List[str] = []
    paused: List[float] = []
    take_no = 0
    voice_name = profile_id  # track current voice across blocks
    with tempfile.TemporaryDirectory(prefix=f"slide_{slide_idx:02d}_sfx_") as tmp_dir:
        for part in parts:
            if part["kind"] == "sfx":
                pieces.append(("sfx", part["name"]))
                continue
            if part["kind"] == "pause":
                pieces.append(("pause", part["seconds"]))
                paused.append(part["seconds"])
                continue
            # Split this narration part by voice/tone changes
            blocks = _parse_blocks(part["source"])
            for block in blocks:
                if block.get("voice"):
                    voice_name = block["voice"]
                block_text = block["text"]
                if not block_text.strip():
                    continue
                take_no += 1
                take_path = os.path.join(tmp_dir, f"take_{take_no:02d}.wav")
                if _generate_voiceover_take_with_voice(
                    block_text,
                    take_path,
                    api_base,
                    voice_name,
                    personality,
                    engine,
                    use_voicebox,
                    language_override,
                ):
                    pieces.append(("wav", take_path))
        inserted = assemble_voiceover(pieces, wav_path)
    for name in inserted:
        print(f"   🥁 Slide {slide_idx}: {name} mid-slide")
    for seconds in paused:
        print(f"   ⏸️  Slide {slide_idx}: {seconds:g}s pause stitched in")
    print(f"   ✅ Slide {slide_idx}: saved {wav_path}")
    return wav_path


def _voiceover_paths_for_slides(
    output_dir: str,
    notes_per_slide: List[str],
    *,
    gen_voiceover: bool,
    generate_missing: bool,
    personality: bool,
    api_base: Optional[str] = None,
    profile_id: Optional[str] = None,
    engine: Optional[str] = None,
    slide_numbers: Optional[List[int]] = None,
    use_voicebox: bool = False,
    language_override: str = "auto",
) -> List[Optional[str]]:
    if gen_voiceover:
        if not profile_id:
            raise RuntimeError(
                "Voice configuration is required when using --gen-voiceover"
            )
        engine_name = "Voicebox" if use_voicebox else "one-voice"
        print(f"\n🎙️  Generating voiceovers via {engine_name}...")
    elif generate_missing:
        missing = _missing_voiceover_slide_numbers(
            output_dir, notes_per_slide, personality=personality
        )
        if missing:
            if not profile_id:
                slides = ", ".join(str(idx) for idx in missing)
                raise RuntimeError(
                    f"Missing voiceover WAV(s) for slide(s) {slides}. "
                    "Set VOICEBOX_PROFILE_ID in .env (or pass --profile-id), "
                    "or pass --gen-voiceover to generate them."
                )
            engine_name = "Voicebox" if use_voicebox else "one-voice"
            print(
                f"\n🎙️  Generating missing voiceovers for slide(s) "
                f"{', '.join(str(idx) for idx in missing)} via {engine_name}..."
            )
        else:
            print("\n🎙️  Using existing voiceover files (pass --gen-voiceover to regenerate)...")
    else:
        print("\n🎙️  Using existing voiceover files (pass --gen-voiceover to regenerate)...")

    if not gen_voiceover and language_override != "auto":
        print(
            f"   ⚠️  --language {language_override} only affects audio generated "
            "now; existing WAVs are reused as-is (pass --gen-voiceover to "
            "re-record them in this language)."
        )

    wav_paths: List[Optional[str]] = []
    for idx, notes_text in enumerate(notes_per_slide, start=1):
        slide_idx = slide_numbers[idx - 1] if slide_numbers else idx
        wav_path = _voiceover_wav_path(output_dir, slide_idx)
        narration = prepare_narration(notes_text, personality=personality)

        if gen_voiceover:
            wav_paths.append(
                _generate_voiceover_for_slide(
                    output_dir,
                    slide_idx,
                    notes_text,
                    api_base,
                    profile_id,
                    personality,
                    engine=engine,
                    use_voicebox=use_voicebox,
                    language_override=language_override,
                )
            )
            continue

        if os.path.isfile(wav_path):
            wav_paths.append(wav_path)
            print(f"   ✅ Slide {slide_idx}: using {wav_path}")
            continue

        if narration and generate_missing:
            wav_paths.append(
                _generate_voiceover_for_slide(
                    output_dir,
                    slide_idx,
                    notes_text,
                    api_base,
                    profile_id,
                    personality,
                    engine=engine,
                    use_voicebox=use_voicebox,
                    language_override=language_override,
                )
            )
            continue

        if narration:
            raise RuntimeError(
                f"Missing voiceover WAV for slide {slide_idx}. "
                "Pass --gen-voiceover to generate it."
            )

        wav_paths.append(None)

    return wav_paths


def _background_music_for_deck(
    notes_per_slide: List[str],
    override: Optional[str] = None,
    enabled: bool = True,
) -> Tuple[Optional[str], float]:
    """Resolve the deck-level background-music track and its volume.

    ``[bgm: ...]`` tags are presentation-scoped rather than per-slide: the
    FIRST tag found across all speaker notes selects the one continuous track
    layered under the entire assembled timeline. An explicit *override* (the
    --bg-music flag) supersedes tags; *enabled=False* (--no-bg-music) turns
    music off entirely even when tags are present.
    """
    if not enabled:
        return None, DEFAULT_BG_MUSIC_VOLUME

    if override:
        return resolve_bgm_path(override), DEFAULT_BG_MUSIC_VOLUME

    for notes in notes_per_slide:
        cues = parse_bgm_cues(notes)
        if cues:
            ref, volume = cues[0]
            resolved_volume = DEFAULT_BG_MUSIC_VOLUME if volume is None else volume
            return resolve_bgm_path(ref), resolved_volume

    return None, DEFAULT_BG_MUSIC_VOLUME


def _render_videos(
    png_paths: List[str],
    wav_paths: List[Optional[str]],
    output_dir: str,
    sanitized_title: str,
    split_points: Optional[List[int]],
    only_slide: Optional[int],
    output_mp4: Optional[str],
    fps: int,
    inter_slide_pause_seconds: float,
    ken_burns_zoom: float = 0.0,
    background_music_path: Optional[str] = None,
    bg_music_volume: float = DEFAULT_BG_MUSIC_VOLUME,
    transition: str = DEFAULT_TRANSITION_STYLE,
    transition_seconds: float = DEFAULT_TRANSITION_SECONDS,
) -> List[str]:
    ranges = compute_slide_ranges(len(png_paths), split_points)
    created: List[str] = []

    for start, end in ranges:
        segment_pngs = png_paths[start:end]
        segment_wavs = wav_paths[start:end]
        label = video_label_for_range(
            sanitized_title,
            start,
            end,
            len(png_paths),
            only_slide=only_slide,
        )
        if output_mp4 and len(ranges) == 1:
            mp4_path = output_mp4
        else:
            mp4_path = os.path.join(output_dir, f"{label}.mp4")

        assemble_presentation_video(
            segment_pngs,
            segment_wavs,
            mp4_path,
            fps=fps,
            inter_slide_pause_seconds=inter_slide_pause_seconds,
            ken_burns_zoom=ken_burns_zoom,
            background_music_path=background_music_path,
            bg_music_volume=bg_music_volume,
            transition=transition,
            transition_seconds=transition_seconds,
        )
        created.append(os.path.abspath(mp4_path))
    return created


def process_google_slides(
    presentation_id: str,
    out_dir: str,
    only_slide: Optional[int],
) -> None:
    is_slides, doc_type = check_document_type(presentation_id)
    if not is_slides:
        raise ValueError(f"Document is not a Google Slides presentation (type: {doc_type})")

    print("\n🔍 Checking for skipped slides...")
    skipped_indices = get_skipped_slide_indices(presentation_id)

    if only_slide is not None:
        if only_slide < 1:
            raise ValueError("--only-slide must be >= 1")
        if only_slide - 1 in skipped_indices:
            print(f"⏭️  Note: --only-slide {only_slide} is marked as skipped in Google Slides")

    print("\n📝 Exporting speaker notes...")
    notes_per_slide = export_speaker_notes(
        presentation_id,
        out_dir,
        skipped_indices=skipped_indices,
        only_slide=only_slide,
    )

    print("\n📥 Exporting slides to PNG...")
    export_slides_to_png(
        presentation_id,
        out_dir,
        skipped_indices=skipped_indices,
        only_slide=only_slide,
    )


def process_pptx(
    pptx_path: str,
    out_dir: str,
    only_slide: Optional[int],
) -> None:
    pptx_path = os.path.abspath(pptx_path)

    if only_slide is not None and only_slide < 1:
        raise ValueError("--only-slide must be >= 1")

    print("\n📝 Exporting speaker notes...")
    export_pptx_speaker_notes(pptx_path, out_dir, only_slide=only_slide)

    print("\n📥 Exporting slides to PNG...")
    export_pptx_slides_to_png(pptx_path, out_dir, only_slide=only_slide)


def main(
    source: str,
    *,
    only_slide: Optional[int] = None,
    profile_id: Optional[str] = None,
    output_mp4: Optional[str] = None,
    export_only: bool = False,
    gen_voiceover: bool = False,
    personality: Optional[bool] = None,
    fps: int = DEFAULT_FPS,
    voicebox_url: Optional[str] = None,
    use_voicebox: bool = False,
    inter_slide_pause_seconds: float = DEFAULT_INTER_SLIDE_PAUSE_SECONDS,
    split_at: Optional[str] = None,
    ken_burns_zoom: float = DEFAULT_KEN_BURNS_ZOOM,
    engine: Optional[str] = None,
    bg_music: Optional[str] = None,
    no_bg_music: bool = False,
    transition: str = DEFAULT_TRANSITION_STYLE,
    transition_seconds: float = DEFAULT_TRANSITION_SECONDS,
    language: Optional[str] = None,
) -> int:
    from narration import parse_language_override

    language_override = parse_language_override(
        language or os.environ.get("DECK_LANGUAGE") or os.environ.get("ONE_VOICE_LANG")
    )
    if language_override != "auto":
        print(f"🌐 Deck language override: {language_override} (all slides)")
    try:
        use_personality = (
            personality_enabled_from_env()
            if personality is None
            else personality
        )
        api_base: Optional[str] = None
        voicebox_profile_id: Optional[str] = None
        voicebox_engine: Optional[str] = None

        # Resolve narration engine: one-voice (default) or Voicebox (--voicebox)
        if use_voicebox:
            # Legacy: Voicebox API — resolve full config
            api_base, voicebox_profile_id, voicebox_engine = get_voicebox_config(
                profile_id_override=profile_id,
                engine_override=engine,
            )
        else:
            # Default: one-voice local TTS
            if not _one_voice_available:
                print(
                    "\n⚠️  one-voice not installed. Falling back to Voicebox.\n"
                    "Install one-voice for local TTS:\n"
                    "  pip install one-voice@git+https://github.com/simagix/one-voice.git\n"
                )
                use_voicebox = True
                api_base, voicebox_profile_id, voicebox_engine = get_voicebox_config(
                    profile_id_override=profile_id,
                    engine_override=engine,
                )
            else:
                voice_name, _ = get_one_voice_config(
                    profile_id_override=profile_id,
                )
                voicebox_profile_id = voice_name
                voicebox_engine = None
                api_base = ""

        split_points: Optional[List[int]] = None
        if split_at:
            split_points = parse_split_at(split_at)

        if is_pptx_path(source):
            deck_title = get_pptx_title(os.path.abspath(source))
        else:
            presentation_id = extract_presentation_id(source)
            deck_title = get_presentation_title(presentation_id)

        sanitized_title = _sanitize_title_for_path(deck_title) or "deck_export"
        out_dir = os.path.join(OUT_BASE_DIR, sanitized_title)
        os.makedirs(out_dir, exist_ok=True)

        deck_title_file = os.path.join(out_dir, "deck_title.txt")
        with open(deck_title_file, "w", encoding="utf-8") as title_file:
            title_file.write(deck_title)

        print(f"📄 Deck: {deck_title}")

        if is_pptx_path(source):
            process_pptx(source, out_dir, only_slide=only_slide)
        else:
            presentation_id = extract_presentation_id(source)
            process_google_slides(presentation_id, out_dir, only_slide=only_slide)

        note_paths = _sorted_slide_assets(out_dir, "_notes.txt")
        png_paths = _sorted_slide_assets(out_dir, ".png")
        if not png_paths:
            raise RuntimeError("No slide PNGs were exported")

        # --only-slide: the exported files keep the requested slide's real
        # number (slide_04.png / slide_04_notes.txt), so stale assets from
        # earlier runs under other numbers are ignored here and later reuse
        # checks (voiceover lookup included) match the correct files.
        if only_slide is not None:
            note_paths = [
                p
                for p in note_paths
                if _slide_number_from_name(p, "_notes.txt") == only_slide
            ]
            png_paths = [
                p for p in png_paths if _slide_number_from_name(p, ".png") == only_slide
            ]
            if not png_paths:
                raise RuntimeError(
                    f"--only-slide {only_slide} produced no PNG export "
                    "(is the slide number in range?)"
                )

        # Real slide numbers for each asset, so voiceover generation and
        # reuse target the correct slide_XX_voiceover.wav even when the
        # asset lists contain a subset of the deck.
        slide_numbers = [
            _slide_number_from_name(p, ".png") or idx + 1
            for idx, p in enumerate(png_paths)
        ]

        notes_per_slide = []
        for note_path in note_paths:
            with open(note_path, encoding="utf-8") as note_file:
                notes_per_slide.append(note_file.read())

        if export_only and not gen_voiceover:
            print(f"\n✅ Export complete (PNGs + notes) in {os.path.abspath(out_dir)}")
            return 0

        generate_missing = not export_only
        needs_voicebox = gen_voiceover or (
            generate_missing
            and bool(
                _missing_voiceover_slide_numbers(
                    out_dir,
                    notes_per_slide,
                    personality=use_personality,
                    slide_numbers=slide_numbers,
                )
            )
        )
        wav_paths = _voiceover_paths_for_slides(
            out_dir,
            notes_per_slide,
            gen_voiceover=gen_voiceover,
            generate_missing=generate_missing,
            personality=use_personality,
            api_base=api_base,
            profile_id=voicebox_profile_id,
            engine=voicebox_engine,
            slide_numbers=slide_numbers,
            use_voicebox=use_voicebox,
            language_override=language_override,
        )
        while len(wav_paths) < len(png_paths):
            wav_paths.append(None)

        if export_only:
            print(f"\n✅ Export complete (PNGs + WAVs) in {os.path.abspath(out_dir)}")
            return 0

        background_music_path, bg_music_volume = _background_music_for_deck(
            notes_per_slide,
            override=bg_music,
            enabled=not no_bg_music,
        )

        created = _render_videos(
            png_paths,
            wav_paths,
            out_dir,
            sanitized_title,
            split_points=split_points if only_slide is None else None,
            only_slide=only_slide,
            output_mp4=output_mp4,
            fps=fps,
            inter_slide_pause_seconds=inter_slide_pause_seconds,
            ken_burns_zoom=ken_burns_zoom,
            background_music_path=background_music_path,
            bg_music_volume=bg_music_volume,
            transition=transition,
            transition_seconds=transition_seconds,
        )
        if len(created) == 1:
            print(f"\n✅ Video saved: {created[0]}")
        else:
            print(f"\n✅ Created {len(created)} video(s):")
            for path in created:
                print(f"   - {path}")
        return 0

    except (RuntimeError, ValueError, requests.RequestException, OSError) as exc:
        print(f"\n❌ Error: {exc}")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export a deck, generate Voicebox narration, and assemble MP4 video(s).",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"deck_to_video v{__version__}",
        help="Show version and exit",
    )
    parser.add_argument(
        "source",
        help="Google Slides ID/URL or path to a local .pptx file",
    )
    parser.add_argument(
        "--profile-id",
        help="Voicebox profile UUID (overrides VOICEBOX_PROFILE_ID in .env)",
    )
    parser.add_argument(
        "--voicebox-url",
        default=None,
        help="Voicebox API base URL (default: http://127.0.0.1:17493)",
    )
    parser.add_argument(
        "--engine",
        choices=SUPPORTED_ENGINES,
        default=None,
        help=(
            "TTS engine to use, overriding the profile's Default Engine. "
            f"One of: {', '.join(SUPPORTED_ENGINES)} (default: the profile's "
            "Default Engine, or VOICEBOX_ENGINE in .env)"
        ),
    )
    parser.add_argument(
        "--voicebox",
        action="store_true",
        help=(
            "Use Voicebox API for narration instead of one-voice local TTS "
            "(default: one-voice)"
        ),
    )
    parser.add_argument(
        "--language",
        type=_cli_language,
        default=None,
        metavar="LANG",
        help=(
            "Narration language: 'auto' (default) detects each slide from its "
            "notes; any other value forces every slide. Accepts a code or a "
            "name — one of "
            f"{', '.join(SUPPORTED_LANGUAGE_CODES)} (e.g. 'zh', 'chinese', "
            "'pt-BR'). Also settable via DECK_LANGUAGE / ONE_VOICE_LANG."
        ),
    )
    parser.add_argument(
        "--only-slide",
        type=int,
        default=None,
        help="Process a single slide (1-based) for quick tests",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_mp4",
        help="Output MP4 path (default: synth/out/<title>/<title>.mp4)",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Export PNGs and notes only; skip FFmpeg video assembly",
    )
    parser.add_argument(
        "--gen-voiceover",
        action="store_true",
        help="Generate voiceover WAVs via Voicebox (default: reuse existing slide_XX_voiceover.wav)",
    )
    parser.add_argument(
        "--personality",
        dest="personality",
        action="store_true",
        help=(
            "Rewrite speaker notes in the profile's voice before TTS "
            "(default: off; also enabled by VOICEBOX_PERSONALITY=1 in .env)"
        ),
    )
    parser.set_defaults(personality=None)
    parser.add_argument(
        "--split-at",
        metavar="N[,N...]",
        help="Split into multiple MP4s at 1-indexed slide numbers (e.g. 10,20)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help=f"Video frames per second (default: {DEFAULT_FPS})",
    )
    parser.add_argument(
        "--inter-slide-pause",
        type=float,
        default=DEFAULT_INTER_SLIDE_PAUSE_SECONDS,
        metavar="SECONDS",
        help=(
            "Silent hold after slides without voiceover; voiced slides "
            "already carry a built-in trailing silence (default: "
            f"{DEFAULT_INTER_SLIDE_PAUSE_SECONDS}; 0 to disable)"
        ),
    )
    ken_burns_group = parser.add_mutually_exclusive_group()
    ken_burns_group.add_argument(
        "--static",
        action="store_true",
        help=(
            "[deprecated] Render slides as static images. This is now the "
            "default behaviour; use --ken-burns to enable the Ken Burns "
            f"zoom/pan (zoom x{DEFAULT_KEN_BURNS_ZOOM} over the slide "
            "duration)"
        ),
    )
    ken_burns_group.add_argument(
        "--ken-burns",
        action="store_true",
        help=(
            "Apply the Ken Burns zoom/pan effect to each slide "
            f"(zoom x{DEFAULT_KEN_BURNS_ZOOM} over the slide duration). "
            "By default slides are rendered as static images."
        ),
    )

    bg_music_group = parser.add_mutually_exclusive_group()
    bg_music_group.add_argument(
        "--bg-music",
        default=None,
        metavar="TRACK",
        help=(
            "Background-music track overriding any [bgm: ...] note tags: a "
            "bare name looked up under assets/ or a path to an .mp3/.wav"
        ),
    )
    bg_music_group.add_argument(
        "--no-bg-music",
        action="store_true",
        help="Ignore [bgm: ...] note tags; assemble without background music",
    )

    parser.add_argument(
        "--transition",
        choices=list(TRANSITION_STYLES),
        default=DEFAULT_TRANSITION_STYLE,
        help=(
            "Effect at each slide change: 'dip-black' fades through black "
            "(default), 'dip-white' flashes through white, 'crossfade' "
            "still-dissolves between slides, 'none' hard-cuts. Every style "
            "also fades out to black at the end; the first slide is fully "
            "visible from frame 0."
        ),
    )
    parser.add_argument(
        "--transition-duration",
        type=float,
        default=DEFAULT_TRANSITION_SECONDS,
        metavar="SECONDS",
        help=(
            "Length of each slide transition and of the opener/closer fades "
            f"(default: {DEFAULT_TRANSITION_SECONDS}; 0 disables transitions)"
        ),
    )

    args = parser.parse_args()

    sys.exit(
        main(
            args.source,
            only_slide=args.only_slide,
            profile_id=args.profile_id,
            output_mp4=args.output_mp4,
            export_only=args.export_only,
            gen_voiceover=args.gen_voiceover,
            personality=args.personality,
            fps=args.fps,
            voicebox_url=args.voicebox_url,
            use_voicebox=args.voicebox,
            inter_slide_pause_seconds=args.inter_slide_pause,
            split_at=args.split_at,
            ken_burns_zoom=DEFAULT_KEN_BURNS_ZOOM if args.ken_burns else 0.0,
            engine=args.engine,
            bg_music=args.bg_music,
            no_bg_music=args.no_bg_music,
            transition=args.transition,
            transition_seconds=args.transition_duration,
            language=args.language,
        )
    )
