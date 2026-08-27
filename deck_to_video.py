#!/usr/bin/env python3
"""
Export a deck (Google Slides or local PPTX) to PNG + speaker notes, generate
voiceover via local Voicebox, and assemble MP4 video(s) with MoviePy.

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
        parse_bgm_cues,
        prepare_narration,
        split_notes_on_sfx,
    )
    from paths import (
        DEFAULT_BG_MUSIC_VOLUME,
        DEFAULT_FPS,
        DEFAULT_INTER_SLIDE_PAUSE_SECONDS,
        DEFAULT_KEN_BURNS_ZOOM,
        DEFAULT_SILENT_SLIDE_SECONDS,
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
    from video_assembly import assemble_presentation_video
    from voicebox_client import (
        SUPPORTED_ENGINES,
        generate_voicebox_audio,
        get_voicebox_config,
        personality_enabled_from_env,
    )
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
) -> List[int]:
    missing: List[int] = []
    for slide_idx, notes_text in enumerate(notes_per_slide, start=1):
        if not prepare_narration(notes_text, personality=personality):
            continue
        if not os.path.isfile(_voiceover_wav_path(output_dir, slide_idx)):
            missing.append(slide_idx)
    return missing


def _ensure_terminal_punctuation(text: str) -> str:
    """End narration with a sentence terminator so Voicebox closes its prosody."""
    if text[-1] not in (".", "!", "?"):
        return text.rstrip() + "."
    return text


def _generate_voiceover_take(
    source: str,
    output_wav: str,
    api_base: str,
    profile_id: str,
    personality: bool,
    engine: Optional[str],
) -> bool:
    """Synthesize one narration take via Voicebox; False when nothing to say."""
    narration_text = prepare_narration(source, personality=personality)
    if not narration_text.strip():
        return False
    generate_voicebox_audio(
        _ensure_terminal_punctuation(narration_text),
        profile_id=profile_id,
        output_wav=output_wav,
        api_base=api_base,
        personality=personality,
        engine=engine,
        instruct=_first_tone_instruct(source),
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
) -> Optional[str]:
    if not prepare_narration(notes_text, personality=personality):
        print(
            f"   ⏭️  Slide {slide_idx}: no narration "
            f"(will use {DEFAULT_SILENT_SLIDE_SECONDS}s silent)"
        )
        return None

    wav_path = _voiceover_wav_path(output_dir, slide_idx)
    parts = split_notes_on_sfx(notes_text, supported=set(SFX_SAMPLES))

    if not any(part["kind"] == "sfx" for part in parts):
        # Single-take path (no sound-effect cues): one Voicebox call, exactly
        # as before. The take keeps the first tone found anywhere in the notes.
        if not _generate_voiceover_take(
            notes_text, wav_path, api_base, profile_id, personality, engine
        ):
            print(f"   ⏭️  Slide {slide_idx}: no narration after prep")
            return None
        # Append a short tail of silence so the final word is never truncated
        # and voiced slides flow into the next with a natural 1s gap.
        _append_silence_to_wav(
            wav_path, duration=DEFAULT_VOICEOVER_TRAIL_SILENCE_SECONDS
        )
        print(f"   ✅ Slide {slide_idx}: saved {wav_path}")
        return wav_path

    # Multi-take path: each recognized "[sfx: ...]" tag splits the notes into
    # separate takes. The sample is spliced between them — landing exactly
    # where the tag sat, even mid-slide — and each take keeps its own
    # [voice:/tone:] context as its style instruct.
    pieces: List[Tuple[str, str]] = []
    inserted: List[str] = []
    take_no = 0
    with tempfile.TemporaryDirectory(prefix=f"slide_{slide_idx:02d}_sfx_") as tmp_dir:
        for part in parts:
            if part["kind"] == "sfx":
                pieces.append(("sfx", part["name"]))
                continue
            take_no += 1
            take_path = os.path.join(tmp_dir, f"take_{take_no:02d}.wav")
            if _generate_voiceover_take(
                part["source"],
                take_path,
                api_base,
                profile_id,
                personality,
                engine,
            ):
                pieces.append(("wav", take_path))
        inserted = assemble_voiceover(pieces, wav_path)
    for name in inserted:
        print(f"   🥁 Slide {slide_idx}: {name} mid-slide")
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
) -> List[Optional[str]]:
    if gen_voiceover:
        if not api_base or not profile_id:
            raise RuntimeError("Voicebox config is required when using --gen-voiceover")
        print(f"\n🎙️  Generating voiceovers via Voicebox ({api_base})...")
    elif generate_missing:
        missing = _missing_voiceover_slide_numbers(
            output_dir, notes_per_slide, personality=personality
        )
        if missing:
            if not api_base or not profile_id:
                slides = ", ".join(str(idx) for idx in missing)
                raise RuntimeError(
                    f"Missing voiceover WAV(s) for slide(s) {slides}. "
                    "Set VOICEBOX_PROFILE_ID in .env (or pass --profile-id), "
                    "or pass --gen-voiceover to generate them."
                )
            print(
                f"\n🎙️  Generating missing voiceovers for slide(s) "
                f"{', '.join(str(idx) for idx in missing)} via Voicebox ({api_base})..."
            )
        else:
            print("\n🎙️  Using existing voiceover files (pass --gen-voiceover to regenerate)...")
    else:
        print("\n🎙️  Using existing voiceover files (pass --gen-voiceover to regenerate)...")

    wav_paths: List[Optional[str]] = []
    for slide_idx, notes_text in enumerate(notes_per_slide, start=1):
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
    inter_slide_pause_seconds: float = DEFAULT_INTER_SLIDE_PAUSE_SECONDS,
    split_at: Optional[str] = None,
    ken_burns_zoom: float = DEFAULT_KEN_BURNS_ZOOM,
    engine: Optional[str] = None,
    bg_music: Optional[str] = None,
    no_bg_music: bool = False,
) -> int:
    try:
        use_personality = (
            personality_enabled_from_env()
            if personality is None
            else personality
        )
        api_base: Optional[str] = None
        voicebox_profile_id: Optional[str] = None
        voicebox_engine: Optional[str] = None

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

        notes_per_slide = []
        for note_path in _sorted_slide_assets(out_dir, "_notes.txt"):
            with open(note_path, encoding="utf-8") as note_file:
                notes_per_slide.append(note_file.read())

        png_paths = _sorted_slide_assets(out_dir, ".png")
        if not png_paths:
            raise RuntimeError("No slide PNGs were exported")

        if export_only and not gen_voiceover:
            print(f"\n✅ Export complete (PNGs + notes) in {os.path.abspath(out_dir)}")
            return 0

        generate_missing = not export_only
        needs_voicebox = gen_voiceover or (
            generate_missing
            and bool(
                _missing_voiceover_slide_numbers(
                    out_dir, notes_per_slide, personality=use_personality
                )
            )
        )
        if needs_voicebox:
            api_base, voicebox_profile_id, voicebox_engine = get_voicebox_config(
                profile_id,
                engine_override=engine,
            )
            if voicebox_url:
                api_base = voicebox_url.rstrip("/")

        wav_paths = _voiceover_paths_for_slides(
            out_dir,
            notes_per_slide,
            gen_voiceover=gen_voiceover,
            generate_missing=generate_missing,
            personality=use_personality,
            api_base=api_base,
            profile_id=voicebox_profile_id,
            engine=voicebox_engine,
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
        help="Export PNGs and notes only; skip MoviePy video assembly",
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
            inter_slide_pause_seconds=args.inter_slide_pause,
            split_at=args.split_at,
            ken_burns_zoom=DEFAULT_KEN_BURNS_ZOOM if args.ken_burns else 0.0,
            engine=args.engine,
            bg_music=args.bg_music,
            no_bg_music=args.no_bg_music,
        )
    )
