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
from typing import List, Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import dotenv  # type: ignore[import-untyped]
import requests  # type: ignore[import-untyped]

from google_slides import (
    check_document_type,
    export_slides_to_png,
    export_speaker_notes,
    extract_presentation_id,
    get_presentation_title,
    get_skipped_slide_indices,
)
from narration import prepare_narration
from paths import (
    DEFAULT_FPS,
    DEFAULT_INTER_SLIDE_PAUSE_SECONDS,
    DEFAULT_SILENT_SLIDE_SECONDS,
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
from video_assembly import assemble_presentation_video
from voicebox_client import (
    generate_voicebox_audio,
    get_voicebox_config,
    personality_enabled_from_env,
)

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


def _generate_voiceover_for_slide(
    output_dir: str,
    slide_idx: int,
    notes_text: str,
    api_base: str,
    profile_id: str,
    personality: bool,
) -> Optional[str]:
    narration = prepare_narration(notes_text, personality=personality)
    if not narration:
        print(
            f"   ⏭️  Slide {slide_idx}: no narration "
            f"(will use {DEFAULT_SILENT_SLIDE_SECONDS}s silent)"
        )
        return None

    wav_path = _voiceover_wav_path(output_dir, slide_idx)
    generate_voicebox_audio(
        narration,
        profile_id=profile_id,
        output_wav=wav_path,
        api_base=api_base,
        personality=personality,
    )
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
) -> int:
    try:
        use_personality = (
            personality_enabled_from_env()
            if personality is None
            else personality
        )
        api_base: Optional[str] = None
        voicebox_profile_id: Optional[str] = None

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
            api_base, voicebox_profile_id = get_voicebox_config(profile_id)
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
        )
        while len(wav_paths) < len(png_paths):
            wav_paths.append(None)

        if export_only:
            print(f"\n✅ Export complete (PNGs + WAVs) in {os.path.abspath(out_dir)}")
            return 0

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
    personality_group = parser.add_mutually_exclusive_group()
    personality_group.add_argument(
        "--personality",
        dest="personality",
        action="store_true",
        help="Rewrite speaker notes in the profile's voice before TTS (default: off)",
    )
    personality_group.add_argument(
        "--no-personality",
        dest="personality",
        action="store_false",
        help="Send speaker notes to Voicebox as plain TTS (default)",
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
            "Silent hold after each slide before the next (default: "
            f"{DEFAULT_INTER_SLIDE_PAUSE_SECONDS}; 0 to disable)"
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
            inter_slide_pause_seconds=args.inter_slide_pause,
            split_at=args.split_at,
        )
    )
