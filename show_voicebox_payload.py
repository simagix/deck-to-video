#!/usr/bin/env python3
"""Print the exact JSON payload deck-to-video would POST to Voicebox /generate.

Reads a speaker-notes file (e.g. out/<deck>/slide_01_notes.txt), runs the same
narration pipeline the app uses (prepare_narration), and prints the payload
without contacting the Voicebox server.

Usage:
    python show_voicebox_payload.py <notes_file> [--personality]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import dotenv  # type: ignore[import-untyped]

from narration import prepare_narration
from paths import ENV_PATH

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Show the JSON payload that would be sent to Voicebox /generate."
    )
    parser.add_argument("notes_file", help="Path to a slide notes .txt file")
    parser.add_argument(
        "--personality",
        action="store_true",
        help="Include the personality flag (text rewrite on the server)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.notes_file):
        print(f"❌ No such file: {args.notes_file}")
        return 1

    dotenv.load_dotenv(ENV_PATH)
    profile_id = os.environ.get("VOICEBOX_PROFILE_ID", "").strip()
    api_base = (
        os.environ.get("VOICEBOX_API_URL")
        or os.environ.get("VOICEBOX_URL")
        or "http://127.0.0.1:17493"
    ).rstrip("/")

    with open(args.notes_file, encoding="utf-8") as f:
        raw = f.read()

    # Exactly what deck_to_video.py calls before handing text to Voicebox.
    prepared = prepare_narration(raw, personality=args.personality)
    if not prepared:
        print("⚠️  prepare_narration returned empty text (nothing would be sent).")
        return 1

    payload: dict[str, object] = {
        "text": prepared,
        "profile_id": profile_id,
        "language": "en",
    }
    if args.personality:
        # Note: generate_voicebox_audio only adds this when the profile has a
        # personality prompt AND the qwen3 refinement model is downloaded.
        payload["personality"] = True

    print(f"Notes file  : {args.notes_file}")
    print(f"api_base    : {api_base}")
    print(f"profile_id  : {profile_id or '<NOT SET — no VOICEBOX_PROFILE_ID in .env>'}")
    print(f"raw notes   : {len(raw)} chars")
    print(f"after narration prep: {prepared!r}")
    print()
    print("=== JSON payload sent to Voicebox /generate ===")
    print(json.dumps(payload, indent=2))
    print()
    print("=== repr of payload['text'] ===")
    print(repr(payload["text"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())