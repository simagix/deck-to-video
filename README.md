# deck_to_video

Turn a presentation deck into a narrated MP4 video. Export slides and speaker notes from **Google Slides** or a local **PPTX** file, generate voiceover audio with a local **[Voicebox](https://github.com/jamiepine/voicebox)** instance, and assemble everything into video with MoviePy.

## Features

- **Google Slides** — export slide PNGs and speaker notes via the Google Drive API
- **PPTX** — render slides locally with LibreOffice and read embedded speaker notes
- **Voiceover** — synthesize narration from speaker notes using Voicebox
- **Video assembly** — combine slides and audio into 1080p MP4 files
- **Flexible output** — export assets only, process a single slide, split into multiple videos, or customize FPS and pauses

## Prerequisites

| Requirement | When needed |
|-------------|-------------|
| Python 3.9+ | Always |
| [Voicebox](https://github.com/jamiepine/voicebox) (desktop app) | Only when using `--gen-voiceover` |
| Google Cloud OAuth credentials | Google Slides input only |
| [LibreOffice](https://www.libreoffice.org/) (`soffice` on `PATH`) | PPTX input only |

MoviePy bundles FFmpeg via `imageio-ffmpeg`, so you do not need to install FFmpeg separately.

## Installation

```bash
git clone https://github.com/simagix/deck_to_video.git
cd deck_to_video
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration

### Voicebox

1. Start the Voicebox desktop app.
2. Create a `.env` file in the project root:

```env
VOICEBOX_PROFILE_ID=your-profile-uuid
```

Optional overrides:

```env
VOICEBOX_API_URL=http://127.0.0.1:17493
```

List available profiles:

```bash
curl http://127.0.0.1:17493/profiles
```

You can also pass `--profile-id` on the command line instead of using `.env`.

### Google Slides (optional)

If you use Google Slides as input:

1. Create a Google Cloud project and enable the **Google Drive API**.
2. Create OAuth 2.0 credentials (Desktop app) and download `credentials.json`.
3. Place `credentials.json` in the project root.
4. Run the script once; it will open a browser to authorize and save `token.json`.

Both `credentials.json` and `token.json` are gitignored — do not commit them.

## Usage

### Basic examples

```bash
# Google Slides (ID or full URL)
python deck_to_video.py 1abcDEFghijklmnop
python deck_to_video.py "https://docs.google.com/presentation/d/1abcDEFghijklmnop/edit"

# Local PPTX
python deck_to_video.py my_deck.pptx

# Generate voiceovers (first run, or when notes changed)
python deck_to_video.py my_deck.pptx --gen-voiceover

# Rebuild video only — reuses existing slide_XX_voiceover.wav files
python deck_to_video.py my_deck.pptx

# Custom output path
python deck_to_video.py my_deck.pptx -o presentation.mp4
```

### Common options

```bash
# Process one slide for a quick test
python deck_to_video.py my_deck.pptx --only-slide 3

# Export PNGs and notes only (skip video assembly)
python deck_to_video.py my_deck.pptx --export-only

# Export PNGs, notes, and new voiceovers
python deck_to_video.py my_deck.pptx --export-only --gen-voiceover

# Split into multiple videos at slide boundaries
python deck_to_video.py my_deck.pptx --split-at 10,20

# Adjust video settings
python deck_to_video.py my_deck.pptx --fps 30 --inter-slide-pause 0.5
```

### CLI reference

| Flag | Description |
|------|-------------|
| `source` | Google Slides ID/URL or path to a `.pptx` file |
| `--profile-id` | Voicebox profile UUID (overrides `.env`) |
| `--voicebox-url` | Voicebox API base URL (default: `http://127.0.0.1:17493`) |
| `--only-slide N` | Process a single slide (1-based index) |
| `-o`, `--output` | Output MP4 path |
| `--export-only` | Export PNGs and notes; skip MP4 assembly |
| `--gen-voiceover` | Generate voiceover WAVs via Voicebox (default: reuse existing files) |
| `--split-at N[,N...]` | Split into multiple MP4s at 1-indexed slide numbers |
| `--fps` | Video frame rate (default: `24`) |
| `--inter-slide-pause SECONDS` | Silent hold between slides (default: `1.0`; use `0` to disable) |

## Output

Files are written to `out/<sanitized_deck_title>/`:

```
out/my_presentation/
├── deck_title.txt
├── slide_01.png
├── slide_01_notes.txt
├── slide_01_voiceover.wav
├── slide_02.png
├── ...
└── my_presentation.mp4
```

- Slides without speaker notes get a short silent segment (default: 3 seconds).
- With `--split-at`, multiple MP4 files are created (e.g. `my_presentation_part1.mp4`, `part2.mp4`, …).
- With `--export-only`, PNG and note files are produced but no MP4 is assembled.
- By default, existing `slide_XX_voiceover.wav` files are reused for video assembly. Pass `--gen-voiceover` to regenerate them via Voicebox.

## Project structure

```
deck_to_video/
├── deck_to_video.py    # CLI entry point
├── google_slides.py    # Google Slides export
├── pptx_source.py      # PPTX export (LibreOffice)
├── voicebox_client.py  # Voicebox TTS client
├── narration.py        # Speaker-note text normalization
├── video_assembly.py   # MoviePy video assembly
├── images.py           # Slide image resizing
├── split_ranges.py     # Multi-part video splitting
├── paths.py            # Shared paths and defaults
├── requirements.txt
└── out/                # Generated output (gitignored)
```

## Troubleshooting

**`VOICEBOX_PROFILE_ID is not set`** — only required with `--gen-voiceover`. Add the profile UUID to `.env` or pass `--profile-id`.

**Google Slides auth errors** — ensure `credentials.json` exists and delete `token.json` to re-authenticate.

**PPTX rendering fails** — install LibreOffice and confirm `soffice` is available:

```bash
which soffice
```

PPTX slide images are rendered by converting the deck to PDF with LibreOffice, then rasterizing each PDF page with `pypdfium2` (installed via `pip install -r requirements.txt`). If you see a slide-count mismatch error, the PDF export may have failed partially — try opening the PPTX in LibreOffice Impress manually.

**No slide PNGs exported** — check that the deck is accessible (Google Slides permissions) or that the PPTX file path is correct.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for the full text.
