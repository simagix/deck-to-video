# deck-to-video

Turn a presentation deck into a narrated MP4 video. Export slides and speaker
notes from a local **PPTX** file, generate voiceover audio with a local TTS
engine — **[one-voice](https://github.com/simagix/one-voice)** (Qwen3-TTS on
Apple Silicon, the default) or **[Voicebox](https://github.com/jamiepine/voicebox)**
— and assemble everything into video with MoviePy.

> Google Slides IDs/URLs are also accepted, but PPTX is the recommended and fully documented workflow — see [Google Slides support (optional)](#google-slides-support-optional).

<!-- ADD VISUALIZATION HERE -->
![A flowchart showing the Deck-to-Video pipeline: input presentation (Google Slides or PPTX), output of extracted images and notes, voiceover processing via the local one-voice TTS engine (or the Voicebox fallback, optionally with a cloned signature voice), and video assembly by MoviePy, leading to the final 1080p MP4.](workflow_diagram.png)

## Features

- **PPTX** — render slides locally with LibreOffice and read embedded speaker notes
- **Voiceover** — synthesize narration from speaker notes with local one-voice (Qwen3-TTS, Apple Silicon) or Voicebox (`--voicebox`)
- **Voice & tone tags** — direct a single slide (or even a paragraph mid-slide) to a cloned voice / style with `[voice: NAME | tone: TAG]`; the tone direction is spoken by one-voice and auto-leak-trimmed, or sent as a rich `instruct` to Voicebox
- **Punchline sound effects** — tag jokes with `[sfx: rimshot]` (*Ba Dum Tss*) or `[sfx: sad_trombone]` (*wah-wah-waaah*) right after the punchline
- **Background music** — drop one `[bgm: ambient_loop]` tag anywhere in your speaker notes to soundtrack the entire video under the narration
- **Google Slides (optional)** — also accepts Google Slides IDs/URLs via the Google Drive API (requires OAuth credentials)
- **Cloned voice / signature narrator** — generate every voiceover with a cloned voice for a consistent, recognizable brand voice across a series (a one-voice voice profile or a Voicebox clone)
- **Video assembly** — combine slides and audio into 1080p MP4 files
- **Slide transitions** — a quick dip-to-black at each slide change (or dip-to-white / still-dissolve crossfade), plus a fade-through-black closer; the first slide stays fully visible from frame 0; `--transition none` for hard cuts
- **Flexible output** — export assets only, process a single slide, split into multiple videos, or customize FPS and pauses

## Prerequisites

| Requirement | When needed |
|-------------|-------------|
| Python 3.12+ | Always |
| Apple Silicon (Metal GPU) | one-voice narration (default) |
| [one-voice](https://github.com/simagix/one-voice) package | one-voice narration (included in `requirements.txt`; Apple-Silicon sibling repo with `voices/`) |
| [Voicebox](https://github.com/jamiepine/voicebox) desktop app | Only when using `--voicebox` |
| [LibreOffice](https://www.libreoffice.org/) (`soffice` on `PATH`) | PPTX input (required) |

MoviePy bundles FFmpeg via `imageio-ffmpeg`, so you do not need to install FFmpeg separately.

Narration defaults to the **one-voice** engine (Qwen3-TTS on Apple Silicon). On
other platforms, or if you prefer voice cloning inside the Voicebox app, pass
`--voicebox` (requires the Voicebox desktop app running; its API is <localhost>).

Google Slides input additionally needs Google Cloud OAuth credentials — see [Google Slides support (optional)](#google-slides-support-optional).

## Installation

```bash
git clone https://github.com/simagix/deck-to-video.git
cd deck-to-video
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# PPTX rendering also needs the LibreOffice desktop app (not installable via pip):
brew install --cask libreoffice   # macOS — see https://www.libreoffice.org/ for other OSes
```

## Configuration

All configuration lives in a `.env` file at the project root — copy
[`.env.example`](.env.example) and fill in your values. `.env` is git-ignored;
`.env.example` is the checked-in template.

### One-voice (default narration engine)

`--gen-voiceover` synthesizes narration with the **one-voice** engine
(Qwen3-TTS on Apple Silicon) unless you pass `--voicebox`. The voice search
path is auto-configured on import: this repo's `voices/` (project-specific,
e.g. `simone`) plus the installed one-voice package's `voices/` (base voices
like `golding`, `neufeld`). Leftmost wins when the same name exists in both.

Create a `.env` file in the project root to override the default voice and/or
the search path:

```env
# Voice used when a slide has no [voice: ...] tag (defaults to the first
# available voice profile).
ONE_VOICE=simone

# Optional: colon-separated list of directories to search for voices.
# Defaults to this repo's voices/ + the installed one-voice package's voices/.
ONE_VOICE_PATH=/path/to/voices:/another/voices/dir
```

List available voice profiles (from the one-voice tool itself):

```bash
one-voice voices
```

There is no server to start — one-voice runs locally and shares MLX with this
repo (Apple Silicon only). For non-Apple-Silicon machines, or if you prefer
voice cloning inside the Voicebox app, use `--voicebox` (next section).

### Voicebox (optional; `--voicebox`)

1. Start the Voicebox desktop app.
2. Create a `.env` file in the project root:

```env
VOICEBOX_PROFILE_ID=your-profile-uuid
```

Optional overrides:

```env
VOICEBOX_API_URL=http://127.0.0.1:17493

# TTS engine used for synthesis (overrides the profile's "Default Engine").
# Supported: qwen (Qwen3-TTS), qwen_custom_voice, luxtts, chatterbox,
#            chatterbox_turbo, tada, kokoro
VOICEBOX_ENGINE=qwen
```

> **On "Default Engine" vs. "Refinement model":** these are two different
> things. The profile's **Default Engine** is the *TTS engine* that actually
> synthesizes the audio (e.g. `qwen` = Qwen3-TTS). The **Refinement model**
> (Settings → Captures → Refinement) is a separate small Qwen3 *LLM* that
> only *rewrites text* for dictation refinement and `--personality` — it never
> produces audio and does **not** control which engine speaks. So to use
> Qwen3-TTS, set `VOICEBOX_ENGINE=qwen` (or the profile's default engine); the
> refinement model is irrelevant to which engine produces the speech.

List available profiles:

```bash
curl http://127.0.0.1:17493/profiles
```

You can also pass `--profile-id` on the command line instead of using `.env`.

#### Using a cloned voice (signature / branding)

A cloned voice keeps a video series recognizable: the same narrator across
every video, so audiences associate it with your brand or creator. Cloning
works in both engines:

- **one-voice (default):** a "voice profile" is a directory with a short
  reference recording. Add `voices/<name>/reference.wav` (optionally a
  `voice.yaml` with a description); the voice then appears in
  `one-voice voices` and can be selected with `ONE_VOICE`, `--profile-id
  <name>`, or a `[voice: <name>]` tag.
- **Voicebox (`--voicebox`):** clone a voice in the desktop app, find its
  profile UUID, and point the tool at that clone:

```env
VOICEBOX_PROFILE_ID=your-clone-uuid
```

Because the profile persists across projects, all your videos share one
signature voice.

> **Tip (Voicebox only):** combine a clone with the `--personality` flag — the
> speaker notes are rewritten in the profile's voice first, then spoken by the
> clone, which reinforces the signature character even more.

### Voice & tone tags (optional)

You can steer the delivery of a single slide's voiceover by tagging your
speaker notes with a light, human-friendly syntax. The tags are **metadata** —
they are stripped from the text sent to the TTS engine:

- **one-voice (default):** each `voice:` names a real voice profile to clone
  from (the project's `voices/` + the installed one-voice package's `voices/`);
  each `tone:` becomes an in-band direction spoken by the model and manually
  trimmed out (see below).
- **Voicebox (`--voicebox`):** `voice:` is a convenience label in the text; the
  actual speaker is still chosen by `VOICEBOX_PROFILE_ID` / `--profile-id`.
  Each `tone:` is translated into a rich style `instruct` on the `/generate`
  payload, so the author only writes a short keyword while the tool sends a
  full instruction.

```text
[voice: Spirit | tone: frustrated]

We shipped the new deployment process, and then it fell over.
```

A tone-only tag works too — handy for a single speaker who just wants to change delivery mid-slide:

```text
Hi... I'm Simone... the Hatchet assistant.
[tone: frustrated]
My typical workday begins long... before I even log on.
```

- `voice` — with one-voice (default) this selects a real voice profile from
  the search path (falls back to `ONE_VOICE` / the first available profile).
  With `--voicebox` it is a convenience label; the actual speaker is still
  chosen by `VOICEBOX_PROFILE_ID` / `--profile-id`.
- `tone` — maps to one of the 28 built-in tones below. With one-voice it becomes
  an in-band direction that is leak-trimmed from the audio; with Voicebox it
  becomes the `/generate` `instruct`.
- Tags are **sticky** within a slide: the last tag's voice & tone persist until
  changed or the block ends. A tag that sets only `voice` or only `tone`
  leaves the other dimension unchanged. Multi-voice/multi-tone slides
  generate one take per block (`slide_XX_voiceover.wav` is assembled from the
  pieces).
- Tags are fully optional. Untagged speaker notes behave exactly as before (no
  tone direction is sent).

The full tone vocabulary lives in `narration.py` (`TONES`), including
`neutral`, `professional`, `friendly`, `warm`, `cheerful`, `excited`,
`enthusiastic`, `confident`, `serious`, `concerned`, `frustrated`, `angry`,
`sad`, `disappointed`, `surprised`, `confused`, `curious`, `skeptical`,
`sarcastic`, `humorous`, `witty`, `dramatic`, `mysterious`, `narrative`,
`explainer`, `whisper`, `robotic`, and `urgent` (matching one-voice's own
`TONES` vocabulary of 31, a superset of these 28).

To inspect the exact JSON that will be sent for a notes file, run:

```bash
python show_voicebox_payload.py out/<deck>/slide_01_notes.txt
```

### Sound effect tags (punchlines)

Drop a bracketed SFX tag right after a punchline — even mid-slide with more
narration following. The notes are split at the tag into separate narration
takes, and the sample is spliced exactly between them (after a short comedic
beat, ~0.25s), so the hit lands precisely where the tag sat in the text:

```text
[voice: Simone | tone: witty]

I asked the intern to auto-merge. [laugh] It merged main into staging.

[sfx: rimshot]

[tone: dramatic]
Twice.
```

- **Rimshot** — `[sfx: rimshot]`, `[badumtss]`, `[ba dum tss]`, `[rimshot]` all play *Ba Dum Tss*. **Sad trombone** — `[sfx: sad_trombone]`, `[sad trombone]`, `[wah wah wah]`. **Drum roll** — `[sfx: drum_roll]`, `[drumroll]`, `[drum roll]` rolls for ~4s, so place it *before* the reveal (`And the winner is… [drumroll] …you!`) to build tension into the punchline. Multiple tags per slide are supported.
- **Sample resolution:** your `assets/ba_dum_tss.wav` always wins; `ba_dum_tss_default.wav` is only a fallback so fresh clones work out of the box (`make_sfx_assets.py` regenerates it without ever touching yours).
- **Stereo survives the splice:** if your sample has more channels than the narration, the slide WAV is upgraded to match and the voice duplicated across channels — a true-stereo rimshot keeps its left/right image (dual-mono recordings are unaffected).
- Each take keeps its own `[voice:` / `[tone:]` context as its style instruct (the example above speaks the setup angry and the follow-up dramatic). Slides with voice/tone tags make one extra TTS call per split.
- Tags are stripped from the text sent to the TTS engine (never read aloud); unknown names are ignored safely.
- The result is baked into `slide_XX_voiceover.wav` at generation time, so video assembly needs no changes and cached-WAV reuse keeps working. Re-run with `--gen-voiceover` after adding or editing a tag.
- **Your own recording:** drop `assets/ba_dum_tss.wav` (convert any licensed rimshot to WAV) and it takes precedence over the synthesized default in `ba_dum_tss_default.wav`, which `make_sfx_assets.py` regenerates without ever touching your file.

### Background music tags ([bgm: ...])

Unlike punchline SFX, background music is **deck-level**: the FIRST `[bgm: ...]`
tag found across any slide's speaker notes selects ONE continuous track, layered
underneath the whole assembled narration+SFX timeline at render time (so it can
span slide boundaries and survive cached-WAV reuse):

```text
[voice: Simone | tone: warm]

Welcome, everyone. Today we automate the boring parts.

[bgm: ambient_loop | volume: 0.18]
```

- **Where it comes from:** a bare name resolves inside `assets/` trying `.mp3`
  then `.wav`; you may also give an explicit path (`[bgm: ~/music/vip.mp3]`).
  A quiet CC0 starter pad ships as `assets/ambient_loop.mp3`, regenerable via
  `python make_bgm_assets.py`.
- **Volume** — defaults to `DEFAULT_BG_MUSIC_VOLUME` (0.15) from `paths.py`;
  override per-tag with `| volume: 0.2`. The track auto-loops until the video
  ends and always respects the narration's own sample rate / channel layout.
- **CLI escape hatches:** `--bg-music TRACK` overrides every tag; mutually
  exclusive `--no-bg-music` silences the deck regardless of notes. A typo'd
  track fails loudly instead of rendering silently missing its score.
- Tags are stripped from the text sent to the TTS engine (never read aloud).

## Usage

### Basic examples

```bash
# Local PPTX
python deck_to_video.py my_deck.pptx

# Generate voiceovers (first run, or when notes changed).
# Defaults to the local one-voice engine; --voicebox switches to the Voicebox app.
python deck_to_video.py my_deck.pptx --gen-voiceover

# Rebuild video only — reuses existing slide_XX_voiceover.wav files
python deck_to_video.py my_deck.pptx

# Custom output path
python deck_to_video.py my_deck.pptx -o presentation.mp4

# Show the installed version
python deck_to_video.py --version   # deck_to_video v0.3.2
```

### Common options

```bash
# Render a single slide into its own video (does NOT touch the full deck's MP4).
# That slide's files keep their real number, so a previous run's voiceover for
# slide 3 is reused correctly; pass --gen-voiceover to regenerate narration.
python deck_to_video.py my_deck.pptx --only-slide 3

# Export PNGs and notes only (skip video assembly)
python deck_to_video.py my_deck.pptx --export-only

# Export PNGs, notes, and new voiceovers
python deck_to_video.py my_deck.pptx --export-only --gen-voiceover

# Split into multiple videos at slide boundaries
python deck_to_video.py my_deck.pptx --split-at 10,20

# Adjust video settings
python deck_to_video.py my_deck.pptx --fps 30 --inter-slide-pause 0.5

# Add Ken Burns zoom/pan to each slide
python deck_to_video.py my_deck.pptx --ken-burns

# Pick the slide-change effect and its length (dip-black is the default)
python deck_to_video.py my_deck.pptx --transition dip-white --transition-duration 0.6
python deck_to_video.py my_deck.pptx --transition none   # hard cuts, no transitions
```

### CLI reference

| Flag | Description |
|------|-------------|
| `source` | Path to a `.pptx` file (Google Slides ID/URL also accepted) |
| `--profile-id` | one-voice voice name (`--profile-id simone`) or a Voicebox profile UUID (overrides `.env`) |
| `--voicebox` | Use the Voicebox API for narration instead of the default one-voice local TTS |
| `--engine NAME` | Voicebox TTS engine, overriding the profile's Default Engine (`qwen`, `qwen_custom_voice`, `luxtts`, `chatterbox`, `chatterbox_turbo`, `tada`, `kokoro`); ignored by one-voice |
| `--voicebox-url` | Voicebox API base URL (default: `http://127.0.0.1:17493`) |
| `--only-slide N` | Process a single slide (1-based index). Only that slide is exported and rendered, producing one MP4 named `<deck>-slide-NN.mp4` (the full deck's video is never overwritten). The slide's files keep their real number (`slide_NN.png` / `slide_NN_notes.txt` / `slide_NN_voiceover.wav`), so a previous run's voiceover for that slide is reused correctly; pass `--gen-voiceover` to regenerate it |
| `-o`, `--output` | Output MP4 path |
| `--export-only` | Export PNGs and notes; skip MP4 assembly |
| `--gen-voiceover` | Generate voiceover WAVs (via one-voice by default, or Voicebox with `--voicebox`) — default: reuse existing files |
| `--personality` | Rewrite speaker notes in the profile's voice before TTS (default: off; also enabled by `VOICEBOX_PERSONALITY=1`) |
| `--split-at N[,N...]` | Split into multiple MP4s at 1-indexed slide numbers |
| `--fps` | Video frame rate (default: `24`) |
| `--inter-slide-pause SECONDS` | Silent hold after slides without voiceover (default: `1.0`; voiced slides carry a built-in 1s tail) |
| `--ken-burns` | Apply the Ken Burns zoom/pan effect to each slide (zoom x1.08 over the slide duration). By default slides are rendered as static images (default: OFF) |
| `--transition STYLE` | Slide-change effect: `dip-black` (default), `dip-white`, `crossfade`, or `none`. Every style also fades out to black at the end; the first slide is fully visible from frame 0 |
| `--transition-duration SECONDS` | Length of each slide transition and of the opener/closer fades (default: `0.5`; `0` disables transitions) |

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
- Each voiceover WAV ends with a built-in 1-second tail of silence, so voiced slides flow into the next slide with a natural 1s gap (no separate inter-slide pause is added on top).
- With `--split-at`, multiple MP4 files are created (e.g. `my_presentation_part1.mp4`, `part2.mp4`, …).
- With `--export-only`, PNG and note files are produced but no MP4 is assembled.
- By default, existing `slide_XX_voiceover.wav` files are reused for video assembly. Pass `--gen-voiceover` to regenerate them (one-voice by default; Voicebox with `--voicebox`).

## Google Slides support (optional)

<details>
<summary>Google Slides IDs/URLs still work, but need one-time OAuth setup.</summary>

`deck_to_video.py` can also read a deck directly from a Google Slides ID or URL. That code path is still supported, but it is not the focus of this guide and requires Google Cloud OAuth credentials:

1. Create a Google Cloud project and enable the **Google Drive API**.
2. Create OAuth 2.0 credentials (Desktop app) and download `credentials.json` into the project root.
3. Run the script once — it opens a browser to authorize and saves `token.json` for later runs.

```bash
python deck_to_video.py 1abcDEFghijklmnop
python deck_to_video.py "https://docs.google.com/presentation/d/1abcDEFghijklmnop/edit"
```

`credentials.json` and `token.json` are both gitignored. If you hit auth errors, delete `token.json` and re-authorize.

</details>

## Project structure

```
deck-to-video/
├── deck_to_video.py      # CLI entry point
├── google_slides.py      # Google Slides export (optional)
├── pptx_source.py        # PPTX export (LibreOffice)
├── one_voice_adapter.py  # one-voice TTS adapter (default narration engine)
├── voicebox_client.py    # Voicebox TTS client (fallback, --voicebox)
├── narration.py          # Speaker-note text normalization + voice/tone splits
├── sfx.py                # Punchline sound-effect WAV mixing
├── bgm.py                # Background-music synthesis
├── make_sfx_assets.py    # Generates assets/ sound-effect samples
├── make_bgm_assets.py    # Generates assets/ambient_loop.mp3
├── assets/               # Bundled sound-effect + bg-music samples
├── voices/               # Project voice profiles (e.g. voices/simone/)
├── video_assembly.py     # MoviePy video assembly
├── images.py             # Slide image resizing
├── split_ranges.py       # Multi-part video splitting
├── paths.py              # Shared paths and defaults
├── requirements.txt
└── out/                  # Generated output (gitignored)
```

## Troubleshooting

**`ModuleNotFoundError: No module named 'dotenv'`** — `python` isn't running from the project virtualenv. Either activate it first:

```bash
source .venv/bin/activate
python deck_to_video.py my_deck.pptx --gen-voiceover
```

or call the venv interpreter directly:

```bash
.venv/bin/python deck_to_video.py my_deck.pptx --gen-voiceover
```

(The PyPI package is `python-dotenv`, which provides the `dotenv` module — it's already in `requirements.txt`. If the venv itself is missing packages, re-run `pip install -r requirements.txt`.)

**`VOICEBOX_PROFILE_ID is not set`** — only required when running with `--voicebox` (or when one-voice maps via this legacy fallback). With the default one-voice engine, set `ONE_VOICE` in `.env` (or let it fall back to the first available voice profile); with Voicebox, add the profile UUID to `.env` or pass `--profile-id`.

**PPTX rendering fails** — install LibreOffice and confirm `soffice` is on your `PATH`:

```bash
# macOS / Linux
which soffice
```

On **Windows**, `soffice.exe` is not always added to `PATH` during a default LibreOffice install. It is usually at `C:\Program Files\LibreOffice\program\soffice.exe` — add that folder to your Environment Variables `PATH` (then restart the terminal) if `soffice` isn't recognized.

PPTX slide images are rendered by converting the deck to PDF with LibreOffice, then rasterizing each PDF page with `pypdfium2` (installed via `pip install -r requirements.txt`). If you see a slide-count mismatch error, the PDF export may have failed partially — try opening the PPTX in LibreOffice Impress manually.

**No slide PNGs exported** — check that the PPTX file path is correct, the deck isn't password-protected, and the LibreOffice PDF conversion succeeded (see above).

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for the full text.
