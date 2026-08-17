"""Narration text normalization and pronunciation helpers."""

from __future__ import annotations

import re
import unicodedata
from typing import List, Optional, Tuple


def _strip_control_chars(text: str) -> str:
    if not text:
        return text
    return "".join(c for c in text if c == "\n" or c == "\r" or c == "\t" or ord(c) >= 32)


def _normalize_unicode_punctuation(text: str) -> str:
    if not text:
        return text
    text = (
        text.replace("\u2014", " - ")
        .replace("\u2013", " - ")
        .replace("\u2012", " - ")
        .replace("\u2015", " - ")
    )
    text = (
        text.replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201a", "'")
        .replace("\u201b", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u201e", '"')
        .replace("\u201f", '"')
        .replace("\u02bc", "'")
        .replace("\u02b9", "'")
        .replace("\u2032", "'")
        .replace("\u2033", '"')
    )
    text = text.replace("\u00a0", " ").replace("\u2026", "...")
    result: List[str] = []
    for char in text:
        if ord(char) < 128:
            result.append(char)
        elif char in "\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u200b\u200c\u200d\u200e\u200f":
            result.append(" ")
        elif ord(char) in range(0x2010, 0x2018) or ord(char) in (0x2039, 0x203A):
            result.append(" - " if char in "\u2010\u2011\u2012\u2013" else " ")
        else:
            category = unicodedata.category(char)
            if category.startswith("L") or category.startswith("N"):
                result.append(char)
            else:
                result.append(" ")
    return "".join(result)


def _ensure_space_after_period_before_capital(text: str) -> str:
    if not text:
        return text
    return re.sub(r"\.\s*([A-Z])", r". \1", text)


def replace_words_for_pronunciation(text: str) -> str:
    """Replace words with SSML <sub> tags for correct pronunciation."""
    if not text or not text.strip():
        return text

    pronunciation_overrides = {
        "mongod": "mongo d",
        "mongos": "mongo s",
        "mongosh": "mongo shell",
        "yaml": "yammel",
        "json": "jason",
        "sql": "sequel",
        "gui": "gooey",
        "api": "A P I",
        "cli": "C L I",
    }
    pronunciation_patterns = [
        (re.compile(r"\.wt\b", re.IGNORECASE), "dot-w-t"),
    ]

    result = text
    override_words_upper = {word.upper() for word in pronunciation_overrides}

    for pattern, pron_alias in pronunciation_patterns:
        result = pattern.sub(
            lambda match, alias=pron_alias: f'<sub alias="{alias}">{match.group(0)}</sub>',
            result,
        )

    for word_lower, pron_alias in pronunciation_overrides.items():
        pattern = re.compile(r"\b" + re.escape(word_lower) + r"\b", re.IGNORECASE)
        result = pattern.sub(
            lambda match, alias=pron_alias: f'<sub alias="{alias}">{match.group(0)}</sub>',
            result,
        )

    def replace_uppercase(match: re.Match[str]) -> str:
        word = match.group(0)
        if word.upper() in override_words_upper:
            return word
        alias = "-".join(word)
        return f'<sub alias="{alias}">{word}</sub>'

    result = re.sub(r"\b[A-Z]{2,}\b", replace_uppercase, result)
    return result


def script_text_for_api(text: str) -> str:
    """Prepare script text as SSML with pronunciation tags."""
    if not text or not text.strip():
        return text
    text = _strip_control_chars(text.strip())
    text = _normalize_unicode_punctuation(text)
    text = _ensure_space_after_period_before_capital(text)
    text = replace_words_for_pronunciation(text)

    sub_placeholders: List[str] = []

    def _save_sub(match: re.Match[str]) -> str:
        sub_placeholders.append(match.group(0))
        return f"__SSML_SUB_{len(sub_placeholders) - 1}__"

    text = re.sub(r'<sub\s+alias="[^"]*">.*?</sub>', _save_sub, text, flags=re.DOTALL)
    escaped = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )
    for index, placeholder in enumerate(sub_placeholders):
        escaped = escaped.replace(f"__SSML_SUB_{index}__", placeholder)
    return f"<speak>{escaped}</speak>"


def narration_plain_for_tts(text: str) -> str:
    """Plain-text narration with pronunciation aliases inlined as hyphenated tokens."""
    if not text or not text.strip():
        return ""

    ssml = script_text_for_api(text)
    if not ssml:
        return ""

    inner = re.sub(r"^<speak>(.*)</speak>$", r"\1", ssml, flags=re.DOTALL)
    pattern = re.compile(r'<sub\s+alias="([^"]*)">[^<]*</sub>')
    pieces: List[str] = []
    last_end = 0
    for match in pattern.finditer(inner):
        start, end = match.span()
        alias = re.sub(r"\s+", "-", match.group(1).strip())
        before = inner[last_end:start]
        pieces.append(before)
        if before and before[-1].isalnum():
            pieces.append(" ")
        pieces.append(alias)
        if end < len(inner) and inner[end].isalnum():
            pieces.append(" ")
        last_end = end
    pieces.append(inner[last_end:])
    inner = "".join(pieces)
    return (
        inner.replace("&apos;", "'")
        .replace("&quot;", '"')
        .replace("&gt;", ">")
        .replace("&lt;", "<")
        .replace("&amp;", "&")
    )


# ============================================================================
# Voicebox tone vocabulary — maps short tone names to rich Voicebox ``instruct``
# instruction strings.  Writers tag their speaker notes with the tone name and
# the client converts it to the full instruction before hitting /generate.
# ============================================================================
TONES: dict[str, str] = {
    "neutral": "Natural, conversational, balanced delivery.",
    "professional": "Professional, polished, confident and measured.",
    "friendly": "Warm, friendly and approachable.",
    "warm": "Warm, sincere and personable.",
    "cheerful": "Bright, cheerful and upbeat, with positive energy.",
    "excited": "Excited and energetic, with genuine enthusiasm.",
    "enthusiastic": "Highly enthusiastic and engaged, while remaining natural.",
    "confident": "Confident, assured and authoritative.",
    "serious": "Serious, deliberate and measured, with appropriate weight.",
    "concerned": "Concerned and thoughtful, conveying genuine worry.",
    "frustrated": "Frustrated and exasperated, with noticeable impatience.",
    "angry": "Angry and forceful, with controlled intensity.",
    "sad": "Sad, subdued and emotionally restrained.",
    "disappointed": "Disappointed and slightly dejected, but controlled.",
    "surprised": "Genuinely surprised, with heightened energy and emphasis.",
    "confused": "Confused and uncertain, as though trying to understand what happened.",
    "curious": "Curious, engaged and inquisitive.",
    "skeptical": "Skeptical and doubtful, with a questioning tone.",
    "sarcastic": "Sarcastic and mocking, with a sharp edge.",
    "humorous": "Humorous and playful, with a light-hearted tone.",
    "witty": "Witty and clever, with a sharp sense of humor.",
    "dramatic": "Dramatic and intense, with strong emotional delivery.",
    "mysterious": "Mysterious and enigmatic, with a conspiratorial whisper.",
    "narrative": "Storytelling narrative, with clear pacing and emphasis.",
    "explainer": "Clear, explanatory, educational tone—like a teacher.",
    "whisper": "Soft and intimate, as if whispering to the listener.",
    "robotic": "Mechanical and flat, with artificial precision.",
    "urgent": "Urgent and urgent, racing against time.",
}


# A voice/tone tag, one of:
#   [voice: NAME | tone: TAG]
#   [voice: NAME]
#   [tone: TAG]
# Named groups: voice, tone (from combined), tone_only (from tone-only form).
_TAG = re.compile(
    r"\[voice:\s*(?P<voice>[^\]|]+?)(?:\s*\|\s*tone:\s*(?P<tone>\w+))?\s*\]"
    r"|\[tone:\s*(?P<tone_only>\w+)\s*\]"
)


def _strip_voice_tone_tags(text: str) -> str:
    """Remove ``[voice: ... | tone: ...]`` / ``[tone: ...]`` tags.

    The tags carry metadata that is consumed separately via the ``instruct``
    parameter; the raw tag text must never appear in the TTS audio.
    """
    if not text:
        return text
    stripped = _TAG.sub("", text)
    stripped = re.sub(r"[ \t]+\n", "\n", stripped)
    stripped = re.sub(r"\n{3,}", "\n\n", stripped)
    return stripped.strip()


def _tone_instruct(tone_key: Optional[str]) -> Optional[str]:
    """Return the full Voicebox instruction string for *tone_key*.

    Returns ``None`` when the tone is unknown or unspecified so callers can
    omit the ``instruct`` field entirely (Voicebox applies the profile default).
    """
    if tone_key is None or tone_key not in TONES:
        return None
    return TONES[tone_key]


def _parse_blocks(notes_text: str) -> List[dict]:
    """Split speaker notes into voice/tone blocks.

    Each returned dict has keys:
        - ``voice``  – the named voice (e.g. "Simone"), the previous value
          when a tag only changes tone, or None if never set.
        - ``tone``   – the tone keyword, the previous value, or None.
        - ``text``   – the narration text for this block (tags stripped).

    A ``[voice: ... | tone: ...]``, ``[voice: ...]``, or ``[tone: ...]`` tag
    begins a new block; everything after it -- including later paragraphs --
    belongs to that block until the next tag (or end of notes).  A tag that
    sets only voice or only tone leaves the other dimension sticky from the
    previous block.  Untagged narration produces a block with voice/tone None
    (backward compatible with the old behavior).
    """
    if not notes_text or not notes_text.strip():
        return []

    text = notes_text.strip()
    matches = list(_TAG.finditer(text))
    if not matches:
        return [{"voice": None, "tone": None, "text": _collapse_notes(text)}]

    blocks: List[dict] = []
    last_voice: Optional[str] = None
    last_tone: Optional[str] = None

    for idx, m in enumerate(matches):
        if idx == 0 and m.start() > 0:
            # Leading narration before the first tag -> its own default block.
            blocks.append(
                {"voice": None, "tone": None, "text": _collapse_notes(text[: m.start()])}
            )
        # Extract whichever fields this tag sets; keep the other sticky.
        voice = m.group("voice")
        tone = m.group("tone") or m.group("tone_only")
        if voice is not None:
            last_voice = voice.strip()
        if tone is not None:
            last_tone = tone.strip().lower()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        block_text = _collapse_notes(text[m.end() : end])
        if block_text:
            blocks.append({"voice": last_voice, "tone": last_tone, "text": block_text})

    return blocks


def _collapse_notes(text: str) -> str:
    """Collapse internal whitespace/newlines in a narration block, then strip."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


STAGE_DIRECTION_PATTERN = re.compile(r"\([^)]*\)")


def strip_stage_directions(text: str) -> str:
    """Remove parenthesized asides from speaker notes.

    Bracketed cues such as ``[laugh]`` are intentionally PRESERVED so the TTS
    engine can interpret them; only ``( ... )`` prose asides are stripped.
    """
    if not text:
        return text
    cleaned = STAGE_DIRECTION_PATTERN.sub(" ", text)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def prepare_narration(raw_notes: str, *, personality: bool = False) -> str:
    """Sanitize notes for Voicebox.

    For personality rewrite, keep natural phrasing so the profile LLM can
    restate the notes in character. For plain TTS, apply pronunciation tags.
    """
    if not raw_notes or not raw_notes.strip():
        return ""
    text = _strip_voice_tone_tags(raw_notes)
    text = strip_stage_directions(text)
    if not text:
        return ""
    if personality:
        text = _strip_control_chars(text.strip())
        text = _normalize_unicode_punctuation(text)
        return _ensure_space_after_period_before_capital(text).strip()
    return narration_plain_for_tts(text).strip()
