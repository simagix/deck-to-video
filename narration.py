"""Narration text normalization and pronunciation helpers."""

from __future__ import annotations

import re
import unicodedata
from typing import List


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
    text = strip_stage_directions(raw_notes)
    if not text:
        return ""
    if personality:
        text = _strip_control_chars(text.strip())
        text = _normalize_unicode_punctuation(text)
        return _ensure_space_after_period_before_capital(text).strip()
    return narration_plain_for_tts(text).strip()
