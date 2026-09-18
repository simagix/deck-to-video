"""Narration text normalization and pronunciation helpers."""

from __future__ import annotations

import re
import unicodedata
from typing import AbstractSet, List, Optional, Tuple


def _strip_control_chars(text: str) -> str:
    if not text:
        return text
    return "".join(c for c in text if c == "\n" or c == "\r" or c == "\t" or ord(c) >= 32)


# ---------------------------------------------------------------------------
# Language detection (stdlib only — per-slide auto-detect for EN/ZH)
# ---------------------------------------------------------------------------

# CJK ranges: CJK Unified Ideographs (+ extensions), compatibility ideographs,
# CJK symbols/punctuation, and fullwidth ASCII variants.
_CJK_RE = re.compile(
    "[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
    "\U00020000-\U0002a6df\U0002a600-\U0002b73f\U0002b740-\U0002b81f"
    "\u3000-\u303f\uff00-\uffef]"
)
_LATIN_RE = re.compile(r"[A-Za-z]")

#: CJK punctuation that must survive _normalize_unicode_punctuation().
#: Without these, Mandarin narration loses its sentence boundaries
#: (。，、；：？！) and Qwen3-TTS prosody degrades.
_CJK_PUNCT_KEEP = frozenset("，。、；：？！「」『』（）【】《》〈〉…—·・")


def cjk_ratio(text: str) -> float:
    """Fraction of CJK vs (CJK + Latin) characters in *text* (0.0–1.0).

    Returns 0.0 for empty / script-neutral text (digits, pure punctuation).
    """
    if not text:
        return 0.0
    cjk = len(_CJK_RE.findall(text))
    latin = len(_LATIN_RE.findall(text))
    if cjk + latin == 0:
        return 0.0
    return cjk / (cjk + latin)


def detect_language(text: str) -> str:
    """Detect narration language: ``"zh"`` or ``"en"``.

    A slide counts as Mandarin when CJK characters are a significant share
    (>20%) of its script-bearing characters — so a mostly-English slide that
    merely mentions ``宫保鸡丁`` stays English, while a bilingual slide with
    real Mandarin content switches to ``"zh"`` (verified: Qwen3-TTS renders
    English words inside ``lang_code="chinese"`` cleanly, but mangles
    Mandarin rendered as ``"english"``).
    """
    if not text or not text.strip():
        return "en"
    return "zh" if cjk_ratio(text) > 0.2 else "en"


def resolve_narration_language(text: str, override: Optional[str] = None) -> str:
    """Resolve the effective per-slide language (``"zh"`` / ``"en"``).

    ``override`` is a deck-level ``--language`` value: ``"en"`` / ``"zh"``
    force every slide; ``None`` / ``"auto"`` auto-detect per slide.
    """
    if override is not None and override.strip().lower() in ("en", "zh"):
        return override.strip().lower()
    return detect_language(text)


#: Deck-to-Voicebox language codes (Voicebox /generate ``language`` field).
VOICEBOX_LANGUAGE_CODES = ("en", "zh")

#: Deck-to-mlx-audio language codes (mlx-audio ``lang_code`` for Qwen3-TTS).
MLX_LANGUAGE_CODES = ("auto", "chinese", "english")


def voicebox_language(lang: str) -> str:
    """Map a narration language (``"zh"``/``"en"``) to a Voicebox code."""
    return "zh" if lang.strip().lower() in ("zh", "chinese") else "en"


def mlx_language(lang: str) -> str:
    """Map a narration language (``"zh"``/``"en"``) to an mlx-audio code."""
    return "chinese" if lang.strip().lower() in ("zh", "chinese") else "english"


def parse_language_override(value: Optional[str]) -> str:
    """Normalize a ``--language`` / env value to ``"auto"``/``"en"``/``"zh"``."""
    if value is None:
        return "auto"
    normalized = value.strip().lower()
    if normalized in ("zh", "chinese", "cn", "mandarin"):
        return "zh"
    if normalized in ("en", "english"):
        return "en"
    return "auto"


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
        elif char in _CJK_PUNCT_KEEP:
            # CJK punctuation carries sentence boundaries for Mandarin TTS —
            # never strip it (the old code mapped it to a space).
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

# A punchline sound-effect tag, e.g. "[sfx: rimshot]" — plus bare aliases
# like "[badumtss]". Stripped from TTS text; resolved by ``parse_sfx_cues``.
_SFX_TAG = re.compile(
    r"\[\s*sfx:\s*(?P<name>[A-Za-z0-9_\- ]+?)\s*\]"
    r"|\[\s*(?P<alias>badumtss|rimshot|ba[\s_\-]?dum[\s_\-]?tss"
    r"|sad[\s_\-]?trombone|wah[\s_\-]?wah[\s_\-]?wah"
    r"|drum[\s_\-]?roll)\s*\]",
    re.IGNORECASE,
)

#: Canonical sound-effect names; tag spellings normalize to these keys.
#: (Alias keys are separator-free because ``_canonical_sfx_name`` strips
#: spaces/underscores/hyphens before lookup.)
SFX_ALIASES = {
    "rimshot": "rimshot",
    "badumtss": "rimshot",
    "sadtrombone": "sad_trombone",
    "wahwahwah": "sad_trombone",
    "drumroll": "drum_roll",
    "chaching": "cha_ching",
    "cha_ching": "cha_ching",
    "door_slam": "door_slam",
}

# A background-music tag: ``[bgm: TRACK]`` with optional trailing spec fields
# separated by pipes, e.g. ``[bgm: ambient_loop | volume: 0.2]``. Field
# separators deliberately exclude newlines (spaces/tabs only) so a stray
# ``[bgm:`` in prose never swallows the following line as a track name; the
# first field is always the reference, further fields are parsed leniently
# (``volume`` honored; anything else ignored for forward compatibility) and
# the whole tag is stripped from TTS text by ``_strip_bgm_tags``.
_BGM_TAG = re.compile(
    r"\[\s*bgm:[ \t]*(?P<ref>[^|\]\r\n]+)"
    r"(?P<spec>(?:[ \t]*\|[ \t]*[^|\]\r\n]*)*)"
    r"[ \t]*\]",
    re.IGNORECASE,
)

#: Volume override inside a ``[bgm: ...]`` spec (e.g. ``| volume: 0.3``).
_BGM_VOLUME = re.compile(r"\bvolume\b\s*[:=]?\s*(\d+(?:\.\d+)?)", re.IGNORECASE)


def _bgm_volume_from_spec(spec: str) -> Optional[float]:
    """Extract an optional positive volume from a BGM tag's trailing spec."""
    match = _BGM_VOLUME.search(spec or "")
    if not match:
        return None
    return float(match.group(1))


def parse_bgm_cues(notes_text: str) -> List[Tuple[str, Optional[float]]]:
    """Return ``(track_ref, volume_or_None)`` for each ``[bgm: ...]`` tag.

    Order-preserving and duplicate-keeping so callers decide precedence; the
    deck pipeline uses only the FIRST cue across all slides, because
    background music spans slide boundaries at video-assembly time rather
    than being baked into per-slide WAVs like punchline SFX.
    """
    cues: List[Tuple[str, Optional[float]]] = []
    for match in _BGM_TAG.finditer(notes_text or ""):
        ref = match.group("ref").strip()
        if ref:
            cues.append((ref, _bgm_volume_from_spec(match.group("spec"))))
    return cues


def _strip_bgm_tags(text: str) -> str:
    """Remove ``[bgm: ...]`` tags.

    Background music is consumed at video-assembly time (see ``bgm.py``);
    the raw tag text must never reach the TTS.
    """
    if not text:
        return text
    stripped = _BGM_TAG.sub("", text)
    stripped = re.sub(r"[ \t]+\n", "\n", stripped)
    stripped = re.sub(r"[ \t]{2,}", " ", stripped)
    return stripped.strip()


def _canonical_sfx_name(raw: str) -> Optional[str]:
    """Map a tag spelling ('Ba Dum Tss', 'rim-shot', …) to its canonical name."""
    return SFX_ALIASES.get(re.sub(r"[\s_\-]+", "", raw.lower()))


def _strip_sfx_tags(text: str) -> str:
    """Remove ``[sfx: ...]`` / ``[badumtss]`` punchline effect tags.

    The tags trigger sound-effect mixing at voiceover-generation time (see
    ``sfx.overlay_sfx_on_wav``); the raw tag text must never reach the TTS.
    """
    if not text:
        return text
    stripped = _SFX_TAG.sub("", text)
    stripped = re.sub(r"[ \t]+\n", "\n", stripped)
    stripped = re.sub(r"[ \t]{2,}", " ", stripped)
    return stripped.strip()


def parse_sfx_cues(notes_text: str) -> List[str]:
    """Return recognized sound effects requested by *notes_text*, in order.

    Recognizes ``[sfx: NAME]`` plus bare aliases such as ``[badumtss]``
    or ``[ba dum tss]``. Unknown names are ignored (but still stripped from
    TTS text), so typos never leak into narration.
    """
    cues: List[str] = []
    for match in _SFX_TAG.finditer(notes_text or ""):
        raw = match.group("name")
        if raw is None:
            raw = match.group("alias")
        canonical = _canonical_sfx_name(raw)
        if canonical and canonical not in cues:
            cues.append(canonical)
    return cues


def split_notes_on_sfx(
    notes_text: str,
    supported: Optional[AbstractSet[str]] = None,
) -> List[dict]:
    """Split speaker notes into narration takes interleaved with SFX cues.

    Recognized sound-effect tags act as split points; everything between two
    tags (including any ``[voice: ... | tone: ...]`` markers) stays together
    as one narration take, so each take can carry its own tone instruct.

    Returns an ordered list of dicts:
        - ``{"kind": "narration", "source": <raw notes substring>}``
        - ``{"kind": "sfx", "name": <canonical sfx name>}``

    Unrecognized SFX names are left inside the surrounding take (they are
    stripped from the TTS text later); recognized ones always split.
    Notes without any recognized cue yield a single narration part covering
    the whole text (backward compatible).
    """
    if not notes_text or not notes_text.strip():
        return []

    allowed = set(supported) if supported is not None else set(SFX_ALIASES.values())

    # Collect split positions for recognized SFX cues only; voice/tone tags
    # stay embedded within their take's raw source text.
    splits: List[Tuple[int, int, str]] = []
    for match in _SFX_TAG.finditer(notes_text):
        raw = match.group("name")
        if raw is None:
            raw = match.group("alias")
        canonical = _canonical_sfx_name(raw)
        if canonical in allowed:
            splits.append((match.start(), match.end(), canonical))

    if not splits:
        return [{"kind": "narration", "source": notes_text}]

    parts: List[dict] = []
    cursor = 0
    for start, end, name in splits:
        source = notes_text[cursor:start]
        if source.strip():
            parts.append({"kind": "narration", "source": source})
        parts.append({"kind": "sfx", "name": name})
        cursor = end
    tail = notes_text[cursor:]
    if tail.strip():
        parts.append({"kind": "narration", "source": tail})
    return parts


def split_notes_on_tone(notes_text: str) -> List[dict]:
    """Split speaker notes into narration segments at tone-change boundaries.

    Each recognized ``[tone: TAG]`` or ``[voice: NAME | tone: TAG]`` tag acts
    as a split point; everything before it becomes one narration segment with
    the preceding tone, everything after becomes the next segment.

    Returns an ordered list of dicts:
        - ``{"kind": "narration", "source": <raw notes substring>}``

    Notes without any tone tag yield a single narration part covering the
    whole text (backward compatible). This lets the single-take path still
    produce separate audio per tone — the same way SFX already does.
    """
    if not notes_text or not notes_text.strip():
        return []

    splits: List[Tuple[int, int]] = []
    for match in _TAG.finditer(notes_text):
        tone = match.group("tone") or match.group("tone_only")
        if tone:
            splits.append((match.start(), match.end()))

    if not splits:
        return [{"kind": "narration", "source": notes_text}]

    parts: List[dict] = []
    cursor = 0
    for start, end in splits:
        source = notes_text[cursor:start]
        if source.strip():
            parts.append({"kind": "narration", "source": source})
        cursor = end
    tail = notes_text[cursor:]
    if tail.strip():
        parts.append({"kind": "narration", "source": tail})
    return parts


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


def _first_tone_instruct(source: str) -> Optional[str]:
    """First usable Voicebox tone instruct found in *source* notes, else None.

    A tone-only tag such as ``[tone: frustrated]`` may appear after leading
    narration, so it can live in a later block.
    """
    for block in _parse_blocks(source):
        if block.get("tone"):
            instruct = _tone_instruct(block["tone"])
            if instruct:
                return instruct
    return None


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
    text = _strip_sfx_tags(text)
    text = _strip_bgm_tags(text)
    text = strip_stage_directions(text)
    if not text:
        return ""
    if personality:
        text = _strip_control_chars(text.strip())
        text = _normalize_unicode_punctuation(text)
        return _ensure_space_after_period_before_capital(text).strip()
    return narration_plain_for_tts(text).strip()
