"""Shared paths and constants for the synth package."""

from __future__ import annotations

import os

SYNTH_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_BASE_DIR = os.path.join(SYNTH_DIR, "out")
CREDENTIALS_PATH = os.path.join(SYNTH_DIR, "credentials.json")
TOKEN_PATH = os.path.join(SYNTH_DIR, "token.json")
ENV_PATH = os.path.join(SYNTH_DIR, ".env")

DEFAULT_VOICEBOX_URL = "http://127.0.0.1:17493"
DEFAULT_SILENT_SLIDE_SECONDS = 3.0
DEFAULT_INTER_SLIDE_PAUSE_SECONDS = 1.0
DEFAULT_VOICEOVER_TRAIL_SILENCE_SECONDS = 1.0
# Punchline sound effects ([sfx: rimshot]): comedic beat between the last
# spoken word and the hit, and how loud the sample plays relative to speech.
DEFAULT_SFX_BEAT_SECONDS = 0.25
DEFAULT_SFX_GAIN = 0.9
# Background music ([bgm: track.mp3]): how loud the music plays relative to
# the voiceover. Applied via MoviePy CompositeAudioClip in video_assembly.py.
DEFAULT_BG_MUSIC_VOLUME = 0.15
DEFAULT_FPS = 24
DEFAULT_KEN_BURNS_ZOOM = 1.15
TARGET_IMAGE_SIZE = (1920, 1080)
KEN_BURNS_IMAGE_SIZE = (1824, 1024)  # Slightly smaller than target to allow for panning
