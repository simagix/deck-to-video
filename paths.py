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
DEFAULT_FPS = 24
TARGET_IMAGE_SIZE = (1920, 1080)
