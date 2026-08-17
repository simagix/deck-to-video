"""Unit tests for voicebox_client payload construction.

The point of these tests is to make observable exactly what text is sent to the
Voicebox ``/generate`` endpoint, so bugs like "[laugh]" being silently stripped
before TTS can be isolated to either this client or the Voicebox server.
"""

from __future__ import annotations

import base64
import json
import os
import tempfile
import unittest
from unittest import mock

from narration import prepare_narration
import voicebox_client

# Raw speaker notes as extracted from the deck (pre-narration).
RAW_NOTES = "Simone \u2026 [laugh] \u2026 Owt!"  # "Simone … [laugh] … Owt!"
PROFILE_ID = "00000000-0000-0000-0000-000000000000"
API_BASE = "http://127.0.0.1:17493"
FAKE_WAV = b"RIFF\x00\x00\x00\x00WAVEfmt mock-sample-data"


def _json_response(payload: dict, ok: bool = True, status_code: int = 200) -> mock.Mock:
    resp = mock.Mock()
    resp.ok = ok
    resp.status_code = status_code
    resp.text = ""
    resp.json.return_value = payload
    resp.content = b""
    return resp


class _GenerateRequestRecorder:
    """Routes HTTP calls and records the JSON posted to /generate."""

    def __init__(self, personality_profile: bool = True):
        self.calls: list[tuple[str, str, dict | None]] = []  # (method, url, json_body)
        self.personality_profile = personality_profile

    def get(self, url: str, **kwargs):
        self.calls.append(("GET", url, kwargs.get("json")))
        if "/settings/captures" in url:
            return _json_response({"llm_model": "0.6B"})
        if "/profiles/" in url:
            personality = "quirky demo narrator" if self.personality_profile else ""
            return _json_response({"personality": personality})
        if "/models/status" in url:
            return _json_response(
                {
                    "models": [
                        {
                            "model_name": "qwen3-0.6b",
                            "downloaded": True,
                            "loaded": True,
                        }
                    ]
                }
            )
        return _json_response({})

    def post(self, url: str, **kwargs):
        body = kwargs.get("json")
        self.calls.append(("POST", url, body))
        if url.rstrip("/").endswith("/generate"):
            # Return audio inline so no follow-up /audio/{id} fetch is needed.
            return _json_response({"audio_base64": base64.b64encode(FAKE_WAV).decode()})
        # e.g. the /llm/generate warmup call.
        return _json_response({})

    def generate_payload(self) -> dict:
        for method, url, body in self.calls:
            if method == "POST" and url.rstrip("/").endswith("/generate"):
                assert body is not None
                return body
        raise AssertionError(f"no POST /generate recorded; calls were: {self.calls}")


class GenerateVoiceboxPayloadTests(unittest.TestCase):
    def _run_generate(self, personality: bool, text: str = RAW_NOTES) -> _GenerateRequestRecorder:
        recorder = _GenerateRequestRecorder()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_wav = tmp.name
        try:
            with (
                mock.patch.object(voicebox_client.requests, "get", side_effect=recorder.get),
                mock.patch.object(voicebox_client.requests, "post", side_effect=recorder.post),
            ):
                voicebox_client.generate_voicebox_audio(
                    text=text,
                    profile_id=PROFILE_ID,
                    output_wav=out_wav,
                    api_base=API_BASE,
                    personality=personality,
                )
        finally:
            if os.path.exists(out_wav):
                os.remove(out_wav)
        return recorder

    def test_plain_tts_preserves_marked_up_text_verbatim(self):
        """Proves voicebox_client is NOT the stripper: it posts text unchanged."""
        recorder = self._run_generate(personality=False, text=RAW_NOTES)
        payload = recorder.generate_payload()

        print("\n>>> JSON sent to Voicebox /generate:\n", json.dumps(payload, indent=2))
        print(">>> repr(payload['text']):", repr(payload["text"]))

        self.assertEqual(payload["text"], RAW_NOTES)
        self.assertIn("[laugh]", payload["text"])
        self.assertEqual(payload["profile_id"], PROFILE_ID)
        self.assertEqual(payload["language"], "en")
        self.assertNotIn("personality", payload)

    def test_personality_mode_still_sends_original_text(self):
        """Even with personality on, the app sends the raw text + a flag."""
        recorder = self._run_generate(personality=True, text=RAW_NOTES)
        payload = recorder.generate_payload()

        print("\n>>> JSON sent to Voicebox /generate (personality):\n", json.dumps(payload, indent=2))
        print(">>> repr(payload['text']):", repr(payload["text"]))

        self.assertEqual(payload["text"], RAW_NOTES)
        self.assertIn("[laugh]", payload["text"])
        self.assertTrue(payload["personality"])

    def test_end_to_end_payload_preserves_laugh(self):
        """The REAL pipeline: raw notes run through prepare_narration first;
        [laugh] must now SURVIVE and arrive in the /generate payload."""
        for personality in (False, True):
            with self.subTest(personality=personality):
                narration_text = prepare_narration(RAW_NOTES, personality=personality)
                recorder = self._run_generate(personality=personality, text=narration_text)
                payload = recorder.generate_payload()
                print(
                    f"\n>>> RAW notes           : {RAW_NOTES!r}\n"
                    f">>> after narration     : {narration_text!r}\n"
                    f">>> text sent to Voicebox: {payload['text']!r}"
                )
                self.assertEqual(payload["text"], narration_text)
                self.assertIn("[laugh]", payload["text"])

    def test_engine_included_when_set(self):
        """When an engine is supplied it must be forwarded to /generate."""
        recorder = _GenerateRequestRecorder()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_wav = tmp.name
        try:
            with (
                mock.patch.object(voicebox_client.requests, "get", side_effect=recorder.get),
                mock.patch.object(voicebox_client.requests, "post", side_effect=recorder.post),
            ):
                voicebox_client.generate_voicebox_audio(
                    text=RAW_NOTES,
                    profile_id=PROFILE_ID,
                    output_wav=out_wav,
                    api_base=API_BASE,
                    engine="qwen",
                )
        finally:
            if os.path.exists(out_wav):
                os.remove(out_wav)
        payload = recorder.generate_payload()
        self.assertEqual(payload["engine"], "qwen")

    def test_engine_omitted_when_unset(self):
        """When no engine is set, the payload must NOT include an engine key
        (Voicebox then uses the profile's Default Engine)."""
        recorder = self._run_generate(personality=False, text=RAW_NOTES)
        payload = recorder.generate_payload()
        self.assertNotIn("engine", payload)

    def test_unsupported_engine_raises(self):
        """An unknown engine should fail fast with a clear message."""
        with self.assertRaises(ValueError):
            voicebox_client.generate_voicebox_audio(
                text=RAW_NOTES,
                profile_id=PROFILE_ID,
                output_wav="/tmp/unused.wav",
                api_base=API_BASE,
                engine="not-a-real-engine",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)