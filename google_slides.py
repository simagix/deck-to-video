"""Google Slides export: PNG thumbnails and speaker notes."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import requests  # type: ignore[import-untyped]
from google.auth.credentials import Credentials as GoogleCredentials  # type: ignore[import-untyped]
from google.auth.transport.requests import Request  # type: ignore[import-untyped]
from google.oauth2.credentials import Credentials  # type: ignore[import-untyped]
from google_auth_oauthlib.flow import InstalledAppFlow  # type: ignore[import-untyped]
from googleapiclient.discovery import build  # type: ignore[import-untyped]
from googleapiclient.errors import HttpError  # type: ignore[import-untyped]

from images import cleanup_old_assets, resize_slide_png
from paths import CREDENTIALS_PATH, TOKEN_PATH

SCOPES = [
    "https://www.googleapis.com/auth/drive",
]

PRESENTATION_ID_FROM_URL = re.compile(r"/presentation/d/([a-zA-Z0-9_-]+)")
FILE_ID_FROM_DRIVE_OPEN = re.compile(r"[?&]id=([a-zA-Z0-9_-]+)")
FILE_ID_FROM_DRIVE_FILE = re.compile(r"/file/d/([a-zA-Z0-9_-]+)")


def extract_presentation_id(url_or_id: str) -> str:
    """Extract a Google Slides presentation ID from a URL or raw ID string."""
    value = (url_or_id or "").strip()
    if not value:
        raise ValueError("Empty presentation id/url")

    for pattern in (PRESENTATION_ID_FROM_URL, FILE_ID_FROM_DRIVE_OPEN, FILE_ID_FROM_DRIVE_FILE):
        match = pattern.search(value)
        if match:
            return match.group(1)

    if re.match(r"^[a-zA-Z0-9_-]+$", value):
        return value

    raise ValueError(f"Could not extract presentation ID from: {url_or_id}")


def get_service(
    api_name: str,
    version: str,
    creds_path: str = CREDENTIALS_PATH,
    token_path: str = TOKEN_PATH,
):
    """Authenticate and return a Google API service client."""
    creds: Optional[GoogleCredentials] = None
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
            creds = flow.run_local_server(port=0)
        if creds is None:
            raise RuntimeError("Failed to obtain credentials")
        with open(token_path, "w", encoding="utf-8") as token_file:
            token_file.write(creds.to_json())
    if creds is None:
        raise RuntimeError("Credentials not available")
    return build(api_name, version, credentials=creds)


def check_document_type(doc_id: str) -> Tuple[bool, str]:
    """Return (is_slides, doc_type_name) for a Google Drive file ID."""
    try:
        drive_service = get_service("drive", "v3")
        file_metadata = drive_service.files().get(  # type: ignore[attr-defined]
            fileId=doc_id.strip(),
            fields="id, mimeType",
            supportsAllDrives=True,
        ).execute()
    except HttpError as exc:
        error_code = exc.resp.status if hasattr(exc, "resp") else None
        if error_code == 404:
            raise ValueError(
                f"Document not found. The ID '{doc_id}' does not exist or you don't have access to it."
            ) from exc
        if error_code == 403:
            raise ValueError(
                f"Access denied. You don't have permission to view this document (ID: '{doc_id}')."
            ) from exc
        if error_code == 400:
            raise ValueError(f"Invalid document ID format: '{doc_id}'") from exc
        raise RuntimeError(f"Google API error ({error_code}): {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to check document type: {exc}") from exc

    mime_type = file_metadata.get("mimeType", "")
    mime_type_map = {
        "application/vnd.google-apps.document": "Google Doc",
        "application/vnd.google-apps.presentation": "Google Slides",
        "application/vnd.google-apps.spreadsheet": "Google Sheets",
        "application/vnd.google-apps.folder": "Folder",
        "application/vnd.google-apps.form": "Google Form",
        "application/vnd.google-apps.drawing": "Google Drawing",
    }
    doc_type_name = mime_type_map.get(mime_type, mime_type or "Unknown")
    is_slides = mime_type == "application/vnd.google-apps.presentation"
    return is_slides, doc_type_name


def get_presentation_title(presentation_id: str) -> str:
    slides_service = get_service("slides", "v1")
    presentation = slides_service.presentations().get(  # type: ignore[attr-defined]
        presentationId=presentation_id
    ).execute()
    return presentation.get("title", "Slides Import")


def get_skipped_slide_indices(pres_id: str) -> set:
    """Return 0-based indices of slides marked as skipped in Google Slides."""
    slides_service = get_service("slides", "v1")
    presentation_data = slides_service.presentations().get(  # type: ignore[attr-defined]
        presentationId=pres_id
    ).execute()
    slides = presentation_data.get("slides", [])

    skipped_indices = set()
    for index, slide in enumerate(slides):
        slide_properties = slide.get("slideProperties", {})
        if slide_properties.get("isSkipped", False):
            skipped_indices.add(index)
            print(f"⏭️  Slide {index + 1} is marked as skipped")

    if skipped_indices:
        print(
            f"📋 Found {len(skipped_indices)} skipped slide(s): "
            f"{sorted(index + 1 for index in skipped_indices)}"
        )
    else:
        print("📋 No slides are marked as skipped")
    return skipped_indices


def export_slides_to_png(
    pres_id: str,
    output_dir: str,
    skipped_indices: Optional[set] = None,
    only_slide: Optional[int] = None,
) -> int:
    """Export Google Slides to PNG files. Returns number of images saved."""
    os.makedirs(output_dir, exist_ok=True)
    if skipped_indices is None:
        skipped_indices = set()

    slides_service = get_service("slides", "v1")
    presentation_data = slides_service.presentations().get(  # type: ignore[attr-defined]
        presentationId=pres_id
    ).execute()
    slides = presentation_data.get("slides", [])
    total_slides = len(slides)
    active_slides = total_slides - len(skipped_indices)
    print(f"Found {total_slides} slides ({active_slides} active, {len(skipped_indices)} skipped)")

    saved_images = 0
    reused_images = 0
    export_index = 0
    for index, slide in enumerate(slides):
        if only_slide is not None and (index + 1) != only_slide:
            continue
        if index in skipped_indices:
            print(f"⏭️  Skipping slide {index + 1} (marked as skipped)")
            continue

        export_index += 1
        fname = os.path.join(output_dir, f"slide_{export_index:02d}.png")
        if os.path.isfile(fname):
            print(f"   ✅ Slide {index + 1}: using existing {os.path.basename(fname)}")
            reused_images += 1
            continue

        slide_id = slide.get("objectId")
        try:
            thumb = slides_service.presentations().pages().getThumbnail(  # type: ignore[attr-defined]
                presentationId=pres_id,
                pageObjectId=slide_id,
                thumbnailProperties_mimeType="PNG",
                thumbnailProperties_thumbnailSize="LARGE",
            ).execute()
            url = thumb.get("contentUrl")
            if not url:
                print(f"⚠️  No contentUrl for slide {index + 1}")
                continue
            slide_resp = requests.get(url, timeout=30)
            slide_resp.raise_for_status()
            with open(fname, "wb") as img_file:
                img_file.write(slide_resp.content)
            resize_slide_png(fname)
            saved_images += 1
        except (requests.RequestException, OSError, KeyError) as exc:
            print(f"⚠️  Failed to export slide {index + 1}: {exc}")

    if reused_images:
        print(
            f"Reused {reused_images} existing slide image(s); "
            f"exported {saved_images} new image(s) to {output_dir}"
        )
    else:
        print(f"Exported {saved_images} slide image(s) to {output_dir}")
    return saved_images + reused_images


def _extract_text_from_shape(shape: Dict[str, Any]) -> str:
    if not shape:
        return ""
    text_obj = shape.get("text", {})
    if not isinstance(text_obj, dict):
        return ""
    parts: List[str] = []
    for text_element in text_obj.get("textElements", []) or []:
        text_run = text_element.get("textRun", {})
        if isinstance(text_run, dict):
            content = text_run.get("content", "")
            if content:
                parts.append(content)
    return "".join(parts)


def _extract_speaker_notes_from_notes_page(notes_page: Dict[str, Any]) -> str:
    if not notes_page:
        return ""

    speaker_notes_oid = None
    notes_props = notes_page.get("notesProperties", {})
    if isinstance(notes_props, dict):
        speaker_notes_oid = notes_props.get("speakerNotesObjectId")

    page_elements = notes_page.get("pageElements", []) or []
    if not isinstance(page_elements, list):
        page_elements = []

    if not speaker_notes_oid:
        for element in page_elements:
            placeholder = (element.get("shape", {}) or {}).get("placeholder", {}) or {}
            if placeholder.get("type") == "BODY":
                speaker_notes_oid = element.get("objectId")
                break

    if speaker_notes_oid:
        for element in page_elements:
            if element.get("objectId") == speaker_notes_oid:
                shape = element.get("shape", {}) or {}
                text = _extract_text_from_shape(shape)
                return text.replace("\r\n", "\n").replace("\r", "\n").strip()

    best_text = ""
    for element in page_elements:
        shape = element.get("shape", {}) or {}
        if not shape:
            continue
        candidate = _extract_text_from_shape(shape).replace("\r\n", "\n").replace("\r", "\n").strip()
        if len(candidate) > len(best_text):
            best_text = candidate
    return best_text


def export_speaker_notes(
    pres_id: str,
    output_dir: str,
    skipped_indices: Optional[set] = None,
    only_slide: Optional[int] = None,
) -> List[str]:
    """Export speaker notes; returns one string per exported slide."""
    if skipped_indices is None:
        skipped_indices = set()

    cleanup_old_assets(output_dir, "_notes.txt")

    slides_service = get_service("slides", "v1")
    presentation_data = slides_service.presentations().get(  # type: ignore[attr-defined]
        presentationId=pres_id
    ).execute()
    slides = presentation_data.get("slides", [])

    notes_file = os.path.join(output_dir, "notes_all.txt")
    notes_lines: List[str] = []
    notes_per_slide_list: List[str] = []

    export_index = 0
    for slide_idx, slide in enumerate(slides):
        slide_num = slide_idx + 1
        if only_slide is not None and slide_num != only_slide:
            continue
        if slide_idx in skipped_indices:
            print(f"\n⏭️  Slide {slide_num}: Skipping (marked as skipped)")
            continue

        export_index += 1
        try:
            slide_properties = slide.get("slideProperties", {}) or {}
            notes_page = slide_properties.get("notesPage", {}) or {}
            notes_text = _extract_speaker_notes_from_notes_page(notes_page)
            notes_per_slide_list.append(notes_text)

            if notes_text:
                notes_lines.append(f"=== Slide {slide_num} (export #{export_index}) ===")
                notes_lines.append(notes_text)
                notes_lines.append("")
                print(f"✅ Found notes for slide {slide_num} ({len(notes_text)} chars)")
            else:
                notes_lines.append(f"=== Slide {slide_num} (export #{export_index}) ===")
                notes_lines.append("(No notes)")
                notes_lines.append("")
                print(f"⚠️  No notes found for slide {slide_num}")
        except (requests.RequestException, OSError, KeyError, AttributeError) as exc:
            print(f"⚠️  Failed to get notes for slide {slide_num}: {exc}")
            notes_per_slide_list.append("")
            notes_lines.append(f"=== Slide {slide_num} (export #{export_index}) ===")
            notes_lines.append(f"(Error: {exc})")
            notes_lines.append("")

    if notes_lines:
        with open(notes_file, "w", encoding="utf-8") as notes_file_handle:
            notes_file_handle.write("\n".join(notes_lines))
        print(f"\n✅ Exported all notes to: {notes_file}")

    for file_idx, notes_text in enumerate(notes_per_slide_list, start=1):
        note_path = os.path.join(output_dir, f"slide_{file_idx:02d}_notes.txt")
        with open(note_path, "w", encoding="utf-8") as note_file_handle:
            note_file_handle.write(notes_text)

    return notes_per_slide_list
