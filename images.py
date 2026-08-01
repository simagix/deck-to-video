"""Image helpers for slide PNG export."""

from __future__ import annotations

import os
from typing import Tuple

from paths import TARGET_IMAGE_SIZE

try:
    from PIL import Image  # type: ignore[import-untyped]

    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def resize_slide_png(path: str, target_size: Tuple[int, int] = TARGET_IMAGE_SIZE) -> bool:
    """Resize a slide PNG to target_size, letterboxed on black. Returns True on success."""
    if not HAS_PIL:
        print("⚠️  Pillow not installed; image saved at original size.")
        return False

    try:
        img = Image.open(path)  # type: ignore[possibly-unbound]
        original_size = img.size
        scale = min(target_size[0] / original_size[0], target_size[1] / original_size[1])
        new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
        resized_img = img.resize(new_size, Image.Resampling.LANCZOS)  # type: ignore[possibly-unbound]
        new_img = Image.new("RGB", target_size, (0, 0, 0))  # type: ignore[possibly-unbound]
        x_offset = (target_size[0] - new_size[0]) // 2
        y_offset = (target_size[1] - new_size[1]) // 2
        new_img.paste(resized_img, (x_offset, y_offset))
        new_img.save(path, "PNG", optimize=True)
        print(
            f"✅ Saved and resized to {target_size[0]}x{target_size[1]}: {path} "
            f"(from {original_size[0]}x{original_size[1]})"
        )
        return True
    except (OSError, ValueError) as exc:
        print(f"⚠️  Failed to resize {path}: {exc}")
        return False


def cleanup_old_assets(output_dir: str, suffix: str) -> None:
    """Remove prior slide_XX{suffix} files in output_dir."""
    if not os.path.isdir(output_dir):
        return
    prefix = "slide_"
    old_files = [
        name
        for name in os.listdir(output_dir)
        if name.startswith(prefix) and name.endswith(suffix)
    ]
    if not old_files:
        return
    print(f"🧹 Cleaning up {len(old_files)} old file(s) matching *{suffix}...")
    for name in old_files:
        try:
            os.remove(os.path.join(output_dir, name))
        except OSError as exc:
            print(f"⚠️  Failed to remove {name}: {exc}")
