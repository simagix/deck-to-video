"""Split slide ranges for multi-part video output."""

from __future__ import annotations

from typing import List, Optional, Tuple


def parse_split_at(split_at: str) -> List[int]:
    """Parse a comma-separated list of 1-indexed split points."""
    points = [int(part.strip()) for part in split_at.split(",") if part.strip()]
    if not points:
        raise ValueError("--split-at requires at least one slide number")
    return sorted(set(points))


def compute_slide_ranges(
    slide_count: int,
    split_points: Optional[List[int]] = None,
) -> List[Tuple[int, int]]:
    """Return 0-indexed [start, end) ranges for each output video."""
    if slide_count < 1:
        raise ValueError("No slides to process")

    if not split_points:
        return [(0, slide_count)]

    split_points = sorted(int(point) for point in split_points)
    if split_points[0] < 1 or split_points[-1] > slide_count:
        raise ValueError(
            f"Split points must be between 1 and {slide_count} (slide numbers are 1-indexed)"
        )

    ranges: List[Tuple[int, int]] = []
    start = 0
    for split in split_points:
        if start < split:
            ranges.append((start, split))
        start = split
    if start < slide_count:
        ranges.append((start, slide_count))
    return ranges


def video_label_for_range(
    sanitized_title: str,
    start: int,
    end: int,
    slide_count: int,
    only_slide: Optional[int] = None,
) -> str:
    """Build an output filename stem for one video segment."""
    pad = max(2, len(str(slide_count)))
    if only_slide is not None and (end - start) == 1:
        return f"{sanitized_title}-slide-{only_slide:0{pad}d}"
    if end - start == slide_count:
        return sanitized_title
    return f"{sanitized_title}-slides-{start + 1:0{pad}d}-{end:0{pad}d}"
