"""Capture sources. Everything downstream of here is source-agnostic (CLAUDE.md §2).

If the team switches hardware, exactly one file in this package changes. No `gopro`/`meta`
conditional may leak past this package.
"""

from __future__ import annotations

from pathlib import Path

from .clip import ClipMeta, ClipSource
from .gpmf import GpmfSource, ImuSample
from .synthetic import InterestSpan, SyntheticSource, demo_source
from .webcam import WebcamSource

__all__ = [
    "ClipSource",
    "ClipMeta",
    "SyntheticSource",
    "InterestSpan",
    "demo_source",
    "WebcamSource",
    "GpmfSource",
    "ImuSample",
    "open_source",
]

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}


def open_source(spec: str, **kwargs):
    """Factory. `spec` is a file path (-> ClipSource), 'webcam', or 'synthetic'."""
    if spec == "webcam":
        return WebcamSource(**kwargs)
    if spec == "synthetic":
        return demo_source(**kwargs) if not kwargs else SyntheticSource(**kwargs)
    p = Path(spec)
    if p.suffix.lower() in VIDEO_SUFFIXES or p.exists():
        return ClipSource(p, **kwargs)
    raise ValueError(f"cannot open source {spec!r}: not a video file, 'webcam', or 'synthetic'")
