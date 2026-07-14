"""Tier-2: turn surviving segments into structured MemoryRecords (CLAUDE.md §5)."""

from .embed import Embedder
from .ner import NerRunner
from .pipeline import Tier2Result, Tier2Runners, process_segment, process_segments
from .summarize import extractive_summary

__all__ = [
    "process_segments",
    "process_segment",
    "Tier2Runners",
    "Tier2Result",
    "Embedder",
    "NerRunner",
    "extractive_summary",
]
