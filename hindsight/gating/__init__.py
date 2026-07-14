"""Tier-1 gating: the four-state hysteresis FSM + the segmenter."""

from .hysteresis import ActiveSpan, HysteresisParams, run_gate
from .segmenter import GateResult, close_or_merge, segment_spans

__all__ = [
    "ActiveSpan",
    "HysteresisParams",
    "run_gate",
    "GateResult",
    "segment_spans",
    "close_or_merge",
]
