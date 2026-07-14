"""Segmenter: ACTIVE spans -> retained Segments, with the structural invariants enforced.

Pipeline (all params from config): min ACTIVE duration filter -> pre/post-roll -> merge-gap
-> max-duration cap. Then the invariants CLAUDE.md §3.3 and BUILD_BRIEF M3 require *in code*:

  * no retained segment shorter than 6.5 s (asserted in contracts.Segment.__post_init__),
  * no retained segment longer than 60 s,
  * reduction = 1 - retained/total is computed and printed every run,
  * segment count <= floor((1 - reduction_target) * T / 6.5)  (=> <= 55 for a 30-min session).

The last bound encodes the >= 80% reduction target as a hard check: if the gate over-produces
past it, the run stops with a message telling you to tune the thresholds, rather than silently
shipping a low-reduction result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..config import Config
from ..contracts import STRUCTURAL_MIN_RETAINED_S, Segment
from ..fusion.linear import FusedWindow
from .hysteresis import ActiveSpan, HysteresisParams, run_gate


@dataclass
class GateResult:
    segments: list[Segment]
    fused: list[FusedWindow]
    total_duration_s: float
    retained_s: float
    reduction: float
    max_segments_bound: int

    @property
    def min_dur(self) -> float:
        return min((s.duration for s in self.segments), default=0.0)

    @property
    def max_dur(self) -> float:
        return max((s.duration for s in self.segments), default=0.0)

    @property
    def mean_dur(self) -> float:
        return (sum(s.duration for s in self.segments) / len(self.segments)) if self.segments else 0.0


@dataclass
class _Span:
    t_start: float
    t_end: float
    active_start: float
    active_end: float
    peak: float
    mean: float
    peak_ts: float


def _to_retained(span: ActiveSpan, pre: float, post: float, total: float) -> _Span:
    return _Span(
        t_start=max(0.0, span.active_start - pre),
        t_end=min(total, span.active_end + post),
        active_start=span.active_start,
        active_end=span.active_end,
        peak=span.peak,
        mean=span.mean,
        peak_ts=span.peak_ts,
    )


def close_or_merge(acc: list[_Span], nxt: _Span, merge_gap: float) -> None:
    """Append `nxt`, or merge it into the previous retained span if the gap is <= merge_gap."""
    if acc and nxt.t_start - acc[-1].t_start >= 0 and (nxt.t_start - acc[-1].t_end) <= merge_gap:
        prev = acc[-1]
        merged = _Span(
            t_start=min(prev.t_start, nxt.t_start),
            t_end=max(prev.t_end, nxt.t_end),
            active_start=min(prev.active_start, nxt.active_start),
            active_end=max(prev.active_end, nxt.active_end),
            peak=max(prev.peak, nxt.peak),
            mean=(prev.mean + nxt.mean) / 2.0,
            peak_ts=prev.peak_ts if prev.peak >= nxt.peak else nxt.peak_ts,
        )
        acc[-1] = merged
    else:
        acc.append(nxt)


def segment_spans(fused: list[FusedWindow], cfg: Config, *, session_id: str, source: str) -> GateResult:
    g = cfg.get("gating")
    total = fused[-1].t_end if fused else 0.0
    spans = run_gate(fused, HysteresisParams.from_config(cfg))

    pre, post = g["pre_roll_s"], g["post_roll_s"]
    min_active, merge_gap, max_dur = g["min_active_s"], g["merge_gap_s"], g["max_duration_s"]

    # 1) min ACTIVE-span filter (measured on the ACTIVE span, not the retained span)
    kept = [s for s in spans if (s.active_end - s.active_start) >= min_active - 1e-9]
    # 2) pre/post-roll
    retained = [_to_retained(s, pre, post, total) for s in kept]
    retained.sort(key=lambda r: r.t_start)
    # 3) merge-gap
    merged: list[_Span] = []
    for r in retained:
        close_or_merge(merged, r, merge_gap)

    # 4) max-duration cap + build Segments (contracts enforces the 6.5 s floor on construction)
    segments: list[Segment] = []
    for i, r in enumerate(merged):
        t_start, t_end = r.t_start, r.t_end
        if t_end - t_start > max_dur:
            t_end = t_start + max_dur  # bound Tier-2 work per segment
        # Honor the frozen 6.5 s structural floor even when session-boundary clamping ate part of a
        # roll (an event in the first 2 s or last 3 s): grow the un-clamped side, within [0, total].
        deficit = STRUCTURAL_MIN_RETAINED_S - (t_end - t_start)
        if deficit > 1e-9:
            grow_end = min(deficit, total - t_end)
            t_end += grow_end
            deficit -= grow_end
            if deficit > 1e-9:
                t_start = max(0.0, t_start - deficit)
        keyframes = tuple(sorted({
            min(max(r.active_start, t_start), t_end),
            min(max(r.peak_ts, t_start), t_end),
            min(max(r.active_end, t_start), t_end),
        }))
        seg = Segment(
            segment_id=f"{session_id}_seg{i:03d}",
            session_id=session_id,
            source=source,
            t_start=t_start,
            t_end=t_end,
            active_start=max(r.active_start, t_start),
            active_end=min(r.active_end, t_end),
            salience_peak=r.peak,
            salience_mean=r.mean,
            keyframe_ts=keyframes,
        )
        assert seg.duration <= max_dur + 1e-6, f"{seg.segment_id} exceeds max_duration {max_dur}"
        assert seg.duration >= STRUCTURAL_MIN_RETAINED_S - 1e-6, f"{seg.segment_id} below floor"
        segments.append(seg)

    retained_s = sum(s.duration for s in segments)
    reduction = 1.0 - (retained_s / total) if total > 0 else 0.0

    reduction_target = cfg.get("targets.reduction_min", 0.80)
    max_segments_bound = math.floor((1.0 - reduction_target) * total / STRUCTURAL_MIN_RETAINED_S)
    if len(segments) > max_segments_bound:
        raise AssertionError(
            f"{len(segments)} segments exceeds the bound floor((1-{reduction_target})*{total:.0f}"
            f"/{STRUCTURAL_MIN_RETAINED_S}) = {max_segments_bound}. The gate is over-producing: "
            f"reduction {reduction:.3f} < target {reduction_target}. Tune (theta_on, theta_off) upward "
            f"or widen persistence — do not ship a low-reduction gate."
        )

    return GateResult(segments=segments, fused=fused, total_duration_s=total,
                      retained_s=retained_s, reduction=reduction,
                      max_segments_bound=max_segments_bound)
