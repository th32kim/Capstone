"""Hysteresis FSM + segmenter structural invariants."""

from __future__ import annotations

import pytest

from hindsight.contracts import STRUCTURAL_MIN_RETAINED_S, Segment
from hindsight.fusion.linear import FusedWindow
from hindsight.gating.hysteresis import HysteresisParams, run_gate
from hindsight.gating.segmenter import segment_spans


def _fused(saliences, window_s=0.5):
    return [FusedWindow(i, i * window_s, (i + 1) * window_s, s, 6) for i, s in enumerate(saliences)]


def test_hysteresis_persistence_rejects_spike():
    p = HysteresisParams(theta_on=0.58, theta_off=0.38, activation_persistence=2,
                         deactivation_persistence=3)
    # a single high window (spike) must NOT open a segment
    spans = run_gate(_fused([0.1, 0.9, 0.1, 0.1, 0.1]), p)
    assert spans == []


def test_hysteresis_opens_and_closes():
    p = HysteresisParams(0.58, 0.38, 2, 3)
    # sustained high (>=2), then sustained low (>=3) closes
    sal = [0.1, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.1, 0.1, 0.1, 0.1]
    spans = run_gate(_fused(sal), p)
    assert len(spans) == 1
    assert spans[0].active_start <= 0.5
    assert spans[0].peak >= 0.7


def test_segment_floor_enforced_by_contract():
    with pytest.raises(ValueError):
        Segment("s", "sess", "src", t_start=0.0, t_end=3.0,  # 3.0 < 6.5 floor
                active_start=1.0, active_end=2.0, salience_peak=0.9, salience_mean=0.8,
                keyframe_ts=(1.0,))


def test_segmenter_reduction_and_bounds(cfg):
    # one clear event in a long quiet session -> reduces, floors/caps hold
    sal = [0.05] * 60 + [0.7] * 12 + [0.05] * 128
    gr = segment_spans(_fused(sal), cfg, session_id="t", source="t")
    assert len(gr.segments) == 1
    seg = gr.segments[0]
    assert seg.duration >= STRUCTURAL_MIN_RETAINED_S - 1e-6
    assert seg.duration <= cfg.get("gating.max_duration_s") + 1e-6
    assert 0.0 <= gr.reduction <= 1.0
    assert len(gr.segments) <= gr.max_segments_bound


def test_segmenter_min_active_filter(cfg):
    # a 1.0 s active blip (< 3.0 s min_active) must be dropped
    sal = [0.05] * 40 + [0.7, 0.7] + [0.05] * 40
    gr = segment_spans(_fused(sal), cfg, session_id="t", source="t")
    assert gr.segments == []


def test_over_production_raises_reduction_bound(cfg):
    # 10 events spaced 12 s apart (retained ~8 s, 4 s gaps -> do NOT merge) over a 120 s session.
    # The gate over-produces past floor(0.2*120/6.5)=3 segments, missing the reduction target.
    sal = ([0.7] * 6 + [0.05] * 18) * 10  # 240 windows = 120 s
    with pytest.raises(AssertionError):
        segment_spans(_fused(sal), cfg, session_id="t", source="t")
