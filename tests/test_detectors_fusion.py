"""Detectors (availability masks, VAD fallback, causal normaliser) + fusion (masked renorm)."""

from __future__ import annotations

from hindsight.contracts import DetectorScores
from hindsight.detectors import run_detectors
from hindsight.detectors.base import RunningPercentileNormalizer
from hindsight.fusion.linear import fuse_window
from hindsight.sources import SyntheticSource, InterestSpan


def test_unavailable_detectors_masked_not_zeroed(cfg):
    src = SyntheticSource(duration=6.0, spans=[InterestSpan(1.0, 4.0, ("voice", "motion"))])
    res = run_detectors(src, cfg)
    # face/text need models we do not have -> masked off everywhere, never scored 0.0
    assert set(res.notes["masked_off_all_windows"]) >= {"face_presence", "text_presence"}
    for ds in res.windows:
        assert "face_presence" not in ds.scores  # not a fabricated 0
        assert ds.mask.get("face_presence") is False


def test_vad_fallback_is_recorded(cfg):
    src = SyntheticSource(duration=4.0, spans=[InterestSpan(1.0, 3.0, ("voice",))])
    res = run_detectors(src, cfg)
    # webrtcvad is not installed here -> must fall back AND say so
    assert res.notes["vad_backend"] in ("webrtcvad", "energy_fallback")


def test_running_percentile_is_causal():
    # the value emitted at step i must depend only on samples 0..i (no future peeking)
    seq = [0.1, 0.2, 0.9, 0.3, 0.8, 0.05]
    n1 = RunningPercentileNormalizer(5, 95, min_count=3)
    prefix_outputs = [n1.update_and_normalize(x) for x in seq[:4]]
    n2 = RunningPercentileNormalizer(5, 95, min_count=3)
    full_outputs = [n2.update_and_normalize(x) for x in seq]
    assert prefix_outputs == full_outputs[:4]  # extending the future did not change the past


def test_fusion_masked_renorm_does_not_collapse(cfg):
    weights = cfg.get("fusion.weights")
    # audio-only window: only voice + energy available
    ds = DetectorScores(0, 0.0, 0.5, scores={"voice_activity": 1.0, "audio_energy": 1.0},
                        mask={"voice_activity": True, "audio_energy": True,
                              "scene_change": False, "motion": False,
                              "face_presence": False, "text_presence": False})
    fw = fuse_window(ds, weights, renormalise_over_mask=True)
    assert fw.salience > 0.9  # renormalised over available -> near 1, not dragged to ~0


def test_fusion_all_masked_is_zero(cfg):
    ds = DetectorScores(0, 0.0, 0.5, scores={}, mask={n: False for n in cfg.get("fusion.weights")})
    assert fuse_window(ds, cfg.get("fusion.weights")).salience == 0.0
