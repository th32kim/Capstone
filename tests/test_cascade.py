"""Cascade: routing, null baseline, budget fail-open, cache round-trip."""

from __future__ import annotations

from hindsight.cascade import run_cascade
from hindsight.cascade.budget import SessionBudget
from hindsight.cascade.cache import VerdictCache, evidence_key
from hindsight.cascade.validators.null import NullValidator
from hindsight.contracts import Evidence, Segment, Verdict


def _seg(seg_id, peak, t_start=0.0):
    return Segment(seg_id, "sess", "src", t_start=t_start, t_end=t_start + 10.0,
                   active_start=t_start + 2.0, active_end=t_start + 7.0,
                   salience_peak=peak, salience_mean=peak * 0.8, keyframe_ts=(t_start + 3.0,))


def _ev(seg_id):
    return Evidence(seg_id, keyframes_jpeg=(b"x",), transcript=None,
                    detector_evidence={}, duration_s=10.0, salience_peak=0.6, salience_mean=0.5)


def test_null_validator_keeps():
    v = NullValidator().validate(_seg("s", 0.6), _ev("s"))
    assert v.keep and v.validator == "null"


def test_routing_auto_vs_uncertain(cfg):
    # tau_hi=0.75, theta_on=0.58 by default
    segs = [_seg("hi", 0.90, 0.0), _seg("mid", 0.65, 20.0)]
    cfg = cfg.with_overrides(**{"cascade.validator": "null", "cascade.cache.enabled": False})
    res = run_cascade(segs, frames=[], cfg=cfg)
    routes = {v.segment_id: v.route for v in res.verdicts}
    assert routes["hi"] == "auto_accept"
    assert routes["mid"] == "validate"
    assert res.stats.n_auto == 1 and res.stats.n_unc == 1


def test_budget_fail_open(cfg):
    b = SessionBudget(max_calls=0, max_cost_usd=0.0)
    assert not b.can_spend()
    segs = [_seg("mid", 0.65, 0.0)]
    cfg = cfg.with_overrides(**{"cascade.validator": "null",
                                "cascade.budget.max_calls_per_session": 0,
                                "cascade.cache.enabled": False})
    res = run_cascade(segs, frames=[], cfg=cfg)
    assert res.verdicts[0].keep and res.verdicts[0].failed_open
    assert res.stats.budget_breached


def test_cache_round_trip(tmp_path):
    cache = VerdictCache(tmp_path)
    ev = _ev("s")
    key = evidence_key(ev, "confighash")
    v = Verdict("s", keep=False, confidence=0.9, reason="drop", tags=("x",),
                validator="clip_local", route="validate")
    cache.put(key, v)
    got = cache.get(key)
    assert got is not None and got.keep is False and got.validator == "clip_local"
    assert evidence_key(ev, "otherhash") != key  # config_hash participates
