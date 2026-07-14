"""The cascade router: auto-accept / uncertain-band validate / (shadow) (CLAUDE.md §4).

    s = segment score (salience_peak by default)
    s >= tau_hi   -> AUTO-ACCEPT, no model call, free
    theta_on<=s<tau_hi -> VALIDATE (one call, budgeted, cached)
    s < theta_on  -> the gate never emitted it (shadow band; handled in eval only)

The validator can only REMOVE candidates (keep/drop) — never add — so recall stays Tier-1's
job. Budget breach and any validator error both fail open (keep=True). Every paid verdict is
cached by sha256(evidence + config_hash) so re-running eval costs $0.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import Config
from ..contracts import Frame, Segment, Verdict
from .budget import SessionBudget
from .cache import VerdictCache, evidence_key
from .evidence import build_evidence
from .validators import build_validator


@dataclass
class CascadeStats:
    n_cand: int = 0
    n_auto: int = 0
    n_unc: int = 0
    n_calls: int = 0
    n_cache_hits: int = 0
    n_failed_open: int = 0
    cost_usd: float = 0.0
    validator: str = ""
    budget_breached: str | None = None
    kept: int = 0
    dropped: int = 0


@dataclass
class CascadeResult:
    verdicts: list[Verdict] = field(default_factory=list)
    stats: CascadeStats = field(default_factory=CascadeStats)


def _score(seg: Segment, field_name: str) -> float:
    return seg.salience_mean if field_name == "salience_mean" else seg.salience_peak


def run_cascade(segments: list[Segment], frames: list[Frame], cfg: Config) -> CascadeResult:
    tau_hi = float(cfg.get("cascade.tau_hi"))
    theta_on = float(cfg.get("gating.theta_on"))
    score_field = cfg.get("cascade.score_field", "salience_peak")
    include_transcript = cfg.get("cascade.evidence.include_transcript", True)

    validator = build_validator(cfg)
    budget = SessionBudget.from_config(cfg)
    cache = VerdictCache(cfg.get("cascade.cache.path", "out/cache/verdicts"))
    use_cache = cfg.get("cascade.cache.enabled", True)

    stats = CascadeStats(validator=validator.name)
    verdicts: list[Verdict] = []

    for seg in segments:
        stats.n_cand += 1
        s = _score(seg, score_field)
        if s >= tau_hi:
            stats.n_auto += 1
            verdicts.append(Verdict(
                segment_id=seg.segment_id, keep=True, confidence=1.0,
                reason=f"auto-accept: score {s:.3f} >= tau_hi {tau_hi}",
                tags=("auto_accept",), validator="auto_accept", route="auto_accept"))
            continue

        # uncertain band -> validate
        stats.n_unc += 1
        transcript = _cheap_transcript(seg, frames, cfg) if include_transcript else None
        ev = build_evidence(seg, frames, cfg, transcript=transcript)
        key = evidence_key(ev, cfg.config_hash)

        if use_cache:
            cached = cache.get(key)
            if cached is not None:
                stats.n_cache_hits += 1
                verdicts.append(cached)
                continue

        if not budget.can_spend():
            stats.budget_breached = budget.breached_reason
            verdicts.append(Verdict(
                segment_id=seg.segment_id, keep=True, confidence=0.0,
                reason=f"budget breached ({budget.breached_reason}); fail open"[:200],
                tags=("fail_open", "budget"), validator=validator.name, route="validate",
                failed_open=True))
            stats.n_failed_open += 1
            continue

        verdict = validator.validate(seg, ev)  # contract: never raises
        budget.record(verdict.cost_usd)
        stats.n_calls += 1
        stats.cost_usd += verdict.cost_usd
        if verdict.failed_open:
            stats.n_failed_open += 1
        if use_cache and not verdict.failed_open:
            cache.put(key, verdict)
        verdicts.append(verdict)

    stats.budget_breached = stats.budget_breached or budget.breached_reason
    stats.kept = sum(1 for v in verdicts if v.keep)
    stats.dropped = sum(1 for v in verdicts if not v.keep)
    return CascadeResult(verdicts=verdicts, stats=stats)


def _cheap_transcript(seg: Segment, frames, cfg) -> str | None:
    """whisper tiny.en on the uncertain-band segment, if faster-whisper is installed; else None.

    The real ASR audio slice is wired in tier2/asr.py; here we only attempt the cheap tiny.en
    pass and return None when the backend is absent (no fabrication).
    """
    from .._deps import optional

    if optional("faster_whisper") is None:
        return None
    return None  # audio-slice plumbing lands with tier2.asr; tiny.en cache key reused there
