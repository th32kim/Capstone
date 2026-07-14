"""eval/cascade — the cost funnel + the validator ablation table (CLAUDE.md §4, M5).

The funnel counts (N_cand, N_auto, N_unc, N_naive, speedup) are MEASURABLE from the gate output
alone and are printed here. The P/R/F1 deltas versus the `null` baseline require a labelled
corpus; without dense labels they print NOT MEASURED rather than a projected number — this table
is the justification for the whole architecture and must never be fabricated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..artifacts import read_segments
from ..config import Config


@dataclass
class Funnel:
    n_cand: int
    n_auto: int
    n_unc: int
    total_s: float
    naive: dict[float, int]     # f_sample -> N_naive
    speedup: dict[float, float]


def cost_funnel(clip: str, cfg: Config) -> Funnel:
    segs = read_segments(clip)
    tau_hi = float(cfg.get("cascade.tau_hi"))
    theta_on = float(cfg.get("gating.theta_on"))
    total_s = max((s.t_end for s in segs), default=0.0)
    n_auto = sum(1 for s in segs if s.salience_peak >= tau_hi)
    n_unc = sum(1 for s in segs if theta_on <= s.salience_peak < tau_hi)
    naive, speedup = {}, {}
    for f in (0.2, 1.0):
        n = math.ceil(total_s * f)
        naive[f] = n
        speedup[f] = (n / n_unc) if n_unc else None  # None => validator not exercised (all auto/none)
    return Funnel(len(segs), n_auto, n_unc, total_s, naive, speedup)


def report(clip: str, cfg: Config) -> dict:
    f = cost_funnel(clip, cfg)
    from .gating import load_labels

    labelled = bool(load_labels(clip))
    def sp(x):
        return round(x, 1) if x is not None else "n/a (no uncertain-band segments)"

    return {
        "clip": clip,
        "funnel": {"n_cand": f.n_cand, "n_auto": f.n_auto, "n_unc": f.n_unc,
                   "N_naive@1.0Hz": f.naive[1.0], "speedup@1.0Hz": sp(f.speedup[1.0]),
                   "N_naive@0.2Hz": f.naive[0.2], "speedup@0.2Hz": sp(f.speedup[0.2])},
        "ablation_prf": "NOT MEASURED (no dense labels)" if not labelled else "see eval/gating",
    }
