"""Weighted linear fusion with the availability mask applied and weights renormalised.

CLAUDE.md §3.2: fuse the six normalised scores by a weighted linear sum, but with the
availability mask applied and the weights **renormalised over the available detectors** —
so an audio-only or video-only window does not collapse to ~0. Weights are config, not code
(a starting point for a sweep, not a result).

A window in which *every* detector is masked out fuses to 0.0 (nothing to say), which the
gate then treats as sub-threshold. That is correct: absence of evidence is not salience.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import Config
from ..contracts import DetectorScores


@dataclass(frozen=True)
class FusedWindow:
    index: int
    t_start: float
    t_end: float
    salience: float
    n_available: int


def fuse_window(ds: DetectorScores, weights: dict[str, float],
                renormalise_over_mask: bool = True) -> FusedWindow:
    avail = [(n, ds.scores[n]) for n in ds.scores if ds.mask.get(n, False)]
    if not avail:
        return FusedWindow(ds.window_index, ds.t_start, ds.t_end, 0.0, 0)

    w_avail = {n: max(0.0, float(weights.get(n, 0.0))) for n, _ in avail}
    total_w = sum(w_avail.values())
    if total_w <= 0.0:
        # available detectors all carry zero weight -> unweighted mean, do not collapse to 0
        salience = sum(s for _, s in avail) / len(avail)
    else:
        if renormalise_over_mask:
            w_avail = {n: w / total_w for n, w in w_avail.items()}
        salience = sum(w_avail[n] * s for n, s in avail)
    salience = min(max(salience, 0.0), 1.0)
    return FusedWindow(ds.window_index, ds.t_start, ds.t_end, salience, len(avail))


def fuse_all(windows: list[DetectorScores], cfg: Config) -> list[FusedWindow]:
    weights = cfg.get("fusion.weights")
    renorm = cfg.get("fusion.renormalise_over_mask", True)
    smoothing = int(cfg.get("fusion.smoothing_windows", 1))
    fused = [fuse_window(ds, weights, renorm) for ds in windows]
    if smoothing > 1:
        fused = _smooth(fused, smoothing)
    return fused


def _smooth(fused: list[FusedWindow], k: int) -> list[FusedWindow]:
    """Causal box smoothing over the last k windows (no peeking forward — FS4)."""
    out: list[FusedWindow] = []
    buf: list[float] = []
    for fw in fused:
        buf.append(fw.salience)
        if len(buf) > k:
            buf.pop(0)
        out.append(FusedWindow(fw.index, fw.t_start, fw.t_end, sum(buf) / len(buf), fw.n_available))
    return out
