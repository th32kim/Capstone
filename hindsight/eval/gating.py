"""eval/gating — segment-level P/R/F1 vs a hand-labelled corpus (FS5), + the joint sweep.

Match = IoU >= match_iou between a predicted retained segment and a ground-truth label span.
CRITICAL (docs/CORPUS.md): recall may be computed ONLY from `dense` labels — assisted labels
never reveal the events the detector never proposed, so recall from them is meaningless. This
module RAISES if asked for recall from assisted labels. It also reports Cohen's κ between two
labellers; κ < 0.60 means the labels (and any F1 from them) are untrustworthy.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from ..artifacts import read_scores
from ..config import Config
from ..contracts import DECISION_WINDOW_S
from ..fusion import fuse_all
from ..gating import segment_spans

LABELS_DIR = Path("data/labels")


@dataclass
class Interval:
    t_start: float
    t_end: float
    mode: str = "dense"
    labeller: str = ""


def load_labels(clip: str) -> list[Interval]:
    path = LABELS_DIR / f"{clip}.csv"
    if not path.exists():
        return []
    out: list[Interval] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out.append(Interval(float(r["t_start"]), float(r["t_end"]),
                                r.get("mode", "dense"), r.get("labeller", "")))
    return out


class AssistedLabelsError(RuntimeError):
    """Raised when recall is requested from assisted labels — it would be flatteringly wrong."""


def recall_labels(labels: list[Interval]) -> list[Interval]:
    """The dense subset usable for recall. RAISES on assisted-only (docs/CORPUS.md)."""
    dense = [lb for lb in labels if lb.mode == "dense"]
    if not dense:
        raise AssistedLabelsError(
            "recall requires dense labels; refusing to compute it from assisted labels "
            "(you never see the events the detector never proposed). See docs/CORPUS.md.")
    return dense


def _iou(a: Interval, b_start: float, b_end: float) -> float:
    lo, hi = max(a.t_start, b_start), min(a.t_end, b_end)
    inter = max(0.0, hi - lo)
    union = (a.t_end - a.t_start) + (b_end - b_start) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class PRF:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float | None
    f1: float | None
    recall_measurable: bool
    reason: str = ""


def prf(pred: list[tuple[float, float]], labels: list[Interval], iou_thr: float,
        allow_recall: bool) -> PRF:
    gts = labels
    matched_gt = set()
    tp = 0
    for (ps, pe) in pred:
        best = max((_iou(g, ps, pe), i) for i, g in enumerate(gts)) if gts else (0.0, -1)
        if best[0] >= iou_thr and best[1] not in matched_gt:
            tp += 1
            matched_gt.add(best[1])
    fp = len(pred) - tp
    fn = len(gts) - len(matched_gt)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    if not allow_recall:
        return PRF(tp, fp, fn, precision, None, None, False,
                   "recall NOT MEASURABLE (no dense labels)")
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return PRF(tp, fp, fn, precision, recall, f1, True)


def cohens_kappa(clip: str, duration: float, window_s: float = DECISION_WINDOW_S) -> float | None:
    labels = load_labels(clip)
    by_labeller: dict[str, list[Interval]] = {}
    for lb in labels:
        by_labeller.setdefault(lb.labeller, []).append(lb)
    if len(by_labeller) < 2:
        return None
    names = list(by_labeller)[:2]
    n = int(duration // window_s) + 1

    def series(intervals):
        s = [0] * n
        for iv in intervals:
            for w in range(int(iv.t_start // window_s), int(iv.t_end // window_s) + 1):
                if 0 <= w < n:
                    s[w] = 1
        return s

    a, b = series(by_labeller[names[0]]), series(by_labeller[names[1]])
    both1 = sum(1 for x, y in zip(a, b) if x == y == 1)
    both0 = sum(1 for x, y in zip(a, b) if x == y == 0)
    po = (both1 + both0) / n
    pa1, pb1 = sum(a) / n, sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return (po - pe) / (1 - pe) if (1 - pe) else 1.0


def evaluate(clip: str, cfg: Config) -> PRF | None:
    from ..artifacts import read_segments

    segs = read_segments(clip)
    labels = load_labels(clip)
    if not labels:
        return None
    allow_recall = any(lb.mode == "dense" for lb in labels)
    pred = [(s.t_start, s.t_end) for s in segs]
    return prf(pred, labels, cfg.get("eval.match_iou", 0.5), allow_recall)


def sweep(clip: str, cfg: Config) -> list[dict]:
    """Joint sweep over (theta_on, theta_off=theta_on-gap, tau_hi). Re-fuses+gates from scores."""
    windows, _names = read_scores(clip)
    labels = load_labels(clip)
    allow_recall = any(lb.mode == "dense" for lb in labels) if labels else False
    grid = cfg.get("eval.sweep")
    gap = grid["theta_off_gap"]
    rows: list[dict] = []
    for th_on in grid["theta_on"]:
        for tau_hi in grid["tau_hi"]:
            c = cfg.with_overrides(**{
                "gating.theta_on": th_on, "gating.theta_off": round(th_on - gap, 4),
                "cascade.tau_hi": tau_hi})
            fused = fuse_all(windows, c)
            try:
                gr = segment_spans(fused, c, session_id=clip, source=clip)
            except AssertionError:
                continue  # over-produced past the reduction bound at this operating point
            m = prf([(s.t_start, s.t_end) for s in gr.segments], labels or [],
                    cfg.get("eval.match_iou", 0.5), allow_recall) if labels else None
            rows.append({
                "theta_on": th_on, "theta_off": round(th_on - gap, 4), "tau_hi": tau_hi,
                "segments": len(gr.segments), "reduction": round(gr.reduction, 3),
                "precision": round(m.precision, 3) if m else None,
                "recall": round(m.recall, 3) if (m and m.recall is not None) else None,
                "f1": round(m.f1, 3) if (m and m.f1 is not None) else None,
            })
    return rows
