"""eval/text_backend — EAST vs MSER precision/recall on a hand-verified text-bearing clip (V2-2).

Ground truth: `data/labels/<clip>_text.csv` (t_start,t_end,text_present), one row per 0.5 s
window. This is a narrower, single-detector judgment than the dense event labels in
`docs/CORPUS.md` (`data/labels/<clip>.csv`) — it only asks "is there legible on-screen text in
this window," not "is this an interesting event." Provenance for the shipped `plane_1_text.csv`:
auto-derived from whether Tesseract found any text on the frame (an independent third algorithm,
not EAST or MSER), then visually spot-checked by a human across the full positive/negative/
borderline range before being trusted (see docs/DESIGN_DELTAS.md D-8 addendum).

This module re-decodes the clip (needs raw frames -- `w.frames` -- unlike the persisted
out/scores/*.csv, which only holds the fused scalar). Each backend is built once, scored on every
labelled window, and reported against threshold `theta_on` (the gate's activation threshold, the
natural operating point for a presence detector) unless a different threshold is requested.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from ..clock import align_windows
from ..config import Config
from ..contracts import DECISION_WINDOW_S
from ..detectors.video.text_presence import TextPresenceDetector

LABELS_DIR = Path("data/labels")


@dataclass
class TextLabel:
    t_start: float
    t_end: float
    text_present: bool


def load_text_labels(clip: str) -> list[TextLabel]:
    path = LABELS_DIR / f"{clip}_text.csv"
    if not path.exists():
        return []
    out: list[TextLabel] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out.append(
                TextLabel(
                    float(r["t_start"]),
                    float(r["t_end"]),
                    r["text_present"].strip() in ("1", "true", "True"),
                )
            )
    return out


@dataclass
class BackendResult:
    backend: str  # the backend that actually ran (honest -- may differ from requested)
    available: bool
    reason: str | None  # why unavailable / why it fell back, if applicable
    n_windows: int
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float | str  # "NOT MEASURED (...)" when undefined, never a fabricated 0/1
    recall: float | str


def _build_detector(cfg: Config, backend: str) -> TextPresenceDetector:
    c = cfg.get("detectors.text_presence")
    east, mser = c.get("east", {}), c.get("mser", {})
    return TextPresenceDetector(
        cadence=1,
        backend=backend,
        min_confidence=c.get("min_confidence", 0.5),
        east_model_path=c.get("east_model_path"),
        east_input_size=east.get("input_size", 320),
        east_nms_iou=east.get("nms_iou", 0.4),
        east_count_saturation=east.get("count_saturation", 20.0),
        east_weights=tuple(east.get("weights", (0.4, 0.3, 0.3))),
        east_max_consecutive_errors=east.get("max_consecutive_errors", 3),
        mser_count_saturation=mser.get("count_saturation", 40.0),
        mser_weights=tuple(mser.get("weights", (0.6, 0.4))),
    )


def _label_for(w_t_start: float, w_t_end: float, labels: list[TextLabel]) -> TextLabel | None:
    mid = (w_t_start + w_t_end) / 2
    for lb in labels:
        if lb.t_start <= mid < lb.t_end:
            return lb
    return None


def run_backend(
    source, cfg: Config, backend: str, labels: list[TextLabel], threshold: float
) -> BackendResult:
    window_s = cfg.get("clock.decision_window_s", DECISION_WINDOW_S)
    frames = list(source.frames())
    chunks = list(source.audio())
    windows = align_windows(frames, chunks, window_s=window_s, duration_s=source.duration_s())

    det = _build_detector(cfg, backend)
    if not det.available:
        return BackendResult(
            "unavailable",
            False,
            "opencv not installed",
            0,
            0,
            0,
            0,
            0,
            "NOT MEASURED",
            "NOT MEASURED",
        )
    if backend == "east" and det.last_backend != "east":
        # honest: requested EAST but no usable model -> refuse to report an EAST number
        return BackendResult(
            "east",
            False,
            det.fallback_reason,
            0,
            0,
            0,
            0,
            0,
            "NOT MEASURED (EAST unavailable: see reason)",
            "NOT MEASURED (EAST unavailable: see reason)",
        )

    tp = fp = fn = tn = 0
    n = 0
    for w in windows:
        lb = _label_for(w.t_start, w.t_end, labels)
        if lb is None:
            continue
        n += 1
        score = det.score(w)
        pred = score is not None and score >= threshold
        if pred and lb.text_present:
            tp += 1
        elif pred and not lb.text_present:
            fp += 1
        elif not pred and lb.text_present:
            fn += 1
        else:
            tn += 1
    precision = (tp / (tp + fp)) if (tp + fp) else "NOT MEASURED (no positive predictions)"
    recall = (tp / (tp + fn)) if (tp + fn) else "NOT MEASURED (no positive labels)"
    return BackendResult(det.last_backend, True, None, n, tp, fp, fn, tn, precision, recall)


def benchmark(source, cfg: Config, clip: str, threshold: float | None = None) -> dict:
    labels = load_text_labels(clip)
    if not labels:
        return {"NOT MEASURED": f"no ground-truth labels at data/labels/{clip}_text.csv"}
    thr = threshold if threshold is not None else float(cfg.get("gating.theta_on", 0.58))
    east = run_backend(source, cfg, "east", labels, thr)
    mser = run_backend(source, cfg, "mser_swt", labels, thr)
    return {
        "clip": clip,
        "threshold": thr,
        "n_labels": len(labels),
        "east": vars(east),
        "mser_swt": vars(mser),
    }
