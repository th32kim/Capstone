"""Artifact I/O — every stage writes a file and is independently re-runnable (CLAUDE.md §1.6).

detect -> out/scores/<clip>.csv          (+ .notes.json for backend/staleness provenance)
gate   -> out/segments/<clip>.jsonl      (+ _salience.png if matplotlib present, else .csv)
validate -> out/verdicts/<clip>.jsonl
process  -> out/records/<clip>.jsonl
Plus out/timings.jsonl, appended by every stage (the report needs the timings — §9).

No fabricated numbers: figures are drawn only from data actually written. If matplotlib is
absent the salience trace is still emitted as CSV, and the caller says so.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

from ._deps import optional
from .contracts import DETECTOR_NAMES, DetectorScores, Segment

OUT = Path("out")


def ensure_dirs() -> None:
    for sub in ("scores", "segments", "verdicts", "records", "store", "figures", "reports",
                "cache/verdicts"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)


def log_timing(stage: str, clip: str, seconds: float, **extra) -> None:
    ensure_dirs()
    row = {"stage": stage, "clip": clip, "seconds": round(seconds, 6), **extra}
    with (OUT / "timings.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


# -- scores ----------------------------------------------------------------
def write_scores(clip: str, windows: list[DetectorScores], names: tuple[str, ...],
                 notes: dict | None = None) -> Path:
    ensure_dirs()
    path = OUT / "scores" / f"{clip}.csv"
    cols = ["window_index", "t_start", "t_end"]
    cols += [f"score_{n}" for n in names] + [f"mask_{n}" for n in names]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for ds in windows:
            row = [ds.window_index, f"{ds.t_start:.3f}", f"{ds.t_end:.3f}"]
            row += [f"{ds.scores.get(n, ''):.6f}" if n in ds.scores else "" for n in names]
            row += [int(ds.mask.get(n, False)) for n in names]
            w.writerow(row)
    if notes is not None:
        (OUT / "scores" / f"{clip}.notes.json").write_text(json.dumps(notes, indent=2))
    return path


def read_scores(clip: str) -> tuple[list[DetectorScores], tuple[str, ...]]:
    path = OUT / "scores" / f"{clip}.csv"
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    names = tuple(n for n in DETECTOR_NAMES if f"score_{n}" in (rows[0] if rows else {}))
    out: list[DetectorScores] = []
    for r in rows:
        scores, mask, stale = {}, {}, {}
        for n in names:
            mask[n] = bool(int(r[f"mask_{n}"]))
            if r[f"score_{n}"] != "":
                scores[n] = float(r[f"score_{n}"])
        out.append(DetectorScores(int(r["window_index"]), float(r["t_start"]), float(r["t_end"]),
                                  scores, mask, stale))
    return out, names


# -- segments --------------------------------------------------------------
def segment_to_dict(s: Segment) -> dict:
    return {
        "segment_id": s.segment_id, "session_id": s.session_id, "source": s.source,
        "t_start": s.t_start, "t_end": s.t_end,
        "active_start": s.active_start, "active_end": s.active_end,
        "salience_peak": s.salience_peak, "salience_mean": s.salience_mean,
        "keyframe_ts": list(s.keyframe_ts), "detector_evidence": s.detector_evidence,
    }


def segment_from_dict(d: dict) -> Segment:
    return Segment(
        segment_id=d["segment_id"], session_id=d["session_id"], source=d["source"],
        t_start=d["t_start"], t_end=d["t_end"],
        active_start=d["active_start"], active_end=d["active_end"],
        salience_peak=d["salience_peak"], salience_mean=d["salience_mean"],
        keyframe_ts=tuple(d["keyframe_ts"]), detector_evidence=d.get("detector_evidence", {}),
    )


def write_segments(clip: str, segments: list[Segment]) -> Path:
    ensure_dirs()
    path = OUT / "segments" / f"{clip}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for s in segments:
            fh.write(json.dumps(segment_to_dict(s)) + "\n")
    return path


def read_segments(clip: str) -> list[Segment]:
    path = OUT / "segments" / f"{clip}.jsonl"
    with path.open(encoding="utf-8") as fh:
        return [segment_from_dict(json.loads(line)) for line in fh if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return path


# -- salience figure -------------------------------------------------------
def write_salience_figure(clip: str, fused, segments: list[Segment]) -> tuple[Path, bool]:
    """Salience trace + shaded retained spans. PNG if matplotlib present, else CSV. Returns (path, is_png)."""
    ensure_dirs()
    plt = optional("matplotlib")
    if plt is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot

        xs = [f.t_start for f in fused]
        ys = [f.salience for f in fused]
        fig, ax = pyplot.subplots(figsize=(12, 3))
        ax.plot(xs, ys, lw=0.8, color="#1f77b4")
        for s in segments:
            ax.axvspan(s.t_start, s.t_end, color="#ff7f0e", alpha=0.25)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("fused salience")
        ax.set_title(f"{clip}: salience + retained spans")
        fig.tight_layout()
        path = OUT / "figures" / f"{clip}_salience.png"
        fig.savefig(path, dpi=110)
        pyplot.close(fig)
        return path, True
    # fallback: emit the trace as CSV so nothing is lost / fabricated
    path = OUT / "figures" / f"{clip}_salience.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["t_start", "salience", "retained"])
        spans = [(s.t_start, s.t_end) for s in segments]
        for f in fused:
            retained = any(a <= f.t_start < b for a, b in spans)
            w.writerow([f"{f.t_start:.3f}", f"{f.salience:.6f}", int(retained)])
    return path, False
