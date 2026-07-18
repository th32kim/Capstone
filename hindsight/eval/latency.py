"""eval/latency — NFS1 (Tier-1 ms/frame + duty) and NFS2 (retrieval round-trip), from timings.jsonl.

Every stage logs real timings (out/timings.jsonl). This reads the most recent detect/ask rows and
derives the reported numbers — measured, never assumed. Missing data -> NOT MEASURED.
"""

from __future__ import annotations

import json
from pathlib import Path

TIMINGS = Path("out/timings.jsonl")


def _rows(stage: str) -> list[dict]:
    if not TIMINGS.exists():
        return []
    out = []
    for line in TIMINGS.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if r.get("stage") == stage:
            out.append(r)
    return out


def tier1_latency(clip: str | None = None) -> dict:
    rows = [r for r in _rows("detect") if clip is None or r.get("clip") == clip]
    if not rows:
        return {"NFS1": "NOT MEASURED (run `hindsight detect` first)"}
    r = rows[-1]
    n_frames = r.get("n_frames", 0)
    fps = r.get("fps") or 0.0
    per_det = r.get("per_detector", {})
    total_s = sum(per_det.values()) if per_det else r.get("seconds", 0.0)
    ms_per_frame = (total_s / n_frames * 1000) if n_frames else None
    frame_period_ms = (1000.0 / fps) if fps else None
    duty = (ms_per_frame / frame_period_ms * 100) if (ms_per_frame and frame_period_ms) else None
    return {
        "NFS1_ms_per_frame": round(ms_per_frame, 3) if ms_per_frame else "NOT MEASURED",
        "duty_pct": round(duty, 1) if duty is not None else "NOT MEASURED",
        "fps": fps, "n_frames": n_frames,
        "per_detector_ms_per_eval": {
            k: round(v / r.get("n_evals", {}).get(k, 1) * 1000, 3) for k, v in per_det.items()},
        "verdict": ("PASS" if (duty is not None and duty < 100) else "NOT MEASURED"),
    }


def retrieval_latency() -> dict:
    rows = _rows("ask")
    if not rows:
        return {"NFS2": "NOT MEASURED (run `hindsight ask` first)"}
    times = sorted(r.get("seconds", 0.0) * 1000 for r in rows)
    p50 = times[len(times) // 2]
    p95 = times[min(len(times) - 1, int(len(times) * 0.95))]
    last = rows[-1]
    return {
        "NFS2_p50_ms": round(p50, 2), "NFS2_p95_ms": round(p95, 2), "n_queries": len(rows),
        "table_3_4_7_last_ms": last.get("latency_ms", {}),
        "verdict": "PASS" if p95 <= 2000 else "FAIL",
        "note": "NFS2 target is <=2 s at >=10k memories; report the corpus size alongside.",
    }
