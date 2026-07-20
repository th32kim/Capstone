"""Verdict cache: sha256(segment_bytes + config_hash) -> Verdict. Re-running eval costs $0.

CLAUDE.md §4. `segment_bytes` is a stable serialisation of the evidence the validator sees
(keyframe bytes + transcript + detector evidence), so the same segment under the same config
never triggers a second paid call.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from ..contracts import Evidence, Verdict


def evidence_key(ev: Evidence, config_hash: str) -> str:
    h = hashlib.sha256()
    for kf in ev.keyframes_jpeg:
        h.update(kf)
    # None ("ASR could not run") and "" ("confirmed silence") are DIFFERENT evidence — see
    # tier2/asr.py transcribe_or_none — and must not collide into one verdict-cache key.
    h.update(b"\x00" if ev.transcript is None else b"\x01" + ev.transcript.encode("utf-8"))
    h.update(json.dumps(ev.detector_evidence, sort_keys=True).encode("utf-8"))
    h.update(f"{ev.duration_s:.3f}|{ev.salience_peak:.6f}|{ev.salience_mean:.6f}".encode())
    h.update(config_hash.encode("utf-8"))
    return h.hexdigest()


class VerdictCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def _file(self, key: str) -> Path:
        return self.path / f"{key}.json"

    def get(self, key: str) -> Verdict | None:
        f = self._file(key)
        if not f.exists():
            return None
        d = json.loads(f.read_text(encoding="utf-8"))
        d["keyframes_jpeg"] = ()  # never stored
        d.pop("keyframes_jpeg", None)
        return Verdict(
            segment_id=d["segment_id"], keep=d["keep"], confidence=d["confidence"],
            reason=d["reason"], tags=tuple(d["tags"]), validator=d["validator"], route=d["route"],
            trim=tuple(d["trim"]) if d.get("trim") else None,
            cost_usd=d.get("cost_usd", 0.0), latency_ms=d.get("latency_ms", 0.0),
            failed_open=d.get("failed_open", False),
        )

    def put(self, key: str, verdict: Verdict) -> None:
        d = asdict(verdict)
        d["tags"] = list(verdict.tags)
        d["trim"] = list(verdict.trim) if verdict.trim else None
        self._file(key).write_text(json.dumps(d, indent=2), encoding="utf-8")
