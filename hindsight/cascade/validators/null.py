"""NullValidator — accepts everything. THE ablation baseline = Tier-1 alone (CLAUDE.md §4).

Written first, always works, never calls a model. `--validator null` must always be
selectable; it is how the cascade's contribution is measured (and cut, if ΔF1 is small).
"""

from __future__ import annotations

from ...contracts import Evidence, Segment, Verdict


class NullValidator:
    name = "null"

    def validate(self, seg: Segment, ev: Evidence) -> Verdict:
        return Verdict(
            segment_id=seg.segment_id, keep=True, confidence=1.0,
            reason="null validator: accept all (Tier-1-only baseline)",
            tags=(), validator="null", route="validate", cost_usd=0.0, latency_ms=0.0,
        )
