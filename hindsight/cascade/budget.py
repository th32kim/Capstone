"""Session budget for the validator. On breach: log, stop calling, fail open (CLAUDE.md §4).

Never lose a memory to a flaky/expensive API. Once max_calls or max_cost is reached, the
router stops calling the validator and keeps every remaining candidate (fail open), which
degrades the system to the known-good Tier-1-only state.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SessionBudget:
    max_calls: int
    max_cost_usd: float
    calls: int = 0
    cost_usd: float = 0.0
    breached_reason: str | None = None

    @classmethod
    def from_config(cls, cfg) -> "SessionBudget":
        b = cfg.get("cascade.budget")
        return cls(max_calls=int(b["max_calls_per_session"]), max_cost_usd=float(b["max_cost_usd"]))

    def can_spend(self, next_cost_usd: float = 0.0) -> bool:
        if self.calls >= self.max_calls:
            self.breached_reason = f"max_calls_per_session ({self.max_calls}) reached"
            return False
        if self.cost_usd + next_cost_usd > self.max_cost_usd:
            self.breached_reason = f"max_cost_usd ({self.max_cost_usd}) would be exceeded"
            return False
        return True

    def record(self, cost_usd: float) -> None:
        self.calls += 1
        self.cost_usd += cost_usd
