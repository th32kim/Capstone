"""D7 dwell — angular velocity dropping below threshold after a spell above it (DESIGN_DELTAS D-3).

"The wearer stopped walking and is now looking at something" — one of the strongest, cheapest
salience cues a wearable has. Sourced from the ego-motion estimate already computed in D2, or
(GoPro) from the GPMF gyro track.

HARD CAVEAT: a GPMF-sourced dwell is OFFLINE-ONLY (GPMF is unreadable while recording) and must
be availability-masked off the real-time path. It is disabled by default in config. When no
angular-velocity source is wired, score() returns None -> masked off (never scored 0).
"""

from __future__ import annotations

from typing import Callable

from ...contracts import Window


class DwellDetector:
    name = "dwell"
    modality = "imu"

    def __init__(self, cadence: int = 1, settle_threshold_dps: float = 15.0,
                 settle_hold_s: float = 1.0, window_s: float = 0.5) -> None:
        self.cadence = cadence
        self.settle_threshold = settle_threshold_dps
        self.settle_hold_windows = max(1, round(settle_hold_s / window_s))
        self._velocity_source: Callable[[], float] | None = None
        self._was_above = False
        self._settled_for = 0

    def wire_velocity(self, source: Callable[[], float]) -> None:
        """Runner wires this to the ego-motion magnitude, or a GPMF gyro series (offline)."""
        self._velocity_source = source

    def score(self, w: Window) -> float | None:
        if self._velocity_source is None:
            return None
        v = self._velocity_source()
        if v > self.settle_threshold:
            self._was_above = True
            self._settled_for = 0
            return 0.0
        # below threshold
        if self._was_above:
            self._settled_for += 1
            if self._settled_for >= self.settle_hold_windows:
                return 1.0  # settled after having moved => strong dwell cue
            return 0.5
        return 0.0
