"""The four-state hysteresis gate (CLAUDE.md §3.3, Table 3.2-8).

States: IDLE -> ARMING -> ACTIVE -> CLOSING. Two thresholds with a hysteresis gap
(theta_on > theta_off) and two persistence counters reject spikes and brief dips. All
parameters are config; nothing here is a bare float.

Output is a list of ACTIVE spans in seconds (onset .. offset), *excluding* pre/post-roll.
The segmenter turns those into retained Segments. The FSM is single-pass and causal so it
can run "while capture is ongoing" (FS4).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..fusion.linear import FusedWindow


@dataclass(frozen=True)
class ActiveSpan:
    active_start: float   # seconds, onset (first window that crossed theta_on)
    active_end: float     # seconds, offset (first sub-theta_off window that confirmed close)
    peak: float
    mean: float
    peak_ts: float


@dataclass
class HysteresisParams:
    theta_on: float
    theta_off: float
    activation_persistence: int
    deactivation_persistence: int

    @classmethod
    def from_config(cls, cfg) -> "HysteresisParams":
        g = cfg.get("gating")
        return cls(g["theta_on"], g["theta_off"],
                   int(g["activation_persistence"]), int(g["deactivation_persistence"]))


def run_gate(fused: list[FusedWindow], p: HysteresisParams) -> list[ActiveSpan]:
    spans: list[ActiveSpan] = []
    state = "IDLE"
    arm_start_idx = 0
    arm_count = 0
    close_start_idx = 0
    close_count = 0
    acc: list[FusedWindow] = []  # windows accumulated for the current ACTIVE span

    def emit(offset_ts: float) -> None:
        if not acc:
            return
        sal = [w.salience for w in acc]
        peak = max(sal)
        peak_ts = acc[sal.index(peak)].t_start
        spans.append(ActiveSpan(
            active_start=acc[0].t_start, active_end=offset_ts,
            peak=peak, mean=sum(sal) / len(sal), peak_ts=peak_ts))

    for fw in fused:
        s = fw.salience
        if state == "IDLE":
            if s >= p.theta_on:
                state, arm_start_idx, arm_count = "ARMING", fw.index, 1
        elif state == "ARMING":
            if s >= p.theta_on:
                arm_count += 1
                if arm_count >= p.activation_persistence:
                    state = "ACTIVE"
                    # the ACTIVE span begins at the onset window where arming started
                    acc = [w for w in fused if arm_start_idx <= w.index <= fw.index]
            else:
                state = "IDLE"
        elif state == "ACTIVE":
            acc.append(fw)
            if s < p.theta_off:
                state, close_start_idx, close_count = "CLOSING", fw.index, 1
        elif state == "CLOSING":
            acc.append(fw)
            if s < p.theta_off:
                close_count += 1
                if close_count >= p.deactivation_persistence:
                    # span ends at the first sub-threshold window (close_start_idx)
                    offset_ts = next(w.t_start for w in fused if w.index == close_start_idx)
                    acc = [w for w in acc if w.index < close_start_idx]
                    emit(offset_ts)
                    state, acc = "IDLE", []
            else:
                state = "ACTIVE"  # dip recovered

    # stream ended while still open -> close at the last window's end
    if state in ("ACTIVE", "CLOSING") and acc:
        emit(acc[-1].t_end)
    return spans
