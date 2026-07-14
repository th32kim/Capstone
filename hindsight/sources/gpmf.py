"""GpmfSource — GoPro GPMF telemetry (gyro/accel/GPS) muxed into the MP4. OPTIONAL.

CLAUDE.md §7 / DESIGN_DELTAS D-3: GPMF is readable only *after* the file is written, never
while recording. So anything sourced from GPMF — the `dwell` detector (D7) — is an
**offline-path detector only** and must be availability-masked off the real-time gating
path. This class is therefore not a `Source`: it does not yield Frame/AudioChunk. It yields
timestamped angular-velocity samples that `detectors/imu/dwell.py` consumes offline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .._deps import optional


@dataclass(frozen=True)
class ImuSample:
    t_ms: int
    gyro_dps: tuple[float, float, float]  # angular velocity, deg/s


OFFLINE_ONLY = True  # never let this feed the real-time path (FS4/NFS1)


class GpmfSource:
    """Reads gyro samples from a GoPro MP4's GPMF track. Empty if no telemetry/parser."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def gyro(self) -> list[ImuSample]:
        """Return gyro samples, or [] if the file has no GPMF track or no parser is installed.

        Returning [] (rather than raising) is deliberate: absence of telemetry means the
        dwell detector masks itself off, which is the correct offline-only behaviour.
        """
        parser = optional("gpmf")
        if parser is None:
            return []
        try:
            return self._parse(parser)
        except Exception:  # noqa: BLE001 — malformed telemetry -> treat as absent, mask off
            return []

    def _parse(self, parser) -> list[ImuSample]:  # pragma: no cover - needs a real GoPro file
        stream = parser.io.extract_gpmf_stream(str(self.path))
        payloads = parser.parse.expand_klv(stream)
        out: list[ImuSample] = []
        for tstamp_ms, gyro_xyz in parser.gyro.iter_gyro(payloads):  # shape depends on lib
            g = np.asarray(gyro_xyz, dtype=float)
            out.append(ImuSample(t_ms=int(tstamp_ms), gyro_dps=(float(g[0]), float(g[1]), float(g[2]))))
        return out
