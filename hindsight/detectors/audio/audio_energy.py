"""D6 audio_energy — short-time RMS in dBFS through a robust causal normaliser.

Absolute levels vary wildly between mics, so we normalise against a *running* 5th/95th
percentile rather than a fixed dBFS threshold (M2). Causal: percentiles use only
samples seen so far.
"""

from __future__ import annotations

import numpy as np

from ...contracts import Window
from ..base import RunningPercentileNormalizer


def rms_dbfs(pcm: np.ndarray, silence_floor_dbfs: float) -> float:
    """Return PCM16 RMS in dBFS, using a configured finite value for digital silence."""
    if pcm.size == 0:
        return float(silence_floor_dbfs)
    samples = np.asarray(pcm, dtype=np.float64)
    rms = float(np.sqrt(np.mean(samples * samples)))
    if rms == 0.0:
        return float(silence_floor_dbfs)
    full_scale = float(-np.iinfo(np.int16).min)
    return max(float(silence_floor_dbfs), float(20.0 * np.log10(rms / full_scale)))


class AudioEnergyDetector:
    name = "audio_energy"
    modality = "audio"

    def __init__(
        self,
        cadence: int,
        norm_percentiles: tuple[float, float],
        silence_floor_dbfs: float,
    ) -> None:
        if silence_floor_dbfs > 0.0:
            raise ValueError("silence_floor_dbfs must be <= 0 (0 dBFS = digital full scale)")
        lo, hi = norm_percentiles
        if not 0.0 <= lo < hi <= 100.0:
            raise ValueError("audio_energy.norm_percentiles must be 0 <= lo < hi <= 100")
        self.cadence = cadence
        self._norm = RunningPercentileNormalizer(*norm_percentiles)
        self.silence_floor_dbfs = float(silence_floor_dbfs)

    def score(self, w: Window) -> float | None:
        if not w.chunks:
            return None
        pcm = np.concatenate([c.pcm for c in w.chunks])
        if pcm.size == 0:
            return None
        dbfs = rms_dbfs(pcm, self.silence_floor_dbfs)
        return self._norm.update_and_normalize(dbfs)
