"""D6 audio_energy — short-time RMS in dBFS through a robust causal normaliser.

Absolute levels vary wildly between mics, so we normalise against a *running* 5th/95th
percentile rather than a fixed dBFS threshold (M2). Causal: percentiles use only
samples seen so far.
"""

from __future__ import annotations

import numpy as np

from ...contracts import Window
from ..base import RunningPercentileNormalizer


class AudioEnergyDetector:
    name = "audio_energy"
    modality = "audio"

    def __init__(self, cadence: int = 1, norm_percentiles: tuple[float, float] = (5, 95)) -> None:
        self.cadence = cadence
        self._norm = RunningPercentileNormalizer(*norm_percentiles)

    def score(self, w: Window) -> float | None:
        if not w.chunks:
            return None
        pcm = np.concatenate([c.pcm for c in w.chunks]).astype(np.float64)
        rms = float(np.sqrt(np.mean(pcm**2))) if pcm.size else 0.0
        dbfs = 20.0 * np.log10(rms / 32768.0 + 1e-9)  # [-180, 0]
        return self._norm.update_and_normalize(dbfs)
