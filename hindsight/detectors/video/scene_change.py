"""D1 scene_change — HSV histogram correlation drop between sampled frames.

Downscale to 160x90, 3-D HSV histogram, cv2.HISTCMP_CORREL between consecutive windows;
score = the *drop* in correlation, clipped to [0,1]. Pure numpy (cv2 optional). The first
window has no predecessor, so it returns None -> masked (not scored 0).

Tuning note (measured on plane_1.MP4 @ 4 fps, 2026-07-19). `(1 - correl)/2` maps correl in
[-1,1] onto [0,1], so *unrelated* scenes (correl ~ 0) read ~0.5 and a genuine content change
reads ~0.4-0.5, while continuous POV footage stays low (measured p90 = 0.21 on plane_1, no hard
cuts -> correctly quiet). Sweeps that were run and rejected: finer HSV bins (12^3/16^3) raise absolute
magnitudes but scale signal and floor proportionally (no SNR gain) at 8x the histogram cost; the
downscale size is irrelevant because a normalised histogram is scale-invariant. Hence the
(8,8,8) / 160x90 defaults are retained. On a neutral palette (e.g. an aircraft cabin) this cue is
inherently weak — that is a property of colour-histogram scene detection, not a bug.
"""

from __future__ import annotations

import numpy as np

from ...contracts import Window
from ..base import downscale, to_hsv01


class SceneChangeDetector:
    name = "scene_change"
    modality = "video"

    def __init__(self, cadence: int = 1, downscale_wh: tuple[int, int] = (160, 90),
                 hist_bins: tuple[int, int, int] = (8, 8, 8)) -> None:
        self.cadence = cadence
        self._wh = tuple(downscale_wh)
        self._bins = tuple(hist_bins)
        self._prev_hist: np.ndarray | None = None

    def _hist(self, rgb: np.ndarray) -> np.ndarray:
        small = downscale(rgb, self._wh)
        hsv = to_hsv01(small).reshape(-1, 3)
        h, _ = np.histogramdd(hsv, bins=self._bins, range=[(0, 1), (0, 1), (0, 1)])
        h = h.astype(np.float64).ravel()
        s = h.sum()
        return h / s if s > 0 else h

    @staticmethod
    def _correl(h1: np.ndarray, h2: np.ndarray) -> float:
        # cv2.HISTCMP_CORREL, replicated exactly.
        a, b = h1 - h1.mean(), h2 - h2.mean()
        denom = np.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / denom) if denom > 1e-12 else 1.0

    def score(self, w: Window) -> float | None:
        if not w.frames:
            return None
        rep = w.frames[len(w.frames) // 2].rgb
        hist = self._hist(rep)
        if self._prev_hist is None:
            self._prev_hist = hist
            return None
        d = self._correl(self._prev_hist, hist)
        self._prev_hist = hist
        return float(np.clip((1.0 - d) / 2.0, 0.0, 1.0))
