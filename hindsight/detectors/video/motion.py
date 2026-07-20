"""D2 motion — EGO-COMPENSATED residual motion (DESIGN_DELTAS D-2).

First-person video is head-mounted; a naive frame-diff fires the entire time the wearer
walks. So: estimate the dominant global translation (phase correlation), shift it out, then
diff the OVERLAPPING region only. The score is *residual* motion — object/person motion with
head motion removed — robust-normalised (running 5th/95th, causal).

The raw global-motion magnitude is kept (`raw_global_motion`, last value + history) because
D7 (dwell) needs it: dwell = that magnitude dropping below threshold after a spell above it.

`cv2.phaseCorrelate` (subpixel) is used when OpenCV is present; otherwise a pure-numpy FFT
phase correlation gives an integer-pixel shift. Both paths use the SAME sign convention (the
shift that maps `prev` onto `gray`) and both genuinely ego-compensate; cv2's subpixel peak is
marginally more accurate (its centroid refinement can land a better integer), so cv2 is the
report-grade default and the numpy path is a faithful integer-only fallback. Which one ran is
recorded in `last_backend` so cv2- and numpy-computed scores are never silently mixed.

Residual is measured on the VALID OVERLAP after the shift (not `np.roll` + a fixed border
crop): a pan reveals new scene at the leading edge, and comparing only the overlapping pixels
means neither the newly-revealed band nor any wrap-around artefact can inflate the residual —
which matters most for large shifts (fast head turns), exactly when ego-motion is largest.

The real limiter is temporal: consecutive windows sample frames ~0.5 s apart (D2 sees one
frame per 0.5 s window, so its effective rate is 2 Hz regardless of decode fps). At that
spacing head motion is large and non-translational (rotation/parallax/blur), which caps how
much any translational model can compensate — see DESIGN_DELTAS D-2 for the measured ceiling
and the "difference adjacent decoded frames" proposal that follows from it.
"""

from __future__ import annotations

import numpy as np

from ..._deps import optional
from ...contracts import Window
from ..base import RunningPercentileNormalizer, downscale, to_gray


def _numpy_phase_shift(a: np.ndarray, b: np.ndarray) -> tuple[int, int]:
    """Integer (dy, dx) shift that maps `a` onto `b`, via FFT phase correlation.

    Sign convention matches cv2.phaseCorrelate(a, b): the returned (dy, dx) is the shift such
    that translating `a` by it aligns `a` with `b`. The raw cross-power argmax yields the
    opposite sign (the shift from b to a), so it is negated here — without this, the fallback
    would ANTI-compensate a pan and leave residual ~ uncompensated.
    """
    fa, fb = np.fft.fft2(a), np.fft.fft2(b)
    r = fa * np.conj(fb)
    r /= np.abs(r) + 1e-8
    corr = np.fft.ifft2(r).real
    dy, dx = np.unravel_index(int(np.argmax(corr)), corr.shape)
    h, w = a.shape
    if dy > h // 2:
        dy -= h
    if dx > w // 2:
        dx -= w
    return -int(dy), -int(dx)


def _overlap_residual(prev: np.ndarray, gray: np.ndarray, dy: int, dx: int) -> float:
    """Mean abs difference over the region where `prev` shifted by (dy, dx) overlaps `gray`.

    gray[i, j] is compared against prev[i - dy, j - dx] for every (i, j) where both indices are
    in bounds. No np.roll, so there is no wrap-around band to exclude. Falls back to the full
    (uncompensated) frame diff if the shift is so large there is no overlap left.
    """
    h, w = gray.shape
    gy0, gy1 = max(0, dy), h + min(0, dy)
    gx0, gx1 = max(0, dx), w + min(0, dx)
    if gy1 - gy0 <= 0 or gx1 - gx0 <= 0:
        return float(np.mean(np.abs(gray - prev)))  # shift ≥ frame: cannot compensate
    g = gray[gy0:gy1, gx0:gx1]
    p = prev[gy0 - dy:gy1 - dy, gx0 - dx:gx1 - dx]
    return float(np.mean(np.abs(g - p)))


class MotionDetector:
    name = "motion"
    modality = "video"

    def __init__(self, cadence: int = 1, downscale_wh: tuple[int, int] = (160, 90),
                 ego_compensate: bool = True, norm_percentiles: tuple[float, float] = (5, 95)) -> None:
        self.cadence = cadence
        self._wh = tuple(downscale_wh)
        self.ego_compensate = ego_compensate
        self._prev: np.ndarray | None = None
        self._norm = RunningPercentileNormalizer(*norm_percentiles)
        self.raw_global_motion: float = 0.0
        self.global_motion_history: list[float] = []
        # Which ego-motion path ran, recorded so the runner/eval never SILENTLY mixes cv2- and
        # numpy-computed scores (subpixel vs integer -> slightly different residuals;
        # DESIGN_DELTAS D-2). Same "be loud about the fallback" contract as VAD.
        self.last_backend = "uninitialised"

    def score(self, w: Window) -> float | None:
        if not w.frames:
            return None
        cur = downscale(w.frames[len(w.frames) // 2].rgb, self._wh)
        gray = to_gray(cur)
        if self._prev is None:
            self._prev = gray
            return None
        prev = self._prev
        self._prev = gray

        dy = dx = 0
        if self.ego_compensate:
            cv2 = optional("cv2")
            if cv2 is not None:
                (sx, sy), _ = cv2.phaseCorrelate(prev.astype(np.float64), gray.astype(np.float64))
                dx, dy = int(round(sx)), int(round(sy))
                self.last_backend = "cv2_phasecorr"
            else:
                dy, dx = _numpy_phase_shift(prev, gray)
                self.last_backend = "numpy_phasecorr"
        else:
            self.last_backend = "raw_framediff"

        residual = _overlap_residual(prev, gray, dy, dx)
        self.raw_global_motion = float(np.hypot(dy, dx))
        self.global_motion_history.append(self.raw_global_motion)
        return self._norm.update_and_normalize(residual)
