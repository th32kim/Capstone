"""D2 motion — EGO-COMPENSATED residual motion (DESIGN_DELTAS D-2).

First-person video is head-mounted; a naive frame-diff fires the entire time the wearer
walks. So: estimate the dominant global translation (phase correlation), warp it out, then
diff. The score is *residual* motion — object/person motion with head motion removed —
robust-normalised (running 5th/95th, causal).

The raw global-motion magnitude is kept (`raw_global_motion`, last value + history) because
D7 (dwell) needs it: dwell = that magnitude dropping below threshold after a spell above it.

`cv2.phaseCorrelate` (subpixel) is used when OpenCV is present; otherwise a pure-numpy FFT
phase correlation gives an integer-pixel shift. Both are genuinely ego-compensating, but they
are NOT equivalent — validated on plane_1.MP4 (4K POV, DESIGN_DELTAS D-2): cv2's subpixel peak,
even after rounding to an integer roll, yields ~35 % lower mean residual than the numpy path
(15.6 vs 24.8) because its centroid refinement lands a better integer shift. OpenCV is a core
dependency; the numpy path is a genuine but weaker fallback. Two things measured and rejected on
that footage, so they are deliberately absent: (i) a Hanning window on the FFT — no accuracy gain,
marginally worse; (ii) a subpixel `warpAffine` instead of the integer `np.roll` — <3 % residual
change. The real limiter is that consecutive windows sample frames ~0.5 s apart (D2 sees one frame
per 0.5 s window, so its effective rate is 2 Hz regardless of decode fps): at that spacing head
motion is large and non-translational, which caps how much any translational model can compensate.
"""

from __future__ import annotations

import numpy as np

from ..._deps import optional
from ...contracts import Window
from ..base import RunningPercentileNormalizer, downscale, to_gray


def _numpy_phase_shift(a: np.ndarray, b: np.ndarray) -> tuple[int, int]:
    """Integer (dy, dx) translation aligning b onto a, via FFT phase correlation."""
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
    return int(dy), int(dx)


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
        # numpy-computed scores (they disagree on the integer shift ~88% of the time -> materially
        # different residuals; DESIGN_DELTAS D-2). Same "be loud about the fallback" contract as VAD.
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

        warped_prev = np.roll(np.roll(prev, dy, axis=0), dx, axis=1)
        # exclude the wrapped border so np.roll artefacts do not inflate the residual
        by, bx = min(abs(dy), gray.shape[0] // 4), min(abs(dx), gray.shape[1] // 4)
        core = np.s_[by : gray.shape[0] - by if by else gray.shape[0],
                     bx : gray.shape[1] - bx if bx else gray.shape[1]]
        residual = float(np.mean(np.abs(gray[core] - warped_prev[core])))

        self.raw_global_motion = float(np.hypot(dy, dx))
        self.global_motion_history.append(self.raw_global_motion)
        return self._norm.update_and_normalize(residual)
