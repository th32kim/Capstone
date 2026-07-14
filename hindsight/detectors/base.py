"""Detector base machinery: causal normalisers, cadence + zero-order-hold policy.

Detectors are small and (mostly) pure: `score(window) -> float | None`. Cross-cutting
policy lives here, in one place:

  * **Causal normalisers** (running 5th/95th percentile). CLAUDE.md §3 / M2: normalisers are
    stateful and must be causal — no peeking at the future — or the online claim (FS4) is a
    lie. The test for this is in tests/.
  * **Cadence + zero-order hold**: D3/D4 run at reduced rate and *hold* their last value
    between evaluations; a held value is `stale` and is recorded as such in the mask — we do
    not silently pretend a held value is fresh (CLAUDE.md §3 note).

A detector that returns None means "could not run this window" -> mask bit False (NOT 0.0).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..contracts import DECISION_WINDOW_S, Window


class RunningPercentileNormalizer:
    """Causal robust normaliser: x -> clip((x - p_lo) / (p_hi - p_lo), 0, 1).

    Percentiles are computed over samples seen *so far only* (current included; never
    future). Bounded history keeps it O(1) amortised and drift-adaptive.
    """

    def __init__(self, lo: float = 5.0, hi: float = 95.0, max_history: int = 6000,
                 min_count: int = 4) -> None:
        self.lo, self.hi = lo, hi
        self.max_history = max_history
        self.min_count = min_count
        self._hist: list[float] = []

    def update_and_normalize(self, x: float) -> float:
        self._hist.append(float(x))
        if len(self._hist) > self.max_history:
            self._hist = self._hist[-self.max_history :]
        if len(self._hist) < self.min_count:
            return 0.0
        p_lo, p_hi = np.percentile(self._hist, [self.lo, self.hi])
        if p_hi - p_lo < 1e-9:
            return 0.0
        return float(np.clip((x - p_lo) / (p_hi - p_lo), 0.0, 1.0))


def to_hsv01(rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 (H,W,3) -> HSV float in [0,1]^3. Pure numpy (cv2 not required)."""
    a = rgb.astype(np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = a.max(axis=-1)
    mn = a.min(axis=-1)
    diff = mx - mn
    h = np.zeros_like(mx)
    mask = diff > 1e-8
    # hue
    rc = np.where(mask, (mx - r) / np.where(mask, diff, 1), 0)
    gc = np.where(mask, (mx - g) / np.where(mask, diff, 1), 0)
    bc = np.where(mask, (mx - b) / np.where(mask, diff, 1), 0)
    h = np.where((mx == r) & mask, bc - gc, h)
    h = np.where((mx == g) & mask, 2.0 + rc - bc, h)
    h = np.where((mx == b) & mask, 4.0 + gc - rc, h)
    h = (h / 6.0) % 1.0
    s = np.where(mx > 1e-8, diff / np.where(mx > 1e-8, mx, 1), 0)
    v = mx
    return np.stack([h, s, v], axis=-1)


def downscale(rgb: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    """Downscale to (W,H). Uses cv2.INTER_AREA if available, else numpy nearest-neighbour."""
    from .._deps import optional

    w, h = size_wh
    cv2 = optional("cv2")
    if cv2 is not None:
        return cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    H, W = rgb.shape[:2]
    ys = np.linspace(0, H - 1, h).astype(int)
    xs = np.linspace(0, W - 1, w).astype(int)
    return rgb[ys][:, xs]


def to_gray(rgb: np.ndarray) -> np.ndarray:
    """Rec.601 luma, float32."""
    a = rgb.astype(np.float32)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


@dataclass
class CadencePolicy:
    """Turns a frame-cadence into an evaluate-every-N-windows schedule + zero-order hold.

    cadence is in *frames* (contracts): 1 = every frame, 24 ~ 1 Hz @ 24 fps. Windows are
    0.5 s. eval_every_windows = max(1, round(cadence / (fps * window_s))). Between evals the
    last value is held and flagged stale.
    """

    cadence: int
    fps: float
    window_s: float = DECISION_WINDOW_S
    _eval_every: int = field(init=False)
    _since_eval: int = field(default=0, init=False)
    _held: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        frames_per_window = max(self.fps * self.window_s, 1e-6)
        self._eval_every = max(1, round(self.cadence / frames_per_window))

    def due(self, window_index: int) -> bool:
        return window_index % self._eval_every == 0

    def apply(self, window_index: int, fresh: float | None) -> tuple[float | None, bool]:
        """Return (value, stale). If not due, hold last value and mark stale."""
        if self.due(window_index):
            if fresh is not None:
                self._held = fresh
                return fresh, False
            return (self._held, self._held is not None) if self._held is not None else (None, False)
        if self._held is None:
            return None, False  # never evaluated yet -> unavailable, not stale-zero
        return self._held, True
