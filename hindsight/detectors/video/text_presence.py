"""D4 text_presence — on-screen text density. 0.5 Hz cadence, zero-order hold.

Backend: EAST text detector (needs the frozen .pb model on disk -- not bundled, see
`east_model_path`) or MSER + a stroke-width-ish heuristic (OpenCV, no model file).
Score = f(box count, text-area fraction, mean conf). All scoring constants come from
config (CLAUDE.md §1: no hard-coded thresholds).

The backend is decided ONCE, at construction, and never changes mid-run: the EAST and
MSER score formulas are incommensurable, so one run's score column must come from exactly
one of them. `backend: east` with no usable model honestly falls back to MSER and records
*why* in `self.fallback_reason` (surfaced via notes + `hindsight detect`); a per-window
EAST inference error masks that window (never a silent formula switch), and persistent
errors disable further attempts for the run.

No OpenCV -> returns None -> availability-masked off (never scored 0). FS8/OCR is
non-essential; a missing backend here must not silently drag salience down.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..._deps import optional
from ...config import REPO_ROOT
from ...contracts import Window
from ..base import to_gray

# Frozen-model constants (tied to the EAST architecture / its training preprocessing,
# not tunables): output layer names, the ImageNet channel means blobFromImage subtracts
# (RGB order — our frames are RGB and swapRB=False), and the 4-px output stride.
_EAST_OUTPUT_LAYERS = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
_EAST_CHANNEL_MEANS = (123.68, 116.78, 103.94)
_EAST_OUTPUT_STRIDE = 4.0

_BACKENDS = ("east", "mser_swt")


def _decode_east(
    scores: np.ndarray, geometry: np.ndarray, min_confidence: float
) -> tuple[list[tuple[int, int, int, int]], list[float]]:
    """Standard EAST output decode (vectorised): per-cell rotated box + confidence."""
    conf_map = scores[0, 0]                       # (rows, cols)
    ys, xs = np.nonzero(conf_map >= min_confidence)
    if ys.size == 0:
        return [], []
    geo = geometry[0]                             # (5, rows, cols)
    x0, x1, x2, x3, angles = (geo[i][ys, xs] for i in range(5))
    cos, sin = np.cos(angles), np.sin(angles)
    h = x0 + x2
    w = x1 + x3
    off_x = xs * _EAST_OUTPUT_STRIDE
    off_y = ys * _EAST_OUTPUT_STRIDE
    end_x = (off_x + cos * x1 + sin * x2).astype(np.int64)   # trunc-toward-zero == int()
    end_y = (off_y - sin * x1 + cos * x2).astype(np.int64)
    start_x = (end_x - w).astype(np.int64)
    start_y = (end_y - h).astype(np.int64)
    boxes = list(zip(start_x.tolist(), start_y.tolist(),
                     w.astype(np.int64).tolist(), h.astype(np.int64).tolist()))
    confidences = conf_map[ys, xs].astype(float).tolist()
    return boxes, confidences


class TextPresenceDetector:
    name = "text_presence"
    modality = "video"

    def __init__(self, cadence: int = 48, backend: str = "east", min_confidence: float = 0.5,
                 east_model_path: str | None = None,
                 east_input_size: int = 320, east_nms_iou: float = 0.4,
                 east_count_saturation: float = 20.0,
                 east_weights: tuple[float, float, float] = (0.4, 0.3, 0.3),
                 east_max_consecutive_errors: int = 3,
                 mser_count_saturation: float = 40.0,
                 mser_weights: tuple[float, float] = (0.6, 0.4)) -> None:
        if backend not in _BACKENDS:
            raise ValueError(f"text_presence.backend must be one of {_BACKENDS}, got {backend!r}")
        if east_input_size % 32 != 0:
            raise ValueError(f"east.input_size must be a multiple of 32, got {east_input_size}")
        if len(east_weights) != 3:
            raise ValueError(f"east.weights needs 3 values (count/area/conf), got {east_weights!r}")
        if len(mser_weights) != 2:
            raise ValueError(f"mser.weights needs 2 values (count/area), got {mser_weights!r}")
        self.cadence = cadence
        self.backend = backend                     # configured backend
        self.min_confidence = min_confidence
        self.east_input_size = east_input_size
        self.east_nms_iou = east_nms_iou
        self.east_count_saturation = east_count_saturation
        self.east_weights = east_weights
        self.east_max_consecutive_errors = east_max_consecutive_errors
        self.mser_count_saturation = mser_count_saturation
        self.mser_weights = mser_weights
        self._cv2 = optional("cv2")
        self.available = self._cv2 is not None
        self.east_errors = 0                       # windows masked on EAST inference errors
        self._consecutive_errors = 0
        self._east_disabled = False
        self.fallback_reason: str | None = None    # why MSER runs although east was configured
        self._east_net = None
        if not self.available:
            self.last_backend = "unavailable"
            return
        if backend == "east":
            self._east_net, self.fallback_reason = self._load_east(east_model_path)
        # honesty: the backend whose formula produces this run's scores. Fixed for the run.
        self.last_backend = "east" if self._east_net is not None else "mser_swt"

    def _load_east(self, path: str | None) -> tuple[object | None, str | None]:
        """Load the EAST net, or return (None, reason) — the reason is user-facing."""
        if not path:
            return None, "east_model_path not set"
        p = Path(path)
        if not p.is_absolute():
            p = REPO_ROOT / p                      # anchor to the repo, not the process CWD
        if not p.exists():
            return None, f"EAST model file not found: {p}"
        try:
            return self._cv2.dnn.readNet(str(p)), None
        except Exception as exc:  # noqa: BLE001
            return None, f"EAST model failed to load ({type(exc).__name__}): {p}"

    def _score_mser(self, cv2, gray: np.ndarray) -> float | None:
        try:
            mser = cv2.MSER_create()
            regions, _ = mser.detectRegions(gray)
        except Exception:  # noqa: BLE001
            return None
        if not regions:
            return 0.0
        h, wdt = gray.shape
        total_area = float(h * wdt)
        text_like = 0
        covered = 0.0
        for r in regions:
            x, y, bw, bh = cv2.boundingRect(r.reshape(-1, 1, 2))
            aspect = bw / (bh + 1e-6)
            fill = len(r) / (bw * bh + 1e-6)
            if 0.1 < aspect < 15 and fill > 0.2 and bh < h * 0.5:  # text-like glyph blobs
                text_like += 1
                covered += bw * bh
        count_score = min(text_like / self.mser_count_saturation, 1.0)
        area_score = min(covered / total_area, 1.0)
        w_count, w_area = self.mser_weights
        return float(np.clip(w_count * count_score + w_area * area_score, 0, 1))

    def _score_east(self, cv2, rgb: np.ndarray) -> float | None:
        # the whole path is guarded: a wrong-architecture .pb that loads but emits
        # unexpected shapes must mask the window, not crash the detector bank.
        size = self.east_input_size
        try:
            blob = cv2.dnn.blobFromImage(
                rgb, 1.0, (size, size), _EAST_CHANNEL_MEANS, swapRB=False, crop=False)
            self._east_net.setInput(blob)
            scores, geometry = self._east_net.forward(_EAST_OUTPUT_LAYERS)
            boxes, confidences = _decode_east(scores, geometry, self.min_confidence)
            if not boxes:
                return 0.0
            idx = cv2.dnn.NMSBoxes(boxes, confidences, self.min_confidence, self.east_nms_iou)
            idx = np.asarray(idx).reshape(-1)
            if idx.size == 0:
                return 0.0
            # union coverage on the network's input grid, clipped to it: overlapping boxes
            # are not double-counted and off-frame extents do not inflate the area term.
            canvas = np.zeros((size, size), dtype=bool)
            for i in idx:
                x, y, bw, bh = boxes[i]
                x0c, y0c = max(x, 0), max(y, 0)
                x1c, y1c = min(x + bw, size), min(y + bh, size)
                if x1c > x0c and y1c > y0c:
                    canvas[y0c:y1c, x0c:x1c] = True
            count_score = min(idx.size / self.east_count_saturation, 1.0)
            area_score = float(canvas.mean())
            mean_conf = float(np.mean([confidences[i] for i in idx]))
            w_count, w_area, w_conf = self.east_weights
            return float(np.clip(
                w_count * count_score + w_area * area_score + w_conf * mean_conf, 0, 1))
        except Exception:  # noqa: BLE001
            return None

    def score(self, w: Window) -> float | None:
        if self._cv2 is None or not w.frames:
            return None
        cv2 = self._cv2
        rgb = w.frames[len(w.frames) // 2].rgb
        if self.last_backend == "east":
            if self._east_disabled:
                return None                        # persistent-failure state: stay masked
            result = self._score_east(cv2, rgb)
            if result is None:
                # inference errored: mask THIS window rather than switch formulas mid-run
                self.east_errors += 1
                self._consecutive_errors += 1
                if self._consecutive_errors >= self.east_max_consecutive_errors:
                    self._east_disabled = True     # stop paying blob+forward for a dead net
            else:
                self._consecutive_errors = 0
            return result
        gray = to_gray(rgb).astype(np.uint8)
        return self._score_mser(cv2, gray)
