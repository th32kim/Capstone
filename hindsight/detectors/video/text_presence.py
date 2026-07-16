"""D4 text_presence — on-screen text density. 0.5 Hz cadence, zero-order hold.

Backend: EAST text detector (needs the frozen .pb model on disk -- not bundled, see
`east_model_path`) or MSER + a stroke-width-ish heuristic (OpenCV, no model file).
Score = f(box count, text-area fraction, mean conf).

`backend: east` with no model file on disk honestly falls back to MSER and records which
backend actually ran in `self.active_backend` (mirrors the VAD energy-fallback pattern) --
it never silently pretends EAST ran (CLAUDE.md §1).

No OpenCV -> returns None -> availability-masked off (never scored 0). FS8/OCR is
non-essential; a missing backend here must not silently drag salience down.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..._deps import optional
from ...contracts import Window
from ..base import to_gray

_EAST_INPUT_SIZE = 320  # must be a multiple of 32
_EAST_OUTPUT_LAYERS = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]


def _decode_east(scores: np.ndarray, geometry: np.ndarray, min_confidence: float):
    """Standard EAST output decode: per-cell rotated box + confidence."""
    num_rows, num_cols = scores.shape[2:4]
    boxes, confidences = [], []
    for y in range(num_rows):
        scores_row = scores[0, 0, y]
        x0, x1, x2, x3 = geometry[0, 0, y], geometry[0, 1, y], geometry[0, 2, y], geometry[0, 3, y]
        angles = geometry[0, 4, y]
        for x in range(num_cols):
            conf = float(scores_row[x])
            if conf < min_confidence:
                continue
            offset_x, offset_y = x * 4.0, y * 4.0
            angle = angles[x]
            cos, sin = np.cos(angle), np.sin(angle)
            h, w = x0[x] + x2[x], x1[x] + x3[x]
            end_x = int(offset_x + cos * x1[x] + sin * x2[x])
            end_y = int(offset_y - sin * x1[x] + cos * x2[x])
            boxes.append((int(end_x - w), int(end_y - h), int(w), int(h)))
            confidences.append(conf)
    return boxes, confidences


class TextPresenceDetector:
    name = "text_presence"
    modality = "video"

    def __init__(self, cadence: int = 48, backend: str = "east", min_confidence: float = 0.5,
                 east_model_path: str | None = None) -> None:
        self.cadence = cadence
        self.backend = backend
        self.min_confidence = min_confidence
        self._cv2 = optional("cv2")
        self.available = self._cv2 is not None
        self.active_backend: str | None = None  # honesty: what actually ran, set on first score()
        self._east_net = self._load_east(east_model_path) if backend == "east" else None

    def _load_east(self, path: str | None):
        if self._cv2 is None or not path:
            return None
        p = Path(path)
        if not p.exists():
            return None
        try:
            return self._cv2.dnn.readNet(str(p))
        except Exception:  # noqa: BLE001
            return None

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
        count_score = min(text_like / 40.0, 1.0)      # ~40 glyphs => saturated
        area_score = min(covered / total_area, 1.0)
        return float(np.clip(0.6 * count_score + 0.4 * area_score, 0, 1))

    def _score_east(self, cv2, rgb: np.ndarray) -> float | None:
        h, w = rgb.shape[:2]
        blob = cv2.dnn.blobFromImage(
            rgb, 1.0, (_EAST_INPUT_SIZE, _EAST_INPUT_SIZE),
            (123.68, 116.78, 103.94), swapRB=False, crop=False)
        self._east_net.setInput(blob)
        try:
            scores, geometry = self._east_net.forward(_EAST_OUTPUT_LAYERS)
        except Exception:  # noqa: BLE001
            return None
        boxes, confidences = _decode_east(scores, geometry, self.min_confidence)
        if not boxes:
            return 0.0
        idx = cv2.dnn.NMSBoxes(boxes, confidences, self.min_confidence, 0.4)
        idx = np.array(idx).reshape(-1) if len(idx) else np.array([], dtype=int)
        if idx.size == 0:
            return 0.0
        sx, sy = w / _EAST_INPUT_SIZE, h / _EAST_INPUT_SIZE
        total_area = float(h * w)
        covered = sum(boxes[i][2] * sx * boxes[i][3] * sy for i in idx)
        count_score = min(idx.size / 20.0, 1.0)        # ~20 boxes => saturated
        area_score = min(covered / total_area, 1.0)
        mean_conf = float(np.mean([confidences[i] for i in idx]))
        return float(np.clip(0.4 * count_score + 0.3 * area_score + 0.3 * mean_conf, 0, 1))

    def score(self, w: Window) -> float | None:
        if self._cv2 is None or not w.frames:
            return None
        cv2 = self._cv2
        rgb = w.frames[len(w.frames) // 2].rgb
        if self.backend == "east" and self._east_net is not None:
            result = self._score_east(cv2, rgb)
            if result is not None:
                self.active_backend = "east"
                return result
            # EAST inference errored on this frame -- fall through to MSER rather than mask off.
        gray = to_gray(rgb).astype(np.uint8)
        result = self._score_mser(cv2, gray)
        if result is not None:
            self.active_backend = "mser_swt"
        return result
