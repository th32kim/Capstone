"""D4 text_presence — on-screen text density. 0.5 Hz cadence, zero-order hold.

Backend: EAST text detector (needs the model on disk) or MSER + a stroke-width-ish
heuristic (OpenCV, no model file). Score = f(box count, text-area fraction, mean conf).

No OpenCV -> returns None -> availability-masked off (never scored 0). FS8/OCR is
non-essential; a missing backend here must not silently drag salience down.
"""

from __future__ import annotations

import numpy as np

from ..._deps import optional
from ...contracts import Window
from ..base import to_gray


class TextPresenceDetector:
    name = "text_presence"
    modality = "video"

    def __init__(self, cadence: int = 48, backend: str = "east", min_confidence: float = 0.5) -> None:
        self.cadence = cadence
        self.backend = backend
        self.min_confidence = min_confidence
        self._cv2 = optional("cv2")
        self.available = self._cv2 is not None

    def score(self, w: Window) -> float | None:
        if self._cv2 is None or not w.frames:
            return None
        cv2 = self._cv2
        gray = to_gray(w.frames[len(w.frames) // 2].rgb).astype(np.uint8)
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
