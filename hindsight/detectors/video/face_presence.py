"""D3 face_presence — presence only, no identity. 1 Hz cadence, zero-order hold.

Backend: MediaPipe Face Detection (self-contained) or OpenCV DNN res10 SSD (needs the
prototxt+caffemodel on disk). Presence-only score = f(max_conf, n_faces, largest_face_area).

If no backend is available it returns None every window -> the detector is availability-
masked off, NOT scored 0 (contracts.py / CLAUDE.md §3). That is the honest behaviour: we
never fabricate a face score we could not compute.
"""

from __future__ import annotations

import numpy as np

from ..._deps import optional
from ...contracts import Window


class FacePresenceDetector:
    name = "face_presence"
    modality = "video"

    def __init__(self, cadence: int = 24, backend: str = "opencv_dnn",
                 min_confidence: float = 0.6, area_weight: float = 0.3) -> None:
        self.cadence = cadence
        self.backend = backend
        self.min_confidence = min_confidence
        self.area_weight = area_weight
        self._impl = self._build()
        self.available = self._impl is not None

    def _build(self):
        mp = optional("mediapipe")
        if mp is not None:
            try:
                fd = mp.solutions.face_detection.FaceDetection(min_detection_confidence=self.min_confidence)
                return ("mediapipe", fd)
            except Exception:  # noqa: BLE001
                return None
        # OpenCV DNN needs model files we do not bundle; only usable if present on disk.
        return None

    def score(self, w: Window) -> float | None:
        if self._impl is None or not w.frames:
            return None
        kind, impl = self._impl
        rgb = w.frames[len(w.frames) // 2].rgb
        if kind == "mediapipe":
            res = impl.process(rgb)
            dets = res.detections or []
            if not dets:
                return 0.0
            confs = [d.score[0] for d in dets]
            areas = [
                d.location_data.relative_bounding_box.width
                * d.location_data.relative_bounding_box.height
                for d in dets
            ]
            max_conf = max(confs)
            largest = float(np.clip(max(areas), 0, 1))
            return float(np.clip((1 - self.area_weight) * max_conf + self.area_weight * largest, 0, 1))
        return None
