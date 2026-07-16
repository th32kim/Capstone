"""D3 face_presence — presence only, no identity. 1 Hz cadence, zero-order hold.

Backend (first that is available): MediaPipe Face Detection (self-contained), or OpenCV's
YuNet detector (a small ONNX model on disk). Presence-only score = f(max_conf, largest_face_area).

Backend substitution (DESIGN_DELTAS D-8): the report named "res10 SSD or MediaPipe". On the
current stack neither is usable — OpenCV 5 removed the Caffe importer (`readNetFromCaffe` is
gone, so the res10 caffemodel cannot load) and the arm64-macOS mediapipe wheel ships only the
Tasks API (no `mp.solutions`). YuNet is OpenCV's own, officially-recommended res10 successor
(`cv2.FaceDetectorYN`), so it is the forced substitute. Model path comes from config, not code.

If no backend is available it returns None every window -> the detector is availability-
masked off, NOT scored 0 (contracts.py / CLAUDE.md §3). That is the honest behaviour: we
never fabricate a face score we could not compute.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..._deps import optional
from ...contracts import Window


class FacePresenceDetector:
    name = "face_presence"
    modality = "video"

    def __init__(self, cadence: int = 24, backend: str = "opencv_dnn",
                 min_confidence: float = 0.6, area_weight: float = 0.3,
                 model_path: str | None = None) -> None:
        self.cadence = cadence
        self.backend = backend
        self.min_confidence = min_confidence
        self.area_weight = area_weight
        self.model_path = model_path
        self._impl = self._build()
        self.available = self._impl is not None

    def _yunet(self):
        """Build OpenCV YuNet if cv2 + the ONNX model on disk are both available, else None."""
        cv2 = optional("cv2")
        if cv2 is None or not self.model_path or not Path(self.model_path).exists():
            return None
        try:
            det = cv2.FaceDetectorYN.create(self.model_path, "", (320, 320),
                                            float(self.min_confidence), 0.3, 5000)
            return ("yunet", det)
        except Exception:  # noqa: BLE001
            return None

    def _build(self):
        # honour an explicit opencv/yunet backend first; otherwise prefer the self-contained
        # mediapipe legacy API if present; fall back to YuNet if a model is on disk.
        if self.backend in ("yunet", "opencv_dnn", "opencv"):
            impl = self._yunet()
            if impl is not None:
                return impl
        mp = optional("mediapipe")
        if mp is not None:
            try:
                fd = mp.solutions.face_detection.FaceDetection(min_detection_confidence=self.min_confidence)
                return ("mediapipe", fd)
            except Exception:  # noqa: BLE001 — e.g. arm64 wheel with no mp.solutions
                pass
        return self._yunet()

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
        if kind == "yunet":
            cv2 = optional("cv2")
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            h, w = bgr.shape[:2]
            impl.setInputSize((w, h))
            _, faces = impl.detect(bgr)
            if faces is None or len(faces) == 0:
                return 0.0
            # YuNet row: [x, y, w, h, 5x2 landmarks, score]; area fraction over the frame.
            max_conf = float(np.clip(faces[:, -1].max(), 0, 1))
            largest = float(np.clip((faces[:, 2] * faces[:, 3]).max() / float(w * h), 0, 1))
            return float(np.clip((1 - self.area_weight) * max_conf + self.area_weight * largest, 0, 1))
        return None
