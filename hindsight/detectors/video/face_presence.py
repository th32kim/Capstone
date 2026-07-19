"""D3 face_presence — presence only, no identity. 1 Hz cadence, zero-order hold.

Backend: MediaPipe Face Detection (self-contained; the only implemented backend) or
OpenCV DNN res10 SSD (config-selectable but NOT implemented — masks off with a recorded
reason). Presence-only score = f(max_conf, n_faces, largest_face_area).

If no backend is available it returns None every window -> the detector is availability-
masked off, NOT scored 0 (contracts.py / CLAUDE.md §3). That is the honest behaviour: we
never fabricate a face score we could not compute.
"""

from __future__ import annotations

import numpy as np

from ..._deps import optional
from ...contracts import Window


_BACKENDS = ("mediapipe", "opencv_dnn")


class FacePresenceDetector:
    name = "face_presence"
    modality = "video"

    def __init__(self, cadence: int = 24, backend: str = "mediapipe",
                 min_confidence: float = 0.6, area_weight: float = 0.3) -> None:
        if backend not in _BACKENDS:
            raise ValueError(f"face_presence.backend must be one of {_BACKENDS}, got {backend!r}")
        self.cadence = cadence
        self.backend = backend
        self.min_confidence = min_confidence
        self.area_weight = area_weight
        self._impl, self.unavailable_reason = self._build()
        self.available = self._impl is not None
        self.last_backend = self.backend if self.available else "unavailable"

    def _build(self) -> tuple[tuple[str, object] | None, str | None]:
        """Build the configured backend, or return (None, reason) — the reason is user-facing."""
        if self.backend == "mediapipe":
            mp = optional("mediapipe")
            if mp is None:
                return None, "mediapipe not installed"
            if not hasattr(mp, "solutions"):
                # mediapipe >= 0.10.30 removed the legacy solutions API (see requirements.txt);
                # say so instead of letting the AttributeError vanish into a generic mask-off.
                ver = getattr(mp, "__version__", "unknown")
                return None, (f"mediapipe {ver} lacks the legacy `mp.solutions` API "
                              "(removed upstream; install the pinned range in requirements.txt)")
            try:
                fd = mp.solutions.face_detection.FaceDetection(min_detection_confidence=self.min_confidence)
                return ("mediapipe", fd), None
            except Exception as exc:  # noqa: BLE001
                return None, f"mediapipe FaceDetection failed to build ({type(exc).__name__}: {exc})"
        # opencv_dnn: the res10 SSD loader is NOT implemented — this masks off regardless of
        # whether the prototxt+caffemodel are on disk. Honest stub, honestly reported.
        return None, "opencv_dnn backend not implemented (res10 SSD loader absent)"

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
