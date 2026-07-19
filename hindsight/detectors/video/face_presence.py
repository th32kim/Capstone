"""D3 face_presence — presence only, no identity. 1 Hz cadence, zero-order hold.

Backends (config-selected, validated — the configured backend is the one that runs, D-8):

* `mediapipe` — MediaPipe Face Detection via the legacy `mp.solutions` API (self-contained,
  needs the pinned mediapipe range from requirements.txt).
* `yunet` — OpenCV's YuNet (`cv2.FaceDetectorYN`, a ~230 KB ONNX model on disk, see
  `model_path`). Substituted for the report's res10 SSD (DESIGN_DELTAS D-9): OpenCV 5
  removed the Caffe importer, so the res10 caffemodel cannot load at all; YuNet is OpenCV's
  officially-recommended successor. `opencv_dnn` / `opencv` are accepted as aliases.

Presence-only score = f(max_conf, largest_face_area) — identical formula for both backends,
so the fusion contract does not depend on which one ran.

If the configured backend cannot build, the detector returns None every window -> it is
availability-masked off, NOT scored 0 (contracts.py / CLAUDE.md §3), and the cause is
recorded in `unavailable_reason` (surfaced by `hindsight detect`). That is the honest
behaviour: we never fabricate a face score we could not compute, and we never silently run
a different backend than the config claims.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..._deps import optional
from ...config import REPO_ROOT
from ...contracts import Window

_BACKENDS = ("mediapipe", "yunet", "opencv_dnn", "opencv")
_YUNET_ALIASES = ("yunet", "opencv_dnn", "opencv")


class FacePresenceDetector:
    name = "face_presence"
    modality = "video"

    def __init__(self, cadence: int = 24, backend: str = "mediapipe",
                 min_confidence: float = 0.6, area_weight: float = 0.3,
                 model_path: str | None = None) -> None:
        if backend not in _BACKENDS:
            raise ValueError(f"face_presence.backend must be one of {_BACKENDS}, got {backend!r}")
        self.cadence = cadence
        # normalise the res10-era aliases: on OpenCV 5 the opencv face path IS YuNet (D-9)
        self.backend = "yunet" if backend in _YUNET_ALIASES else backend
        self.min_confidence = min_confidence
        self.area_weight = area_weight
        self.model_path = model_path
        self._cv2 = optional("cv2")
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
                # mediapipe >= 0.10.30 removed the legacy solutions API (see requirements.txt),
                # and some platform wheels (e.g. arm64 macOS) ship only the Tasks API; say so
                # instead of letting the AttributeError vanish into a generic mask-off.
                ver = getattr(mp, "__version__", "unknown")
                return None, (f"mediapipe {ver} lacks the legacy `mp.solutions` API "
                              "(removed upstream; install the pinned range in requirements.txt)")
            try:
                fd = mp.solutions.face_detection.FaceDetection(min_detection_confidence=self.min_confidence)
                return ("mediapipe", fd), None
            except Exception as exc:  # noqa: BLE001
                return None, f"mediapipe FaceDetection failed to build ({type(exc).__name__}: {exc})"
        return self._build_yunet()

    def _build_yunet(self) -> tuple[tuple[str, object] | None, str | None]:
        if self._cv2 is None:
            return None, "opencv not installed (yunet backend needs cv2)"
        if not self.model_path:
            return None, ("face_presence.model_path not set "
                          "(yunet needs the ONNX model; see configs/faces.yaml)")
        p = Path(self.model_path)
        if not p.is_absolute():
            p = REPO_ROOT / p                      # anchor to the repo, not the process CWD
        if not p.exists():
            return None, f"YuNet model file not found: {p}"
        try:
            det = self._cv2.FaceDetectorYN.create(str(p), "", (320, 320),
                                                  float(self.min_confidence), 0.3, 5000)
            return ("yunet", det), None
        except Exception as exc:  # noqa: BLE001
            return None, f"YuNet failed to build ({type(exc).__name__}): {p}"

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
            bgr = self._cv2.cvtColor(rgb, self._cv2.COLOR_RGB2BGR)
            h, wd = bgr.shape[:2]
            impl.setInputSize((wd, h))
            _, faces = impl.detect(bgr)
            if faces is None or len(faces) == 0:
                return 0.0
            # YuNet row: [x, y, w, h, 5x2 landmarks, score]; area fraction over the frame.
            max_conf = float(np.clip(faces[:, -1].max(), 0, 1))
            largest = float(np.clip((faces[:, 2] * faces[:, 3]).max() / float(wd * h), 0, 1))
            return float(np.clip((1 - self.area_weight) * max_conf + self.area_weight * largest, 0, 1))
        return None
