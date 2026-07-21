"""OCR — Tesseract (CLAUDE.md §5, FS8 — NON-ESSENTIAL, safe to disable).

Up to `max_keyframes` frames, sampled evenly across the segment (not just its start — on-screen
text is often only in view partway through a long segment) -> concatenated on-screen text. Lazy
pytesseract (needs the tesseract binary). Unavailable/disabled -> "" (honest empty).
"""

from __future__ import annotations

import numpy as np

from .._deps import optional

# pytesseract's binary path is module-global state; remember its pristine default once so
# an Ocr built without `tesseract_cmd` never inherits a previous instance's override.
_ORIG_TESSERACT_CMD: str | None = None


class Ocr:
    def __init__(self, cfg=None) -> None:
        global _ORIG_TESSERACT_CMD
        self.enabled = bool(cfg.get("tier2.ocr.enabled", True)) if cfg else True
        self.max_keyframes = int(cfg.get("tier2.ocr.max_keyframes", 3)) if cfg else 3
        self._pt = optional("pytesseract")
        # explicit binary path (config: tier2.ocr.tesseract_cmd) -- pip installing pytesseract
        # only gets the wrapper; the tesseract binary itself is a system install and is
        # frequently NOT on PATH (esp. Windows). Leave unset to rely on PATH.
        tesseract_cmd = cfg.get("tier2.ocr.tesseract_cmd", None) if cfg else None
        if self._pt is not None:
            if _ORIG_TESSERACT_CMD is None:
                _ORIG_TESSERACT_CMD = self._pt.pytesseract.tesseract_cmd
            self._pt.pytesseract.tesseract_cmd = tesseract_cmd or _ORIG_TESSERACT_CMD
        # pytesseract present but the binary missing still yields "" (caught at call time)
        self.backend = "tesseract" if (self.enabled and self._pt is not None) else "unavailable"

    def _sample(self, frames_rgb: list[np.ndarray]) -> list[np.ndarray]:
        """Evenly-spaced indices across the segment, not just its first N frames — a segment
        can run tens of seconds and on-screen text (a sign, a menu, a screen) may only be in
        view partway through it."""
        n = len(frames_rgb)
        if n <= self.max_keyframes:
            return frames_rgb
        idx = np.linspace(0, n - 1, self.max_keyframes).round().astype(int)
        return [frames_rgb[i] for i in idx]

    def read(self, frames_rgb: list[np.ndarray]) -> str:
        if not self.enabled or self._pt is None or not frames_rgb:
            return ""
        from PIL import Image

        texts = []
        for rgb in self._sample(frames_rgb):
            try:
                texts.append(self._pt.image_to_string(Image.fromarray(rgb)).strip())
            except Exception:  # noqa: BLE001 — missing binary etc. -> honest empty
                return ""
        return "\n".join(t for t in texts if t).strip()
