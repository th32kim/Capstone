"""Captioning — BLIP-base (CLAUDE.md §5, FS7).

One keyframe (peak salience) in, one caption out. Lazy transformers+torch. Unavailable ->
"" (honest empty), never a made-up caption.
"""

from __future__ import annotations

import numpy as np

from .._deps import optional


class Captioner:
    def __init__(self, cfg=None) -> None:
        self.model_name = (cfg.get("tier2.caption.model") if cfg else None) \
            or "Salesforce/blip-image-captioning-base"
        self._proc = None
        self._model = None
        transformers = optional("transformers")
        if transformers is not None and optional("torch") is not None:
            try:
                self._proc = transformers.BlipProcessor.from_pretrained(self.model_name)
                self._model = transformers.BlipForConditionalGeneration.from_pretrained(self.model_name)
                self.backend = "blip"
            except Exception:  # noqa: BLE001
                self._model = None
        if self._model is None:
            self.backend = "unavailable"

    def caption(self, rgb: np.ndarray | None) -> str:
        if self._model is None or rgb is None:
            return ""
        from PIL import Image

        inputs = self._proc(Image.fromarray(rgb), return_tensors="pt")
        out = self._model.generate(**inputs, max_new_tokens=30)
        return self._proc.decode(out[0], skip_special_tokens=True).strip()
