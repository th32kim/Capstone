"""ASR — faster-whisper base.en, int8 (CLAUDE.md §5, FS6).

The cascade's cheap tiny.en pass is cached by content hash and reused/upgraded here (base.en).
If faster-whisper is unavailable the transcript is "" — an honest empty, never a fabricated
sentence. Track A only; a segment with no speech legitimately yields "".
"""

from __future__ import annotations

import numpy as np

from .._deps import optional
from ..contracts import AUDIO_SAMPLE_RATE


class Asr:
    def __init__(self, cfg=None, model: str | None = None) -> None:
        self.model_name = model or (cfg.get("tier2.asr.model") if cfg else None) or "base.en"
        self.compute_type = (cfg.get("tier2.asr.compute_type") if cfg else None) or "int8"
        self._model = None
        fw = optional("faster_whisper")
        if fw is not None:
            try:
                self._model = fw.WhisperModel(self.model_name, compute_type=self.compute_type)
                self.backend = "faster_whisper"
            except Exception:  # noqa: BLE001
                self._model = None
        if self._model is None:
            self.backend = "unavailable"

    def transcribe(self, pcm: np.ndarray) -> str:
        if self._model is None or pcm is None or len(pcm) == 0:
            return ""
        audio = pcm.astype(np.float32) / 32768.0
        segments, _ = self._model.transcribe(audio, language="en", sampling_rate=AUDIO_SAMPLE_RATE)
        return " ".join(s.text.strip() for s in segments).strip()
