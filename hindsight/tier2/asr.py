"""ASR — faster-whisper base.en, int8 (CLAUDE.md §5, FS6).

The cascade's cheap tiny.en pass is cached by content hash and reused/upgraded here (base.en).
If faster-whisper is unavailable the transcript is "" — an honest empty, never a fabricated
sentence. Track A only; a segment with no speech legitimately yields "".
"""

from __future__ import annotations

import numpy as np

from .._deps import optional


class Asr:
    def __init__(self, cfg=None, model: str | None = None) -> None:
        # Config.get raises on a missing key unless a default is passed — every read here
        # must supply one so a config written against an older schema still builds.
        self.model_name = model or (cfg.get("tier2.asr.model", None) if cfg else None) or "base.en"
        self.compute_type = (cfg.get("tier2.asr.compute_type", None) if cfg else None) or "int8"
        self.device = (cfg.get("tier2.asr.device", None) if cfg else None) or "cpu"
        self._model = None
        self.build_error: str | None = None   # why the model failed to build (e.g. bad device)
        fw = optional("faster_whisper")
        if fw is not None:
            try:
                self._model = fw.WhisperModel(
                    self.model_name, device=self.device, compute_type=self.compute_type)
                self.backend = "faster_whisper"
            except Exception as exc:  # noqa: BLE001
                self._model = None
                self.build_error = f"{type(exc).__name__}: {exc}"
        if self._model is None:
            # distinguish "not installed" from "installed but failed to build" — a device
            # misconfiguration must not masquerade as a missing dependency.
            self.backend = ("unavailable" if self.build_error is None
                            else f"unavailable ({self.build_error})")

    def transcribe(self, pcm: np.ndarray) -> str:
        if self._model is None or pcm is None or len(pcm) == 0:
            return ""
        audio = pcm.astype(np.float32) / 32768.0
        # faster-whisper's transcribe() has no sampling_rate param -- it always expects 16 kHz
        # float32 PCM, which matches the frozen AUDIO_SAMPLE_RATE contract, so no conversion needed.
        segments, _ = self._model.transcribe(audio, language="en")
        return " ".join(s.text.strip() for s in segments).strip()
