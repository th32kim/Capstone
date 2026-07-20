"""D5 voice_activity — webrtcvad, 30 ms frames @ 16 kHz, with an RMS-energy fallback.

Window score = fraction of voiced 30 ms frames in the 0.5 s window. Highest fusion weight.

If webrtcvad is missing or throws, we fall back to an adaptive RMS-energy gate AND SAY SO:
`last_backend` flips to "energy_fallback" and the runner records it in the notes sidecar +
timings log. A silent fallback that quietly degrades F1 is, per M2, the worst outcome — so
this one is loud.
"""

from __future__ import annotations

import numpy as np

from ..._deps import optional
from ...contracts import AUDIO_SAMPLE_RATE, Window
from ..base import RunningPercentileNormalizer
from .audio_energy import rms_dbfs


class VoiceActivityDetector:
    name = "voice_activity"
    modality = "audio"

    def __init__(
        self,
        cadence: int,
        aggressiveness: int,
        frame_ms: int,
        fallback: str,
        fallback_norm_percentiles: tuple[float, float],
        fallback_threshold: float,
        fallback_min_dbfs: float,
        silence_floor_dbfs: float,
    ) -> None:
        if frame_ms not in (10, 20, 30):
            raise ValueError("webrtcvad frame_ms must be one of 10, 20, or 30")
        if not 0 <= aggressiveness <= 3:
            raise ValueError("webrtcvad aggressiveness must be between 0 and 3")
        if fallback != "energy":
            raise ValueError(f"unsupported voice_activity fallback {fallback!r}")
        if not 0.0 <= fallback_threshold <= 1.0:
            raise ValueError("voice_activity fallback_threshold must be in [0,1]")
        if fallback_min_dbfs > 0.0 or silence_floor_dbfs > 0.0:
            raise ValueError("dBFS levels must be <= 0 (0 dBFS = digital full scale)")
        lo, hi = fallback_norm_percentiles
        if not 0.0 <= lo < hi <= 100.0:
            raise ValueError("fallback_norm_percentiles must be 0 <= lo < hi <= 100")
        self.cadence = cadence
        self.frame_ms = frame_ms
        self.fallback = fallback
        self.fallback_threshold = float(fallback_threshold)
        self.fallback_min_dbfs = float(fallback_min_dbfs)
        self.silence_floor_dbfs = float(silence_floor_dbfs)
        self._vad = None
        self.last_backend = "uninitialised"
        self.fallback_reason: str | None = None
        webrtcvad = optional("webrtcvad")
        if webrtcvad is not None:
            try:
                self._vad = webrtcvad.Vad(aggressiveness)
            except Exception:  # noqa: BLE001
                self._vad = None
                self.fallback_reason = "webrtcvad_initialization_failed"
        else:
            self.fallback_reason = "webrtcvad_unavailable"
        self._energy_norm = RunningPercentileNormalizer(*fallback_norm_percentiles)

    def _pcm(self, w: Window) -> np.ndarray:
        if not w.chunks:
            return np.empty(0, dtype=np.int16)
        return np.concatenate([c.pcm for c in w.chunks]).astype(np.int16)

    def score(self, w: Window) -> float | None:
        pcm = self._pcm(w)
        if pcm.size == 0:
            return None
        if self._vad is not None:
            try:
                return self._score_webrtc(pcm)
            except Exception:  # noqa: BLE001 — degrade loudly, never lose the window
                self._vad = None
                self.fallback_reason = "webrtcvad_runtime_error"
        return self._score_energy(pcm)

    def _score_webrtc(self, pcm: np.ndarray) -> float | None:
        frame_len = int(AUDIO_SAMPLE_RATE * self.frame_ms / 1000)  # 480 @ 30 ms
        n = pcm.size // frame_len
        if n == 0:
            # window too short for even one VAD frame -> could not run -> masked, NOT a real 0.0
            # (CLAUDE.md §3). Do not claim the webrtcvad backend for a window it never scored.
            return None
        self.last_backend = "webrtcvad"
        voiced = 0
        for i in range(n):
            frame = pcm[i * frame_len : (i + 1) * frame_len].tobytes()
            if self._vad.is_speech(frame, AUDIO_SAMPLE_RATE):
                voiced += 1
        return voiced / n

    def _score_energy(self, pcm: np.ndarray) -> float:
        self.last_backend = "energy_fallback"
        frame_len = int(AUDIO_SAMPLE_RATE * self.frame_ms / 1000)
        n = max(pcm.size // frame_len, 1)
        voiced = 0
        for i in range(n):
            frame = pcm[i * frame_len : (i + 1) * frame_len]
            dbfs = rms_dbfs(frame, self.silence_floor_dbfs)
            normalised = self._energy_norm.update_and_normalize(dbfs)
            if dbfs >= self.fallback_min_dbfs or normalised >= self.fallback_threshold:
                voiced += 1
        return voiced / n
