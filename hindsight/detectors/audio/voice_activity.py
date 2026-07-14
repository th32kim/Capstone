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


class VoiceActivityDetector:
    name = "voice_activity"
    modality = "audio"

    def __init__(self, cadence: int = 1, aggressiveness: int = 2, frame_ms: int = 30,
                 fallback: str = "energy") -> None:
        self.cadence = cadence
        self.frame_ms = frame_ms
        self.fallback = fallback
        self._vad = None
        self.last_backend = "uninitialised"
        webrtcvad = optional("webrtcvad")
        if webrtcvad is not None:
            try:
                self._vad = webrtcvad.Vad(aggressiveness)
            except Exception:  # noqa: BLE001
                self._vad = None
        self._energy_norm = RunningPercentileNormalizer(5, 95)

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
                pass
        return self._score_energy(pcm)

    def _score_webrtc(self, pcm: np.ndarray) -> float:
        self.last_backend = "webrtcvad"
        frame_len = int(AUDIO_SAMPLE_RATE * self.frame_ms / 1000)  # 480 @ 30 ms
        n = pcm.size // frame_len
        if n == 0:
            return 0.0
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
            frame = pcm[i * frame_len : (i + 1) * frame_len].astype(np.float64)
            rms = float(np.sqrt(np.mean(frame**2))) if frame.size else 0.0
            if self._energy_norm.update_and_normalize(rms) > 0.5:
                voiced += 1
        return voiced / n
