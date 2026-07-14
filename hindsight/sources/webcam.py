"""WebcamSource — live webcam + mic, for fast iteration. Secondary source.

Bounded by `duration` so it terminates. Video via OpenCV, audio via sounddevice, both
lazily imported and resampled/rechunked to the frozen contract (PCM16 mono 16 kHz, 20 ms).
This is the "watch it work live" path; the corpus and all eval run through ClipSource.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .._deps import require
from ..contracts import AUDIO_CHUNK_MS, AUDIO_SAMPLE_RATE, AudioChunk, Frame


@dataclass
class WebcamSource:
    name: str = "webcam"
    device: int = 0
    fps: float = 30.0
    sample_rate: int = AUDIO_SAMPLE_RATE
    duration: float = 30.0

    def duration_s(self) -> float:
        return self.duration

    def frames(self):
        cv2 = require("cv2", feature="WebcamSource video")
        cap = cv2.VideoCapture(self.device)
        t0 = time.monotonic()
        try:
            while time.monotonic() - t0 < self.duration:
                ok, bgr = cap.read()
                if not ok:
                    break
                t_ms = int(round((time.monotonic() - t0) * 1000))
                yield Frame(t_ms=t_ms, rgb=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()

    def audio(self):
        sd = require("sounddevice", feature="WebcamSource audio")
        chunk_samples = int(AUDIO_SAMPLE_RATE * AUDIO_CHUNK_MS / 1000)
        n_chunks = int(self.duration * 1000 / AUDIO_CHUNK_MS)
        with sd.InputStream(samplerate=AUDIO_SAMPLE_RATE, channels=1, dtype="int16") as stream:
            for i in range(n_chunks):
                data, _ = stream.read(chunk_samples)
                t_ms = i * AUDIO_CHUNK_MS
                yield AudioChunk(t_ms=t_ms, pcm=np.asarray(data, dtype=np.int16).reshape(-1))
