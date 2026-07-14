"""SyntheticSource — deterministic generated frames + audio, for tests and demos.

No decode backend, no I/O: numpy only. Given a seed and a list of "interest" spans, it
produces a video stream (motion / scene-cut / brightness cues) and a PCM16 mono 16 kHz
audio stream (voiced / energetic cues) that the real detectors pick up, so the whole
detect -> fuse -> gate funnel runs end to end without a file. Determinism is guaranteed:
frames() and audio() each seed a fresh RNG from the base seed, so call order never matters
(CLAUDE.md §1.7).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..contracts import AUDIO_CHUNK_MS, AUDIO_SAMPLE_RATE, AudioChunk, Frame


@dataclass(frozen=True)
class InterestSpan:
    """A window of elevated salience to inject."""

    t_start: float
    t_end: float
    kinds: tuple[str, ...] = ("motion", "voice")  # subset of {motion, voice, scene, bright}

    def active(self, t_s: float) -> bool:
        return self.t_start <= t_s < self.t_end


@dataclass
class SyntheticSource:
    """A deterministic Source implementation. Satisfies contracts.Source."""

    name: str = "synthetic"
    fps: float = 12.0
    sample_rate: int = AUDIO_SAMPLE_RATE
    duration: float = 20.0
    height: int = 48
    width: int = 64
    seed: int = 1337
    spans: list[InterestSpan] = field(default_factory=list)

    def duration_s(self) -> float:
        return self.duration

    # -- video -------------------------------------------------------------
    def frames(self):
        rng = np.random.default_rng(self.seed ^ 0xF00D)
        n = int(round(self.duration * self.fps))
        # A stable low-frequency background; scene cuts swap its phase.
        base = rng.integers(40, 90, size=(self.height, self.width, 3), dtype=np.uint8)
        scene_id = 0
        for i in range(n):
            t_s = i / self.fps
            t_ms = int(round(t_s * 1000.0))
            kinds = self._kinds_at(t_s)
            if "scene" in kinds and int(t_s * self.fps) % max(1, int(self.fps)) == 0:
                scene_id += 1
                base = rng.integers(40, 90, size=(self.height, self.width, 3), dtype=np.uint8)
            frame = base.copy()
            if "bright" in kinds:
                frame = np.clip(frame.astype(np.int16) + 60, 0, 255).astype(np.uint8)
            if "motion" in kinds:
                # a moving bright block => residual motion after ego-comp is non-zero
                bw, bh = self.width // 6, self.height // 6
                x = int((t_s * 40) % (self.width - bw))
                y = self.height // 2 - bh // 2
                frame[y : y + bh, x : x + bw] = 240
            # a little per-frame sensor noise so nothing is perfectly static
            noise = rng.integers(-4, 5, size=frame.shape, dtype=np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            yield Frame(t_ms=t_ms, rgb=frame)

    # -- audio -------------------------------------------------------------
    def audio(self):
        rng = np.random.default_rng(self.seed ^ 0xBEEF)
        chunk_samples = int(self.sample_rate * AUDIO_CHUNK_MS / 1000)  # 320 @ 16 kHz/20 ms
        n_chunks = int(round(self.duration * 1000 / AUDIO_CHUNK_MS))
        for i in range(n_chunks):
            t_s = i * AUDIO_CHUNK_MS / 1000.0
            t_ms = int(round(t_s * 1000.0))
            kinds = self._kinds_at(t_s)
            amp = 1500 if ("voice" in kinds or "energy" in kinds) else 120
            sig = rng.normal(0, amp, size=chunk_samples)
            if "voice" in kinds:
                # add a voiced-like tone so webrtcvad/energy both respond
                tt = (np.arange(chunk_samples) + i * chunk_samples) / self.sample_rate
                sig = sig + 4000 * np.sin(2 * np.pi * 180 * tt)
            pcm = np.clip(sig, -32768, 32767).astype(np.int16)
            yield AudioChunk(t_ms=t_ms, pcm=pcm)

    def _kinds_at(self, t_s: float) -> set[str]:
        kinds: set[str] = set()
        for span in self.spans:
            if span.active(t_s):
                kinds.update(span.kinds)
        return kinds


def demo_source(seed: int = 1337, duration: float = 240.0) -> SyntheticSource:
    """A canned, *realistic* passive session: mostly boring, with a few short interest spans.

    A real lifelog is dominated by walking/idle/silence; interest is sparse. That is what
    lets the funnel reduce >= 80% (CLAUDE.md §3.3). A short clip densely packed with events
    cannot meet that target and the gate rightly refuses to pretend it did.
    """
    return SyntheticSource(
        seed=seed,
        duration=duration,
        spans=[
            InterestSpan(30.0, 35.0, ("motion", "voice", "bright")),
            InterestSpan(80.0, 86.0, ("motion", "voice", "scene")),
            InterestSpan(140.0, 144.0, ("voice", "scene")),
            InterestSpan(200.0, 206.0, ("motion", "voice")),
        ],
    )
