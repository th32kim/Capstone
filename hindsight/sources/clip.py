"""ClipSource — the PRIMARY source. Any MP4/MOV -> Frame + AudioChunk on one clock.

GoPro, Meta Ray-Ban, iPhone, phone-on-a-lanyard: all just files here. Audio is decoded to
PCM16 mono 16 kHz regardless of the container's native codec/rate (FS2). Video and audio
are decoded on *independent* passes sharing the container's presentation timestamps, so
A/V drift can be *measured* (clock.av_drift_ms), not assumed.

Backends, tried in order and imported lazily (CLAUDE.md §1 / _deps):
  1. PyAV (`av`) — decodes both streams and resamples audio in-container. Preferred.
  2. OpenCV (`cv2`) for video + soundfile/librosa for audio — fallback.

If no backend is installed the constructor still succeeds; iterating raises a clear
MissingDependency. Metadata probing (`meta()`) needs a backend too.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .._deps import optional, require
from ..contracts import AUDIO_CHUNK_MS, AUDIO_SAMPLE_RATE, AudioChunk, Frame


@dataclass(frozen=True)
class ClipMeta:
    width: int
    height: int
    fps: float
    native_sample_rate: int | None
    duration_s: float


class _AudioChunker:
    """Turn resampled PCM blocks into fixed 20 ms chunks while preserving the first media PTS."""

    def __init__(
        self, sample_rate: int = AUDIO_SAMPLE_RATE, chunk_ms: int = AUDIO_CHUNK_MS
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_samples = int(sample_rate * chunk_ms / 1000)
        self._buffer = np.empty(0, dtype=np.int16)
        self._next_t_s: float | None = None

    def push(self, samples: np.ndarray, start_s: float | None) -> list[AudioChunk]:
        samples = np.asarray(samples, dtype=np.int16).reshape(-1)
        if samples.size == 0:
            return []
        if self._next_t_s is None:
            self._next_t_s = float(start_s) if start_s is not None else 0.0
        self._buffer = np.concatenate((self._buffer, samples))
        chunks: list[AudioChunk] = []
        while self._buffer.size >= self.chunk_samples:
            pcm = self._buffer[: self.chunk_samples].copy()
            self._buffer = self._buffer[self.chunk_samples :]
            chunks.append(AudioChunk(t_ms=int(round(self._next_t_s * 1000.0)), pcm=pcm))
            self._next_t_s += self.chunk_samples / self.sample_rate
        return chunks


def _downscale_long_edge(rgb: np.ndarray, max_long_edge: int | None):
    if not max_long_edge:
        return rgb
    h, w = rgb.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_long_edge:
        return rgb
    cv2 = optional("cv2")
    scale = max_long_edge / long_edge
    nw, nh = int(round(w * scale)), int(round(h * scale))
    if cv2 is not None:
        return cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    # numpy nearest-neighbour fallback (rare path; av usually present with cv2 absent)
    ys = (np.linspace(0, h - 1, nh)).astype(int)
    xs = (np.linspace(0, w - 1, nw)).astype(int)
    return rgb[ys][:, xs]


class ClipSource:
    """A file-backed Source. Satisfies contracts.Source."""

    def __init__(self, path: str | Path, *, target_fps: float | None = None,
                 max_long_edge_px: int | None = 1280, name: str | None = None) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.name = name or self.path.name
        self.target_fps = target_fps
        self.max_long_edge_px = max_long_edge_px
        self._meta: ClipMeta | None = None
        self.sample_rate = AUDIO_SAMPLE_RATE  # we always resample to this
        # Real recordings ship malformed/truncated packets (esp. the final one). We skip an
        # undecodable packet and keep going rather than crash the whole pipeline; the counts are
        # surfaced in the detect notes so a lossy decode is never silent (CLAUDE.md §1).
        self.video_decode_errors = 0
        self.audio_decode_errors = 0

    # -- metadata ----------------------------------------------------------
    @property
    def fps(self) -> float:
        return self.meta().fps

    def duration_s(self) -> float:
        return self.meta().duration_s

    def meta(self) -> ClipMeta:
        if self._meta is not None:
            return self._meta
        av = optional("av")
        if av is not None:
            self._meta = self._meta_av(av)
        else:
            self._meta = self._meta_cv()
        return self._meta

    def _meta_av(self, av) -> ClipMeta:
        with av.open(str(self.path)) as container:
            vstreams = container.streams.video
            astreams = container.streams.audio
            if not vstreams:
                raise ValueError(f"{self.path} has no video stream")
            v = vstreams[0]
            fps = float(v.average_rate) if v.average_rate else 0.0
            width, height = v.codec_context.width, v.codec_context.height
            native_sr = int(astreams[0].codec_context.sample_rate) if astreams else None
            duration = float(container.duration / av.time_base) if container.duration else 0.0
        return ClipMeta(width, height, fps, native_sr, duration)

    def _meta_cv(self) -> ClipMeta:
        cv2 = require("cv2", feature="ClipSource metadata (no PyAV)")
        cap = cv2.VideoCapture(str(self.path))
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = (n / fps) if fps else 0.0
        finally:
            cap.release()
        return ClipMeta(width, height, fps, None, duration)

    # -- video -------------------------------------------------------------
    def frames(self):
        av = optional("av")
        if av is not None:
            yield from self._frames_av(av)
        else:
            yield from self._frames_cv()

    def _frame_step_s(self) -> float:
        # decode-time subsample: Tier-1 samples one frame per 0.5 s window, so a wearable's
        # 30-60 fps is wasteful. target_fps (config) caps decode rate; null = native.
        return (1.0 / self.target_fps) if self.target_fps else 0.0

    def _frames_av(self, av):
        step = self._frame_step_s()
        next_emit = 0.0
        with av.open(str(self.path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            # demux packet-by-packet so one corrupt packet is skipped, not fatal (the demux flush
            # packet also drains buffered frames — a bare container.decode() drops that tail).
            for packet in container.demux(stream):
                try:
                    decoded = list(packet.decode())
                except av.error.FFmpegError:
                    self.video_decode_errors += 1
                    continue
                for frame in decoded:
                    if frame.pts is None:
                        continue
                    t_s = float(frame.pts * stream.time_base)
                    if step and t_s + 1e-9 < next_emit:
                        continue
                    next_emit = t_s + step
                    rgb = frame.to_ndarray(format="rgb24")
                    yield Frame(t_ms=int(round(t_s * 1000)),
                                rgb=_downscale_long_edge(rgb, self.max_long_edge_px))

    def _frames_cv(self):
        cv2 = require("cv2", feature="ClipSource video decode (no PyAV)")
        step = self._frame_step_s()
        next_emit = 0.0
        cap = cv2.VideoCapture(str(self.path))
        try:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                t_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if step and t_s + 1e-9 < next_emit:
                    continue
                next_emit = t_s + step
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                yield Frame(t_ms=int(round(t_s * 1000)),
                            rgb=_downscale_long_edge(rgb, self.max_long_edge_px))
        finally:
            cap.release()

    # -- audio -------------------------------------------------------------
    def audio(self):
        av = optional("av")
        if av is not None:
            yield from self._audio_av(av)
        else:
            yield from self._audio_fallback()

    def _audio_av(self, av):
        def _emit(rs, chunker):
            samples = rs.to_ndarray().reshape(-1).astype(np.int16)
            start_s = (
                float(rs.pts * rs.time_base)
                if rs.pts is not None and rs.time_base is not None
                else None
            )
            return chunker.push(samples, start_s)

        with av.open(str(self.path)) as container:
            if not container.streams.audio:
                return
            stream = container.streams.audio[0]
            resampler = av.AudioResampler(format="s16", layout="mono", rate=AUDIO_SAMPLE_RATE)
            chunker = _AudioChunker()
            # demux packet-by-packet: a single corrupt/truncated packet (common as the FINAL
            # packet of a real recording) must not lose the whole audio track — skip it and go on.
            for packet in container.demux(stream):
                try:
                    decoded = list(packet.decode())
                except av.error.FFmpegError:
                    self.audio_decode_errors += 1
                    continue
                for frame in decoded:
                    for rs in resampler.resample(frame):
                        yield from _emit(rs, chunker)
            # AudioResampler can buffer samples internally. Flush them so the measured
            # final audio timestamp is not shortened by decoder/resampler latency.
            for rs in resampler.resample(None):
                yield from _emit(rs, chunker)

    def _audio_fallback(self):
        sf = require("soundfile", feature="ClipSource audio decode (no PyAV)")
        librosa = optional("librosa")
        data, sr = sf.read(str(self.path), dtype="int16", always_2d=True)
        mono = data.mean(axis=1)
        if sr != AUDIO_SAMPLE_RATE:
            if librosa is None:
                raise RuntimeError(f"need librosa to resample {sr}->{AUDIO_SAMPLE_RATE} Hz")
            mono = librosa.resample(mono.astype(np.float32), orig_sr=sr, target_sr=AUDIO_SAMPLE_RATE)
        mono = np.clip(mono, -32768, 32767).astype(np.int16)
        chunk_samples = int(AUDIO_SAMPLE_RATE * AUDIO_CHUNK_MS / 1000)
        for i in range(0, len(mono) - chunk_samples + 1, chunk_samples):
            t_ms = int(round(i / AUDIO_SAMPLE_RATE * 1000))
            yield AudioChunk(t_ms=t_ms, pcm=mono[i : i + chunk_samples].copy())
