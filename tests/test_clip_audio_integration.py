"""Real-backend integration coverage for PyAV decoding/resampling and WebRTC VAD."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from hindsight.clock import av_drift_ms
from hindsight.config import load_config
from hindsight.detectors import run_detectors
from hindsight.sources.clip import ClipSource

av = pytest.importorskip("av")
pytest.importorskip("webrtcvad")


def _write_test_clip(path) -> None:
    """Encode two seconds of 10 fps video and mono 48 kHz audio into a real MP4."""
    duration_s = 2
    video_fps = 10
    audio_rate = 48_000
    audio_frame_samples = 960  # 20 ms at the native rate

    with av.open(str(path), mode="w") as container:
        video = container.add_stream("mpeg4", rate=video_fps)
        video.width = 64
        video.height = 48
        video.pix_fmt = "yuv420p"

        audio = container.add_stream("aac", rate=audio_rate)
        audio.layout = "mono"

        for i in range(duration_s * video_fps):
            rgb = np.full((48, 64, 3), 30 + i, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts = i
            frame.time_base = Fraction(1, video_fps)
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)

        for i in range(duration_s * audio_rate // audio_frame_samples):
            start = i * audio_frame_samples
            t = (np.arange(audio_frame_samples) + start) / audio_rate
            # Speech-like harmonic signal, kept well inside PCM16 range.
            pcm = (
                5_000 * np.sin(2 * np.pi * 180 * t) + 2_500 * np.sin(2 * np.pi * 360 * t)
            ).astype(np.int16)
            frame = av.AudioFrame.from_ndarray(pcm.reshape(1, -1), format="s16", layout="mono")
            frame.sample_rate = audio_rate
            frame.pts = start
            frame.time_base = Fraction(1, audio_rate)
            for packet in audio.encode(frame):
                container.mux(packet)
        for packet in audio.encode():
            container.mux(packet)


def test_clip_source_audio_contract_and_real_vad(tmp_path):
    path = tmp_path / "audio_contract.mp4"
    _write_test_clip(path)

    source = ClipSource(path)
    meta = source.meta()
    frames = list(source.frames())
    chunks = list(source.audio())

    assert meta.native_sample_rate == 48_000
    assert source.sample_rate == 16_000
    assert chunks
    assert all(chunk.pcm.dtype == np.int16 for chunk in chunks)
    assert all(chunk.pcm.ndim == 1 and len(chunk.pcm) == 320 for chunk in chunks)
    assert [chunk.t_ms for chunk in chunks] == sorted(chunk.t_ms for chunk in chunks)
    assert all(b.t_ms - a.t_ms == 20 for a, b in zip(chunks, chunks[1:]))
    assert av_drift_ms(frames, chunks) <= 100

    result = run_detectors(source, load_config("default"))
    assert result.notes["vad_backend"] == "webrtcvad"
    assert result.notes["vad_fallback_reason"] is None
    assert result.notes["n_chunks"] == len(chunks)
