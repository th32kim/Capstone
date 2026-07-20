"""Focused tests for the 16 kHz source contract and Tier-1 audio detectors."""

from __future__ import annotations

import numpy as np
import pytest

from hindsight.contracts import AUDIO_SAMPLE_RATE, AudioChunk, Window
from hindsight.detectors.audio.audio_energy import AudioEnergyDetector, rms_dbfs
from hindsight.detectors.audio.voice_activity import VoiceActivityDetector
from hindsight.sources.clip import _AudioChunker


def _window(pcm: np.ndarray) -> Window:
    chunks = tuple(
        AudioChunk(t_ms=i * 20, pcm=pcm[i * 320 : (i + 1) * 320].copy())
        for i in range(len(pcm) // 320)
    )
    return Window(index=0, t_start=0.0, t_end=0.5, frames=(), chunks=chunks)


def _fallback_detector(**overrides) -> VoiceActivityDetector:
    params = {
        "cadence": 1,
        "aggressiveness": 2,
        "frame_ms": 30,
        "fallback": "energy",
        "fallback_norm_percentiles": (5, 95),
        "fallback_threshold": 0.5,
        "fallback_min_dbfs": -45.0,
        "silence_floor_dbfs": -120.0,
    }
    params.update(overrides)
    return VoiceActivityDetector(**params)


def test_audio_chunker_preserves_media_start_timestamp():
    chunker = _AudioChunker()
    chunks = chunker.push(np.arange(640, dtype=np.int16), start_s=1.25)
    assert [chunk.t_ms for chunk in chunks] == [1250, 1270]
    assert all(len(chunk.pcm) == AUDIO_SAMPLE_RATE * 20 // 1000 for chunk in chunks)


def test_audio_chunker_carries_partial_blocks_without_retiming():
    chunker = _AudioChunker()
    assert chunker.push(np.ones(100, dtype=np.int16), start_s=0.4) == []
    chunks = chunker.push(np.ones(220, dtype=np.int16), start_s=9.0)
    assert len(chunks) == 1
    assert chunks[0].t_ms == 400


def test_rms_dbfs_uses_configured_silence_floor():
    assert rms_dbfs(np.zeros(320, np.int16), -110.0) == -110.0
    assert rms_dbfs(np.full(320, 32767, np.int16), -110.0) == pytest.approx(0.0, abs=0.01)


def test_audio_energy_masks_an_empty_pcm_window():
    detector = AudioEnergyDetector(cadence=1, norm_percentiles=(5, 95), silence_floor_dbfs=-120.0)
    empty = Window(0, 0.0, 0.5, (), (AudioChunk(0, np.empty(0, np.int16)),))
    assert detector.score(empty) is None


def test_energy_fallback_detects_steady_speech_and_rejects_quiet_audio():
    detector = _fallback_detector()
    detector._vad = None
    quiet = np.full(AUDIO_SAMPLE_RATE // 2, 10, dtype=np.int16)
    speech = np.full(AUDIO_SAMPLE_RATE // 2, 5000, dtype=np.int16)
    assert detector.score(_window(quiet)) == 0.0
    assert detector.score(_window(speech)) == 1.0
    assert detector.last_backend == "energy_fallback"


def test_webrtcvad_runtime_failure_permanently_switches_to_loud_fallback():
    class BrokenVad:
        def is_speech(self, frame: bytes, sample_rate: int) -> bool:
            raise RuntimeError("backend failed")

    detector = _fallback_detector()
    detector._vad = BrokenVad()
    speech = np.full(AUDIO_SAMPLE_RATE // 2, 5000, dtype=np.int16)
    assert detector.score(_window(speech)) == 1.0
    assert detector._vad is None
    assert detector.last_backend == "energy_fallback"
    assert detector.fallback_reason == "webrtcvad_runtime_error"


@pytest.mark.parametrize("frame_ms", [0, 15, 40])
def test_vad_rejects_unsupported_frame_duration(frame_ms):
    with pytest.raises(ValueError):
        _fallback_detector(frame_ms=frame_ms)


class _AlwaysSpeech:
    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        return True


def test_webrtc_masks_window_too_short_for_one_frame():
    # a window with fewer than one 30 ms (480-sample) frame CANNOT run webrtcvad -> it must be
    # MASKED (None), not scored a fabricated 0.0, and must not claim the webrtcvad backend (§3).
    detector = _fallback_detector()
    detector._vad = _AlwaysSpeech()
    short = Window(0, 0.0, 0.5, (), (AudioChunk(0, np.full(320, 5000, np.int16)),))  # 20 ms
    assert detector.score(short) is None
    assert detector.last_backend != "webrtcvad"


def test_webrtc_scores_a_real_voiced_fraction():
    # a full window of "always speech" must score 1.0 via the real webrtcvad code path (guards the
    # voiced/n computation, not just the backend label).
    detector = _fallback_detector()
    detector._vad = _AlwaysSpeech()
    full = _window(np.full(AUDIO_SAMPLE_RATE // 2, 5000, dtype=np.int16))
    assert detector.score(full) == 1.0
    assert detector.last_backend == "webrtcvad"


def test_build_detectors_tolerates_config_predating_the_new_audio_keys():
    # a config written before the fallback_*/silence_floor keys existed must still build the audio
    # detectors (no KeyError) — every stage stays independently re-runnable (§1.6).
    from hindsight.config import _freeze
    from hindsight.detectors import build_detectors

    cfg = _freeze({"detectors": {
        "voice_activity": {"enabled": True, "cadence": 1},
        "audio_energy": {"enabled": True, "cadence": 1},
    }}, source="test")
    names = [d.name for d in build_detectors(cfg)]
    assert "voice_activity" in names and "audio_energy" in names


def test_audio_detectors_reject_out_of_range_config():
    from hindsight.detectors.audio.audio_energy import AudioEnergyDetector

    with pytest.raises(ValueError):
        _fallback_detector(silence_floor_dbfs=10.0)          # dBFS must be <= 0
    with pytest.raises(ValueError):
        _fallback_detector(fallback_norm_percentiles=(95, 5))  # lo < hi required
    with pytest.raises(ValueError):
        AudioEnergyDetector(cadence=1, norm_percentiles=(5, 95), silence_floor_dbfs=5.0)
