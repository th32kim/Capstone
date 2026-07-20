"""ASR + the shared transcript cache: decode-identity keying, cascade tiny.en on the
uncertain band ONLY (and only for validators that read evidence), Tier-2 reuse, and honest
degradation when faster-whisper is absent.

faster-whisper may or may not be installed in the running environment, so tests inject a
fake module into sys.modules where a real transcription path is exercised, and force
ImportError (sys.modules entry = None) where absence is required. The fake's transcribe()
signature deliberately has NO **kwargs: passing an argument faster-whisper does not accept
(e.g. the old sampling_rate= bug) raises TypeError and fails the test — a regression guard.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from hindsight.cascade import run_cascade
from hindsight.cascade.cache import evidence_key
from hindsight.contracts import AudioChunk, Evidence, Segment
from hindsight.tier2.asr import Asr, TranscriptCache, slice_pcm


# --------------------------------------------------------------------------- fakes
class _FakeSegment:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeWhisperModel:
    last_kwargs: dict = {}
    last_local_files_only: bool | None = None

    def __init__(self, model_name: str, device: str = "auto",
                 compute_type: str | None = None, local_files_only: bool = False) -> None:
        assert device == "cpu"  # config must pin cpu — "auto" crashes on CUDA-less NVIDIA boxes
        self.model_name = model_name
        _FakeWhisperModel.last_local_files_only = local_files_only

    def transcribe(self, audio, language=None, beam_size=None, temperature=None):
        # no **kwargs on purpose: an unexpected kwarg (sampling_rate=...) must TypeError
        _FakeWhisperModel.last_kwargs = {
            "language": language, "beam_size": beam_size, "temperature": temperature}
        return iter([_FakeSegment(f"fake:{self.model_name}:{len(audio)}")]), None


@pytest.fixture
def fake_whisper(monkeypatch):
    mod = types.ModuleType("faster_whisper")
    mod.WhisperModel = _FakeWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", mod)
    return mod


def _cfg_with_cache(cfg, tmp_path):
    return cfg.with_overrides(**{"tier2.asr.transcript_cache": str(tmp_path / "tc")})


# --------------------------------------------------------------------------- cache
def test_transcript_cache_roundtrip(tmp_path):
    cache = TranscriptCache(tmp_path)
    pcm = np.arange(320, dtype=np.int16)
    key = cache.key(pcm, "tiny.en|en|5|0.0|int8|cpu")
    assert cache.get(key) is None                      # miss
    cache.put(key, "")                                 # silence is a REAL result
    assert cache.get(key) == ""                        # hit, distinguishable from miss
    cache.put(key, "hello")
    assert cache.get(key) == "hello"
    assert cache.key(pcm, "base.en|en|5|0.0|int8|cpu") != key   # decode identity in the key
    assert cache.key(pcm[:-1], "tiny.en|en|5|0.0|int8|cpu") != key  # content in the key
    assert not list(tmp_path.glob("*.tmp"))            # atomic put leaves no temp files


def test_decode_settings_participate_in_cache_key(cfg, tmp_path):
    c = _cfg_with_cache(cfg, tmp_path)
    base = Asr(c, lazy=True)
    small_en = Asr(c.with_overrides(**{"tier2.asr.model": "small"}), lazy=True)
    small_ko = Asr(c.with_overrides(**{"tier2.asr.model": "small",
                                       "tier2.asr.language": "ko"}), lazy=True)
    pcm = np.arange(320, dtype=np.int16)
    keys = {TranscriptCache.key(pcm, a.decode_id) for a in (base, small_en, small_ko)}
    assert len(keys) == 3   # model AND language change the transcript -> must change the key


def test_en_model_coerces_language(cfg, tmp_path):
    # tiny.en is English-only: a configured 'ko' cannot apply (faster-whisper would coerce
    # silently); the effective language must be what the cache key sees.
    c = _cfg_with_cache(cfg, tmp_path).with_overrides(**{"tier2.asr.language": "ko"})
    assert Asr(c, model="tiny.en", lazy=True).language == "en"
    assert Asr(c, model="small", lazy=True).language == "ko"


def test_slice_pcm_bounds():
    chunks = [AudioChunk(t_ms=i * 20, pcm=np.full(320, i, dtype=np.int16)) for i in range(100)]
    pcm = slice_pcm(chunks, 0.5, 1.0)
    assert pcm.dtype == np.int16
    # 20 ms chunks with 500 <= t_ms <= 1000 -> indices 25..50 inclusive
    assert pcm.size == 26 * 320
    assert slice_pcm(chunks, 10.0, 11.0).size == 0


def test_evidence_key_distinguishes_none_from_silence():
    def ev(transcript):
        return Evidence("s", keyframes_jpeg=(b"x",), transcript=transcript,
                        detector_evidence={}, duration_s=10.0,
                        salience_peak=0.6, salience_mean=0.5)
    # None = ASR could not run; "" = confirmed silence. Different evidence, different key.
    assert evidence_key(ev(None), "h") != evidence_key(ev(""), "h")


# --------------------------------------------------------------------------- Asr
def test_transcribe_runs_and_caches(cfg, tmp_path, fake_whisper, monkeypatch):
    c = _cfg_with_cache(cfg, tmp_path)
    asr = Asr(c)
    assert asr.backend == "faster_whisper"
    pcm = np.arange(1600, dtype=np.int16)
    text = asr.transcribe(pcm)
    assert text == "fake:base.en:1600"                 # would be "" if a bad kwarg TypeError'd
    assert _FakeWhisperModel.last_kwargs["temperature"] == 0.0  # determinism (§1.7)
    # a fresh Asr with NO backend still answers from the cache — the reuse property
    monkeypatch.setitem(sys.modules, "faster_whisper", None)  # forces ImportError
    asr2 = Asr(c, lazy=True)
    assert asr2.transcribe_or_none(pcm) == "fake:base.en:1600"


def test_honest_empty_without_backend(cfg, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "faster_whisper", None)  # force ImportError regardless of env
    asr = Asr(_cfg_with_cache(cfg, tmp_path))
    assert asr.backend == "unavailable"
    pcm = np.arange(1600, dtype=np.int16)
    assert asr.transcribe_or_none(pcm) is None         # could not run — not "silent"
    assert asr.transcribe(pcm) == ""                   # Tier-2 contract: honest empty
    assert not list((tmp_path / "tc").glob("*.txt"))   # nothing cached from a non-run
    assert asr.transcribe(np.empty(0, dtype=np.int16)) == ""


# --------------------------------------------------------------------------- cascade
def _seg(seg_id, peak, t_start=0.0):
    return Segment(seg_id, "sess", "src", t_start=t_start, t_end=t_start + 10.0,
                   active_start=t_start + 2.0, active_end=t_start + 7.0,
                   salience_peak=peak, salience_mean=peak * 0.8, keyframe_ts=(t_start + 3.0,))


def _chunks(duration_s=35.0):
    n = int(duration_s * 1000) // 20
    return [AudioChunk(t_ms=i * 20, pcm=np.full(320, i % 251, dtype=np.int16)) for i in range(n)]


def test_cascade_transcribes_uncertain_band_only(cfg, tmp_path, fake_whisper, monkeypatch):
    # clip_local READS evidence (fails open here: no torch) — so transcripts are built,
    # but only for the uncertain band, never for auto-accepts.
    c = _cfg_with_cache(cfg, tmp_path).with_overrides(**{
        "cascade.validator": "clip_local", "cascade.cache.enabled": False})
    segs = [_seg("hi", 0.90, 0.0), _seg("mid", 0.65, 20.0)]  # tau_hi=0.75 -> auto / uncertain
    res = run_cascade(segs, frames=[], cfg=c, chunks=_chunks())
    assert res.stats.n_auto == 1 and res.stats.n_unc == 1
    assert res.stats.n_transcripts == 1                 # ONLY the uncertain segment paid for ASR
    assert len(list((tmp_path / "tc").glob("*.txt"))) == 1
    assert all(v.keep for v in res.verdicts)            # clip_local failed open without torch
    # tier-2 reuse: same slice + same decode identity => cache hit with no backend at all
    monkeypatch.setitem(sys.modules, "faster_whisper", None)  # forces ImportError
    tiny = Asr(c, model="tiny.en", lazy=True)
    pcm = slice_pcm(_chunks(), segs[1].t_start, segs[1].t_end)
    assert tiny.transcribe_or_none(pcm) == f"fake:tiny.en:{pcm.size}"


def test_null_validator_skips_transcripts(cfg, tmp_path, fake_whisper):
    # The null baseline is "Tier-1 alone" (§1.8): it must not pay for evidence ASR it
    # ignores, or the measured cascade cost comparison is polluted.
    c = _cfg_with_cache(cfg, tmp_path).with_overrides(**{
        "cascade.validator": "null", "cascade.cache.enabled": False})
    res = run_cascade([_seg("mid", 0.65)], frames=[], cfg=c, chunks=_chunks(15.0))
    assert res.stats.n_unc == 1
    assert res.stats.n_transcripts == 0
    assert not (tmp_path / "tc").exists()               # tiny.en never even constructed a cache
    assert all(v.keep for v in res.verdicts)


def test_cascade_no_backend_no_crash(cfg, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "faster_whisper", None)  # force ImportError regardless of env
    c = _cfg_with_cache(cfg, tmp_path).with_overrides(**{
        "cascade.validator": "clip_local", "cascade.cache.enabled": False})
    res = run_cascade([_seg("mid", 0.65)], frames=[], cfg=c, chunks=_chunks(15.0))
    assert res.stats.n_transcripts == 0                 # attempted, honestly None
    assert all(v.keep for v in res.verdicts)            # nothing lost to a missing backend


def test_tier2_asr_honours_on_device_local_files_only(cfg, tmp_path, fake_whisper):
    # §1.5: nothing on the on-device path may open a socket. Tier-2's process ASR (not just the
    # cascade) must pin local_files_only under on_device_only=true, else `process` can download.
    from hindsight.tier2.pipeline import Tier2Runners

    c = _cfg_with_cache(cfg, tmp_path)
    assert c.get("on_device_only", True) is True
    assert Tier2Runners.build(c).asr.local_files_only is True
    cloud = c.with_overrides(**{"on_device_only": False})
    assert Tier2Runners.build(cloud).asr.local_files_only is False


def test_cascade_asr_honours_on_device_local_files_only(cfg, tmp_path, fake_whisper):
    # the cascade's tiny.en evidence pass must likewise forbid the network on-device.
    c = _cfg_with_cache(cfg, tmp_path).with_overrides(**{
        "cascade.validator": "clip_local", "cascade.cache.enabled": False})
    _FakeWhisperModel.last_local_files_only = None
    run_cascade([_seg("mid", 0.65)], frames=[], cfg=c, chunks=_chunks(15.0))
    assert _FakeWhisperModel.last_local_files_only is True   # on_device_only defaults to true
