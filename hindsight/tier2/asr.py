"""ASR — faster-whisper base.en, int8 (CLAUDE.md §5, FS6) + the shared transcript cache.

The cache is the M5 contract: the cascade's cheap tiny.en pass (uncertain-band segments
ONLY) and Tier-2's base.en upgrade hash the SAME audio slice — both go through
:func:`slice_pcm`, one function, never typed twice — keyed sha256(pcm bytes || decode
identity). The decode identity covers everything that changes the OUTPUT (model, language,
beam, temperature, compute type, device), so a config change can never serve a transcript
the current settings would not produce (§1.7). Re-running `validate` or `process` with the
same settings costs zero ASR recompute. A cached "" is a real result (silence) and is
distinguished from a miss by file existence; writes are atomic so a killed run can never
fabricate one.

If faster-whisper is unavailable, Tier-2's transcribe() returns "" — an honest empty,
never a fabricated sentence — and the cascade's transcribe_or_none() returns None so the
evidence pack says "could not run" rather than "silent". Decoding is greedy
(temperature 0.0, from config): determinism is non-negotiable (CLAUDE.md §1.7).
"""

from __future__ import annotations

import hashlib
import os
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from .._deps import optional
from ..contracts import AudioChunk


def slice_pcm(chunks: Iterable[AudioChunk], t_start: float, t_end: float) -> np.ndarray:
    """The one segment-audio slicer. The cascade and Tier-2 MUST slice identically or
    their cache keys diverge and the tiny.en work is thrown away — so both call this."""
    parts = [c.pcm for c in chunks if t_start <= c.t_ms / 1000.0 <= t_end]
    return np.concatenate(parts).astype(np.int16) if parts else np.empty(0, dtype=np.int16)


class TranscriptCache:
    """sha256(pcm bytes || decode_id) -> transcript. File existence == hit ("" is a real result)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @staticmethod
    def key(pcm: np.ndarray, decode_id: str) -> str:
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(pcm).tobytes())
        h.update(decode_id.encode("utf-8"))
        return h.hexdigest()

    def _file(self, key: str) -> Path:
        return self.path / f"{key}.txt"

    def get(self, key: str) -> str | None:
        f = self._file(key)
        return f.read_text(encoding="utf-8") if f.exists() else None

    def put(self, key: str, text: str) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        # atomic: existence == complete result. A run killed mid-write must never leave a
        # truncated file that every later run serves as a real transcript (§1.1).
        tmp = self._file(key).with_suffix(".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, self._file(key))


class Asr:
    def __init__(self, cfg=None, model: str | None = None, *, lazy: bool = False,
                 local_files_only: bool = False) -> None:
        self.model_name = model or (cfg.get("tier2.asr.model") if cfg else None) or "base.en"
        self.compute_type = (cfg.get("tier2.asr.compute_type") if cfg else None) or "int8"
        # "cpu" is the report's premise (int8 CPU). faster-whisper's default device="auto"
        # picks CUDA on any NVIDIA machine and then dies at encode() if the CUDA runtime
        # (cublas64_12.dll) is absent — pin it unless config explicitly opts into cuda.
        self.device = (cfg.get("tier2.asr.device", "cpu") if cfg else "cpu")
        self.beam_size = int(cfg.get("tier2.asr.beam_size", 5)) if cfg else 5
        self.temperature = float(cfg.get("tier2.asr.temperature", 0.0)) if cfg else 0.0
        self.language = (cfg.get("tier2.asr.language", "en") if cfg else "en") or None
        if self.model_name.endswith(".en"):
            # .en models are English-only; faster-whisper silently coerces the language anyway.
            # Making it explicit keeps decode_id honest: the EFFECTIVE language, not the wish.
            self.language = "en"
        # local_files_only=True forbids any network fetch inside model build. The router sets
        # it from on_device_only: nothing on the validation path may open a socket (§1.5).
        self.local_files_only = local_files_only
        # Everything that changes decode OUTPUT participates in the cache identity — a config
        # change must never be served a transcript the current settings would not produce.
        self.decode_id = (f"{self.model_name}|{self.language}|{self.beam_size}"
                          f"|{self.temperature}|{self.compute_type}|{self.device}")
        cache_dir = (cfg.get("tier2.asr.transcript_cache", "out/cache/transcripts")
                     if cfg else "out/cache/transcripts")
        self.cache = TranscriptCache(cache_dir)
        self._model = None
        self._build_attempted = False
        self._warned = False
        self.backend = "unavailable"
        self.last_error = ""
        if not lazy:
            self._ensure_model()

    def _fail_loudly(self, msg: str) -> None:
        """Backend present but broken is the invisible failure — a silent fallback that
        quietly degrades output is the worst outcome (M2). Say so, once, on stderr.
        (A missing module stays quiet: that is the normal honest-degradation path.)"""
        self.last_error = msg[:300]
        if not self._warned:
            print(f"[asr:{self.model_name}] {self.last_error} -> transcript unavailable",
                  file=sys.stderr)
            self._warned = True

    def _ensure_model(self) -> bool:
        """Build the model at most once; a failed build is not retried per segment."""
        if self._model is not None:
            return True
        if self._build_attempted:
            return False
        self._build_attempted = True
        fw = optional("faster_whisper")
        if fw is None:
            return False
        try:
            self._model = fw.WhisperModel(self.model_name, device=self.device,
                                          compute_type=self.compute_type,
                                          local_files_only=self.local_files_only)
            self.backend = "faster_whisper"
            return True
        except Exception as exc:  # noqa: BLE001
            self._model = None
            self._fail_loudly(f"model build failed ({type(exc).__name__}: {exc})")
            return False

    def transcribe(self, pcm: np.ndarray) -> str:
        """Tier-2 contract: always a str; "" for silence AND for no-backend (honest empty)."""
        out = self.transcribe_or_none(pcm)
        return out if out is not None else ""

    def transcribe_or_none(self, pcm: np.ndarray) -> str | None:
        """Cache-first. None == could not run (no audio, or no backend and no cached result) —
        the cascade uses this so Evidence.transcript distinguishes 'unavailable' from 'silent'."""
        if pcm is None or len(pcm) == 0:
            return None
        key = self.cache.key(pcm, self.decode_id)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        if not self._ensure_model():
            return None
        # PCM16 mono @ 16 kHz (the frozen contract) is exactly what faster-whisper expects
        # for a raw array — it takes no sample-rate argument, so never pass one.
        audio = pcm.astype(np.float32) / 32768.0
        try:
            segments, _info = self._model.transcribe(
                audio, language=self.language, beam_size=self.beam_size,
                temperature=self.temperature)
            text = " ".join(s.text.strip() for s in segments).strip()
        except Exception as exc:  # noqa: BLE001 — a decode failure is "could not run", never a crash
            self._fail_loudly(f"decode failed ({type(exc).__name__}: {exc})")
            return None
        self.cache.put(key, text)
        return text
