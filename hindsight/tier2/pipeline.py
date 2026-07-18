"""Tier-2 pipeline (CLAUDE.md §5): Phase A parallel, Phase B sequential, per surviving segment.

Phase A (concurrent, pure functions of the raw segment): ASR, captioning, OCR.
Phase B (sequential): NER -> extractive summary -> embedding (len == 384, asserted).

Runs ONLY on accepted segments (O(N_accepted)). Model runners are built once and reused. The
driver measures per-stage time and per-segment total so NFS3 (backlog does not grow) can be
*derived* from measurement, not assumed.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from ..contracts import AudioChunk, Frame, MemoryRecord, Segment, Verdict
from .asr import Asr, slice_pcm
from .caption import Captioner
from .embed import Embedder
from .ner import NerRunner
from .ocr import Ocr
from .summarize import extractive_summary


@dataclass
class Tier2Runners:
    asr: Asr
    caption: Captioner
    ocr: Ocr
    ner: NerRunner
    embed: Embedder
    cfg: object

    @classmethod
    def build(cls, cfg) -> "Tier2Runners":
        return cls(Asr(cfg), Captioner(cfg), Ocr(cfg), NerRunner(cfg), Embedder(cfg), cfg)

    def backends(self) -> dict[str, str]:
        return {"asr": self.asr.backend, "caption": self.caption.backend, "ocr": self.ocr.backend,
                "ner": self.ner.backend, "embed": self.embed.backend}


@dataclass
class Tier2Result:
    records: list[MemoryRecord] = field(default_factory=list)
    stage_seconds: dict[str, float] = field(default_factory=dict)
    per_segment_seconds: list[float] = field(default_factory=list)
    backends: dict[str, str] = field(default_factory=dict)


def _slice(seg: Segment, frames: list[Frame], chunks: list[AudioChunk]):
    segf = [f for f in frames if seg.t_start <= f.t_ms / 1000.0 <= seg.t_end]
    # audio MUST go through slice_pcm so the transcript-cache key matches the cascade's
    pcm = slice_pcm(chunks, seg.t_start, seg.t_end)
    peak = None
    if segf:
        target = seg.keyframe_ts[len(seg.keyframe_ts) // 2] if seg.keyframe_ts else seg.active_start
        peak = min(segf, key=lambda f: abs(f.t_ms / 1000.0 - target)).rgb
    return segf, pcm, peak


async def process_segment(seg: Segment, frames, chunks, runners: Tier2Runners,
                          verdict: Verdict | None, timings: dict) -> MemoryRecord:
    segf, pcm, peak = _slice(seg, frames, chunks)
    rec = MemoryRecord(seg.segment_id, seg.t_start, seg.t_end, seg.source, on_device=True)

    async def timed(name, fn, *a):
        t0 = time.perf_counter()
        out = await asyncio.to_thread(fn, *a)
        timings[name] = timings.get(name, 0.0) + (time.perf_counter() - t0)
        return out

    # Phase A — concurrent
    rec.transcript, rec.caption, rec.ocr_text = await asyncio.gather(
        timed("asr", runners.asr.transcribe, pcm),
        timed("caption", runners.caption.caption, peak),
        timed("ocr", runners.ocr.read, [f.rgb for f in segf]),
    )
    # Phase B — sequential
    text = f"{rec.transcript}\n{rec.caption}\n{rec.ocr_text}"
    rec.entities = await timed("ner", runners.ner.run, text)
    k = runners.cfg.get("tier2.summarize.k_sentences", 2)
    d = runners.cfg.get("tier2.summarize.damping", 0.85)
    it = runners.cfg.get("tier2.summarize.max_iter", 30)
    rec.summary = await timed("sum", extractive_summary, text, k, d, it)
    rec.embedding = await timed("emb", runners.embed.embed, rec.summary or text.strip())

    if verdict is not None:
        rec.salience_peak = seg.salience_peak
        rec.validator_verdict = ("auto_accept" if verdict.route == "auto_accept"
                                 else "null" if verdict.validator == "null" else "ai_keep")
        rec.validator_confidence = verdict.confidence
        rec.validator_reason = verdict.reason
    else:
        rec.salience_peak = seg.salience_peak
    rec.validate()  # d==384 assertion
    return rec


def process_segments(segments: list[Segment], frames, chunks, cfg,
                     verdicts: dict[str, Verdict] | None = None) -> Tier2Result:
    runners = Tier2Runners.build(cfg)
    timings: dict[str, float] = {}
    per_seg: list[float] = []
    records: list[MemoryRecord] = []
    verdicts = verdicts or {}

    async def run_all():
        for seg in segments:
            t0 = time.perf_counter()
            rec = await process_segment(seg, frames, chunks, runners, verdicts.get(seg.segment_id), timings)
            per_seg.append(time.perf_counter() - t0)
            records.append(rec)

    asyncio.run(run_all())
    return Tier2Result(records=records, stage_seconds=timings, per_segment_seconds=per_seg,
                       backends=runners.backends())
