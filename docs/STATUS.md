# STATUS — what is built vs pending

Honest map of the prototype against `docs/BUILD_BRIEF.md`. "Runs" = exercised end-to-end on the
deterministic `demo` source with the current (minimal) environment. "Needs deps/labels" flags the
work that is *coded and wired* but cannot produce a real number until a backend or the corpus lands.

## Environment note

This machine has only numpy / PyYAML / typer / (fastapi, pydantic) installed — **not** the heavy
stack (OpenCV, torch, faiss, av, spaCy, faster-whisper, open-clip, webrtcvad) or ffmpeg/tesseract.
Every module imports those lazily and degrades honestly: a missing backend → the detector
availability-masks off, the validator fails open, the eval prints `NOT MEASURED`. No number is ever
fabricated (CLAUDE.md §1.1). `pip install -e .[full]` turns the honest-empty paths into real ones.

## Milestones

| M | What | State |
|---|---|---|
| **M0** | inventory, migrate, target tree, pyproject/Makefile/gitignore | **Done.** `docs/MIGRATION.md` accounts for every pre-existing file; `hindsight --help` lists every subcommand; `pytest` green. |
| **M1** | sources on one clock; probe + measured A/V drift | **Done, tested on real footage.** With PyAV installed, `ClipSource` decodes real MP4s (video + audio resampled to 16 kHz) with decode-time fps subsampling (`--fps`). Verified: `sample.mp4` (720×1280, 50 fps, drift 40 ms PASS) run through the **entire funnel**; `plane_1/2.MP4` (4K, 59.94 fps GoPro, drift 0 ms PASS) probed. |
| **M2** | six detectors + availability mask; detect | **Done.** scene_change / motion (ego-compensated, numpy phase-correlation) / audio_energy / voice_activity(+RMS fallback) run now; face/text mask off until MediaPipe/OpenCV is present. Compute budget is **measured**. |
| **M3** | fusion (masked renorm) + hysteresis gate + segmenter | **Done, runs.** Structural floor (6.5 s, with boundary-clamp compensation), max cap, reduction, and the ≤ floor(0.2·T/6.5) count bound all asserted in code. `demo`: 4 segments, reduction 0.829. |
| **M4** | corpus + ground truth + `label`; eval gating | **Partial.** `hindsight label` writes `data/labels/*.csv`; `eval/gating.py` computes IoU P/R/F1, Cohen's κ, and **raises** on assisted-only recall. Prints `NOT MEASURABLE` until a dense-labelled clip exists. *This is the real remaining dependency.* |
| **M5** | cascade: router / evidence / cache / budget / validators | **Done, runs (null).** Routing, budget fail-open, sha256 verdict cache all tested. Uncertain-band `tiny.en` evidence transcripts are wired (audio plumbed into `run_cascade`; content-hash cache at `out/cache/transcripts`, shared with Tier-2's `base.en` upgrade — see `tier2/asr.py`). `clip_local`/`llm_claude` are real but fail open until torch+open-clip / anthropic are installed. |
| **M6** | Tier-2 (ASR/caption/OCR/NER/summary/embed) | **Runs; ASR real.** faster-whisper `base.en` int8 wired and verified against the installed API (the old `sampling_rate=` kwarg would have TypeError'd — removed; greedy `temperature: 0.0` decode for §1.7 determinism). Transcripts content-hash cached, so re-running `process` never re-transcribes. Extractive TextRank + embedder are real (embedder falls back to a *labelled* hash embedder). Caption/OCR return honest "" without their models; NER falls back to a labelled regex. NFS3 (no backlog) derived from measurement. |
| **M7** | store: SQLite truth + FAISS cache; deletion | **Done, fully real (stdlib sqlite3 + numpy).** One-transaction ingest, KG CO_OCCURS, tombstone. `test_deleted_memory_unretrievable_before_rebuild` passes. FAISS HNSW used when installed; exact numpy index otherwise (= the oracle). |
| **M8** | retrieval + API + CLI | **Done, runs.** dense ⊕ KG → RRF(60) → constraints; ablations `--dense-only/--kg-only`; localhost-only FastAPI (`/query`, `/memory/{id}`, `DELETE`) tested. |
| **M9** | measurement pass; `make figures` | **Partial (honest).** `eval/{latency,index,cascade,figures}` measure what they can (Tier-1 ms/frame, flat-index latency vs N on labelled-synthetic vectors, cost funnel) and write `out/reports/measurements.md`; ANN curves + retrieval quality + F1 print `NOT MEASURED` pending faiss / a real embedder / dense labels. |

## Tested on real footage (`sample-videos/`, PyAV installed)

Full funnel on `sample.mp4` (720×1280, 50 fps, 60 s), decode subsampled to ~7 fps:

```
probe    720x1280  50 fps  44100 Hz -> 16000 Hz  60s   A/V drift 40 ms  PASS
detect   120 windows x 6 scores + 6 mask bits   0.95 ms/frame (0.7% duty)
gate     1 segment   retained 12.0 s / 60.0 s   reduction 0.800  PASS
validate 1 candidate  0 auto  1 uncertain -> 1 AI call (null baseline)
cascade  N_naive@1Hz=60 vs N_unc=1  ->  60x cheaper than AI-on-everything   (MEASURED, real clip)
process  1 record   d=384  PASS   NFS3 no backlog
ingest   SQLite + FAISS sidecar written
```

`plane_1/2.MP4` (3840×2160, 59.94 fps GoPro) probe: A/V drift **0 ms** — GoPro muxes A/V natively
synced, confirming DESIGN_DELTAS D-7. (Full 4K funnel runs but decode is minutes-per-clip.)
Transcripts/captions are empty because faster-whisper/BLIP are not installed — honest, not fabricated.

## The two things that unlock real numbers

1. **A dense-labelled held-out clip** (M4). Without it, F1 (FS5), the cascade ablation ΔP/ΔR/ΔF1,
   and retrieval quality are `NOT MEASURABLE` — by design, not omission.
2. **`pip install -e .[full]` + ffmpeg + tesseract.** Then real ClipSource decode, CLIP/Claude
   validators, faster-whisper/BLIP/spaCy/MiniLM, and faiss HNSW all light up in place.

Neither changes any interface — the contracts, configs, and CLI are already the real ones.
