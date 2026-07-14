# SPEC_MAP — the frozen §2 IDs, and what in this repo exercises each one

**Use these IDs and only these IDs.** P3's section drafted against a local FS4–FS8 / NFS1–NFS7 scheme
and P4's against FS-9/10/11 / NFS-5/6/7. Neither survives contact with the frozen §2 master list.
Two traps in particular: **NFS7 is *platform compatibility*, not 30-day retention** (storage is
**NFS6**), and **retrieval latency is NFS2 (≤ 2 s @ ≥ 10 000 memories)**, not the "≤ 500 ms" P4 used
locally. See `docs/DESIGN_DELTAS.md` D-5.

## Functional

| ID | Spec (frozen) | Class | Exercised by | Measured how |
|---|---|---|---|---|
| FS1 | POV video ≥ 1280×720, ≥ 24 fps *(relaxed to ≥ 15 fps to match the Meta SDK floor)* | Ess | `sources/clip.py` | `hindsight probe` prints res + fps |
| FS2 | Audio ≥ 16 kHz, A/V timestamp alignment within 100 ms | Ess | `sources/clip.py`, `clock.py` | `hindsight probe` prints **measured** max drift |
| FS3 | Six per-modality salience scores per processing window | Ess | `detectors/` | `out/scores/*.csv` — 6 score cols + 6 mask bits per 0.5 s row |
| FS4 | Fuse into one signal and segment the stream **while capture is ongoing** | Ess | `fusion/`, `gating/` | single-pass, causal normalisers, bounded queue — there is a test |
| FS5 | Gating accuracy **F1 ≥ 0.70** vs a hand-labelled corpus | Ess | `eval/gating.py` (+ `eval/cascade.py`) | segment-level IoU ≥ 0.5 match → P/R/F1 |
| FS6 | Speech transcription per retained segment | Ess | `tier2/asr.py` | transcript non-empty when VAD says speech |
| FS7 | Visual caption per retained segment | Ess | `tier2/caption.py` | caption non-empty |
| FS8 | On-screen text (OCR) | **Non-ess** | `tier2/ocr.py` | ocr_text; safe to disable if it blows the budget |
| FS9 | Named-entity extraction | Ess | `tier2/ner.py` | `entities[]` populated |
| FS10 | Summary + vector embedding per segment | Ess | `tier2/summarize.py`, `tier2/embed.py` | `len(embedding) == 384`, asserted |
| FS11 | Persist in a combined KG + vector store | Ess | `store/` | SQLite + FAISS both written; ingest test |
| FS12 | NL query → ranked list of memories | Ess | `retrieval/`, `api/`, `cli` | `hindsight ask` returns ranked hits |

## Non-functional

| ID | Spec (frozen) | Class | Exercised by | Measured how |
|---|---|---|---|---|
| NFS1 | Tier-1 at the full capture frame rate, no dropped frames | Ess | `detectors/`, `gating/` | `eval/latency.py`: ms/frame, duty %, queue depth, drop count |
| NFS2 | NL query returns within **2 s** for **≥ 10 000** memories | Ess | `retrieval/` | `eval/latency.py` p50/p95 at N = 10 k |
| NFS3 | Tier-2 keeps up: backlog must not grow | Ess | `tier2/pipeline.py` | mean per-segment latency ≤ mean segment inter-arrival — **derive it, don't assume it** |
| NFS4 | Gating on-device; no raw media to third parties without explicit consent | Ess | `configs/default.yaml` (`on_device_only: true`), `cascade/` | assert no socket on the gating/validation path in the on-device profile |
| NFS5 | ≥ 30 min continuous operation on one charge | Non-ess | *(Track B only)* | not measurable in Track A — print `NOT MEASURABLE (Track A)` |
| NFS6 | ≥ 8 h of retained memories within **2 GB** | Non-ess | `store/`, proxy transcode | measure bytes per retained-hour |
| NFS7 | **Platform**: iOS 17+; pipeline runs on macOS **and** Linux | Ess | CI | run the pipeline on both |
| NFS8 | Prototype hardware cost ≤ CAD $1 000 | Non-ess | — | BOM (~$636 as submitted) |

## The rule

`out/reports/measurements.md` tabulates every ID above against its **measured** value and a verdict of
`PASS` / `FAIL` / `NOT MEASURED`. There is no fourth verdict. A spec is never reported as satisfied
because the architecture looks capable of satisfying it.
