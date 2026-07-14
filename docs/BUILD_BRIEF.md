# BUILD BRIEF — Hindsight Track A prototype

Work the milestones in order. Each has a **Definition of Done** and **the exact command that
proves it**. Do not start M(n+1) until M(n)'s proving command passes and its artifact exists.

A milestone is not done because the code runs. It is done because the command prints the number,
and the number is real.

---

## M0 — Inventory and migrate

The repo already has loose folders (`audio/`, `video/`, `fusion/`, …) and some working detector
code. **Read it before you move it.** Working code is worth more than a tidy tree.

**Do:**
1. Walk the existing tree. Write `docs/MIGRATION.md`: for every existing file — what it does, does
   it work, where it goes, or why it is deleted.
2. Restructure to the target tree below. Preserve any detector that already works; port it into the
   `Detector` protocol rather than rewriting it.
3. `pyproject.toml`, `requirements.txt`, `Makefile`, `.gitignore` (ignore `data/corpus/`, `out/`,
   `*.db`, `*.faiss`).

**Target tree:**
```
hindsight/
  contracts.py          # FROZEN dataclasses. everything imports from here.
  config.py             # loads + validates + freezes configs/*.yaml
  clock.py              # one monotonic timebase; window alignment
  sources/    __init__ clip.py webcam.py synthetic.py gpmf.py
  detectors/  __init__ base.py
              video/  scene_change.py motion.py face_presence.py text_presence.py
              audio/  voice_activity.py audio_energy.py
              imu/    dwell.py
  fusion/     linear.py
  gating/     hysteresis.py segmenter.py
  cascade/    router.py evidence.py cache.py budget.py
              validators/  null.py clip_local.py llm_claude.py
  tier2/      pipeline.py asr.py caption.py ocr.py ner.py summarize.py embed.py
  store/      schema.sql sqlite_store.py faiss_index.py kg.py retention.py
  retrieval/  engine.py rrf.py constraints.py
  api/        server.py
  eval/       gating.py cascade.py retrieval.py latency.py index_bench.py figures.py
  cli.py
configs/  default.yaml  onDevice.yaml  cloud.yaml
data/     corpus/  labels/  queries/  manifest.yaml
out/      scores/ segments/ verdicts/ records/ store/ figures/ reports/
tests/    docs/
```

**DoD:** `hindsight --help` lists every subcommand. `pytest -q` passes (even if trivially).
`docs/MIGRATION.md` exists and accounts for every pre-existing file.

---

## M1 — Contracts and sources

`contracts.py` is provided in this bundle — **use it as-is**. It is the frozen interface.

**Do:** implement `ClipSource` (MP4/MOV → `Frame` + `AudioChunk` streams on one monotonic clock),
`SyntheticSource`, `WebcamSource`. Decode audio to **PCM16 mono 16 kHz** regardless of container.
Video and audio timestamps must share a base and be independently paced (do not couple the audio
chunk rate to the video frame rate).

**DoD:**
```
$ hindsight probe data/corpus/clip01.mp4
clip01.mp4  1920x1080  29.97 fps  48000 Hz -> 16000 Hz  00:12:31
A/V drift over full clip:  max 18 ms   (FS2 requires <= 100 ms)   PASS
```
Drift must be **measured** (compare the last frame's `t_ms` against the last audio chunk's `t_ms`
after independent decode), not assumed.

---

## M2 — The six detectors (the first guard)

Each detector implements:
```python
class Detector(Protocol):
    name: str
    modality: Literal["video", "audio", "imu"]
    cadence: int            # 1 = every frame/window; 24 = every 24th frame
    def score(self, w: Window) -> float | None   # [0,1], or None if unavailable -> mask bit 0
```

Notes that will save you a day each:
- **D2 `motion` must compensate for ego-motion.** POV footage is head-mounted; raw frame-diff fires
  the entire time the wearer is walking. Estimate the dominant global translation
  (`cv2.phaseCorrelate` on grayscale, downscaled), warp, *then* diff. Emit **residual** motion.
  Keep the raw global-motion magnitude too — D7 needs it.
- **D3/D4 run at reduced cadence** (1 Hz / 0.5 Hz) and **hold their last value** between evaluations.
  A held value is `stale` — record staleness in the mask, do not silently pretend it is fresh.
- **D5 `voice_activity`** = `webrtcvad`, 30 ms frames @ 16 kHz. Window score = fraction of voiced
  frames in the 0.5 s window. If webrtcvad is missing or throws, fall back to an RMS-energy gate and
  **say so in the mask/logs** — a silent fallback that quietly degrades F1 is the worst outcome.
- **D6 `audio_energy`**: normalise against a *running percentile* (5th/95th), not a fixed dBFS
  threshold. Absolute levels vary wildly between mics.
- Normalisers are stateful and must be **causal** (no peeking at the future) if the online claim
  (FS4: "while capture is ongoing") is to mean anything. Write the test.

**DoD:**
```
$ hindsight detect clip01
wrote out/scores/clip01.csv   (1502 windows x 6 scores + 6 mask bits)
per-frame cost @ 29.97 fps:  scene 1.4  motion 2.1  face 0.5*  text 0.4*  vad 0.2  energy 0.1  = 4.7 ms/frame  (14.1% duty)
                             (* amortised over cadence)
```
The compute-budget line is a **measured** replacement for the design report's 4.62 ms/frame
allocation. Report both; if they disagree, that is a finding, not a bug to hide.

---

## M3 — Fusion and the hysteresis gate

Fusion: weighted linear sum, **availability mask applied, weights renormalised over available
detectors**. Gate: the four-state FSM with the Table 3.2-8 parameters (see `CLAUDE.md` §3.3 and
`configs/default.yaml`).

Then `segmenter.py`: pre-roll, post-roll, merge-gap, min ACTIVE duration, max duration, and
`close_or_merge_segment()`.

**Assertions that must be in the code, not just the tests:**
- no retained segment shorter than **6.5 s** (structural floor: 1.5 s min ACTIVE + 2.0 pre + 3.0 post)
- no retained segment longer than **60 s**
- `reduction = 1 - retained_duration / total_duration` is computed and printed every run
- segment count ≤ `floor(0.20 * T / 6.5)` — for a 30-min session, **≤ 55**

**DoD:**
```
$ hindsight gate clip01
12 segments   retained 118.4 s / 751.0 s   reduction 0.842   (target >= 0.80)  PASS
min 8.0 s   max 41.5 s   mean 9.9 s
wrote out/segments/clip01.jsonl
wrote out/figures/clip01_salience.png     # salience trace + shaded retained spans
```

---

## M4 — Corpus and ground truth (the gate on everything after this)

Nothing downstream can be *evaluated* until this exists. See `docs/CORPUS.md` for the protocol.

**Do:**
1. `data/manifest.yaml` — one entry per clip: id, path, duration, scene type, who labelled it.
2. `hindsight label clip01` — keyboard-driven review UI. Scrub, mark event spans, type a category.
   Writes `data/labels/clip01.csv` → `t_start,t_end,category,labeller`.
3. **Two labelling modes, and the distinction is critical:**
   - **assisted** — pre-seed the UI with Tier-1's proposals; the human accepts/rejects/adjusts. Fast.
     **Cannot be used to measure recall** — you never see the events the detector never proposed.
   - **dense** — the human labels every event from scratch, blind to the detector. Slow.
     **This is the only mode that can measure recall.**
   Require **at least one full clip labelled dense**, held out, and use it as the recall set.
   Mark the mode in the CSV. Refuse to compute recall from assisted labels — raise.
4. Two labellers on at least one clip; report **Cohen's κ**. If κ < 0.60 the labels are not
   trustworthy and the F1 number means nothing.

**DoD:** `hindsight eval gating` runs and prints a P/R/F1 table. The recall column is sourced from a
dense-labelled clip or it prints `NOT MEASURABLE (no dense labels)`.

---

## M5 — The second guard: cascade + validators

This is the new architecture. Read `CLAUDE.md` §4 in full before writing a line.

**Do:**
1. `cascade/router.py` — `τ_hi` auto-accept / uncertain band / shadow band.
2. `cascade/evidence.py` — keyframe selection (3 frames: post-pre-roll, peak, pre-post-roll),
   downscale to ≤512 px, JPEG q70; cheap `whisper tiny.en` transcript **for uncertain-band segments
   only**, cached by content hash and reused by Tier-2.
3. `validators/null.py` — accept all. This is the **ablation baseline**; write it first.
4. `validators/clip_local.py` — CLIP ViT-B/32 zero-shot against a positive/negative prompt bank
   (prompt bank lives in `configs/prompts.yaml`, not in code). On-device. NFS4-safe. **The default.**
5. `validators/llm_claude.py` — Claude multimodal, strict JSON, one retry on parse failure, then
   **fail open**. Enabled only under `configs/cloud.yaml`, which prints a consent banner.
6. `cascade/cache.py` — `sha256(segment_bytes + config_hash) -> Verdict`. Re-running eval costs $0.
7. `cascade/budget.py` — `max_calls_per_session`, `max_cost_usd`. On breach: log, stop, fail open.

**DoD:**
```
$ hindsight eval cascade clip01
                     P      R      F1     AI calls   $        wall
tier1 only         0.61   0.88   0.72        0     0.0000     0.0 s
+ clip_local       0.79   0.86   0.82        7     0.0000     2.1 s
+ llm_claude       0.86   0.85   0.85        7     0.0134    11.4 s

candidates 12   auto-accept 5   uncertain 7
naive AI-on-everything: 751 s @ 1.0 Hz = 751 calls   ->  cascade is 107x cheaper
recall delta vs tier1: -0.03 (validator can only remove; this is expected and bounded)
```
Those numbers are **illustrative of the format**, not of the result. Print what you measure.

---

## M6 — Tier-2

Six stages, Phase A parallel / Phase B sequential, per `CLAUDE.md` §5.

**DoD:**
```
$ hindsight process clip01
9 records   mean 5.8 s/segment   (ASR 4.9  cap 0.5  ocr 0.2  ner 0.02  sum 0.04  emb 0.01)
all embeddings d=384             PASS
mean segment inter-arrival 62.6 s  >  mean processing 5.8 s   -> no backlog (NFS3)  PASS
wrote out/records/clip01.jsonl
```
The last line is NFS3, proved from measurement rather than assumed. That is exactly the kind of
derived result this project wants.

---

## M7 — The store

`schema.sql` is in this bundle. SQLite = source of truth. FAISS = derived cache.

**Do:** ingest in one SQLite transaction (segment → `SEGMENT`; entities → `ENTITY` + `MENTION`;
co-mentions → `RELATION(CO_OCCURS)`), then append the vector to the FAISS sidecar under
`embedding_id`. Implement `retention.py`: tombstone (`live = 0`) → retrieval filters non-live
`embedding_id`s → periodic rebuild from live rows.

**DoD:** a test named `test_deleted_memory_unretrievable_before_rebuild` that:
deletes a segment, does **not** rebuild the index, runs a query that previously returned it, and
asserts it is gone. This is the project's privacy differentiator; if this test does not exist, the
claim is not true.

---

## M8 — Retrieval, API, CLI

`retrieve()` per `CLAUDE.md` §6. FastAPI: `POST /query`, `GET /memory/{id}`, `DELETE /memory/{id}`,
bound to **localhost only**. `hindsight ask "..."` for the CLI.

`constraints.py`: parse temporal ("last Tuesday", "yesterday", "this morning"), person, and place
constraints out of the query and apply them post-fusion.

**DoD:**
```
$ hindsight ask "what did the professor say about the data contract"
1. [0.041] 00:14:22-00:14:51  "…discussing the frozen interface between capture and processing"
           entities: Mike Cooper-Stachowsky (PERSON), E5 (GPE)
2. ...
round-trip 47 ms   (embed 12 | ner 8 | ann 0.6 | kg 3 | rrf 1 | sqlite 4 | ...)
```

---

## M9 — The measurement pass (this is where the report gets its numbers)

Every figure in the submitted design report that is currently labelled *"projected / illustrative
pending measurement (§4.1)"* becomes measurable here. Regenerate them with the **same filenames** so
they drop straight into 498B.

| Figure / table | What to measure | Command |
|---|---|---|
| Fig 3.4-2 — latency vs corpus size | Flat / IVF-Flat / HNSW / IVF-PQ single-query latency at N = 1k, 10k, 100k | `hindsight eval index` |
| Fig 3.4-4 — efSearch knee | recall@10 (vs the **Flat oracle**) and latency, sweeping efSearch ∈ {8,16,32,48,64,96,128} | `hindsight eval index` |
| Fig 3.4-5 — hybrid ablation | precision@5, nDCG@10, MRR for dense-only / KG-only / RRF on the labelled query set | `hindsight eval retrieval` |
| Table 3.4-7 — round-trip budget | measured per-stage ms | `hindsight eval latency` |
| P/R/F1 + reduction sweep | joint sweep over (θ_on, θ_off, τ_hi) | `hindsight eval gating --sweep` |
| **NEW** cascade ablation | the M5 table | `hindsight eval cascade` |
| **NEW** cost funnel | N_naive vs N_cand vs N_unc vs N_accept | `hindsight eval cascade` |

For the 10k/100k index benchmarks you will not have 100k real segments. `hindsight synth --n 100000`
generates them — but **do not draw uniform random unit vectors**, which makes ANN look artificially
easy. Embed 100k real sentences so the vectors have realistic cluster structure. Label every
synthetic-corpus figure as synthetic, in the caption, in the code.

**DoD:** `make figures` regenerates `out/figures/` from scratch, from measurements, with zero
hand-entered numbers, and `out/reports/measurements.md` tabulates each spec ID against its measured
value and a PASS / FAIL / NOT MEASURED verdict.

---

## Order of attack if time is short

M0 → M1 → M2 → M3 → **M4** → M5 → M6 → M7 → M8 → M9.

M4 (the corpus) is the real dependency. Everything after it is engineering; without it, every number
downstream is a guess, and a guess is worth zero marks and less than zero trust.
