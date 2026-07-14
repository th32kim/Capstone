# CLAUDE.md — Hindsight Prototype (Track A)

You are working on **Hindsight**, a passive AI memory / lifelogging system. This repo is
**Track A**: the Python reference implementation of the whole pipeline, running against
**recorded first-person video files**. Track B (the iOS app) is a separate repo and is not
your concern.

Read `docs/BUILD_BRIEF.md` for the milestone plan and the exact command that proves each
milestone done. Read `docs/DESIGN_DELTAS.md` before changing anything that the design report
already froze. Read `docs/SPEC_MAP.md` when you need to know which spec ID a thing satisfies.

---

## 0. The one-paragraph version

Point-of-view video comes in. A bank of **cheap, deterministic detectors** runs over the
*entire* stream and scores every 0.5 s window for salience. A **hysteresis gate** turns that
score into a small number of candidate "interest segments" — typically ≤ 20 % of the footage.
Only *then* does anything expensive run: an **AI validator** looks at the candidates the gate
was unsure about and drops the false positives; then a **Tier-2 pipeline** (ASR, captioning,
OCR, NER, summarisation, embedding) turns the survivors into structured `MemoryRecord`s;
then those land in a **SQLite + FAISS store** that answers natural-language queries by fusing
a dense vector channel and a knowledge-graph channel with **Reciprocal Rank Fusion**.

**The point of the architecture is that cost is shaped like a funnel.** Never run a model over
the whole video. That is the design, and it is also the thing being evaluated.

```
  video file ──► detectors (O(T),   cheap, always)   ──► 6 scores / 0.5 s window
                     │
                     ▼
                 fusion + hysteresis gate  ──► N_cand candidate segments   (~≤55 per 30 min)
                     │
                     ▼
       ┌── s ≥ τ_hi ────────────────────► AUTO-ACCEPT (no AI call)
       ├── θ_on ≤ s < τ_hi ─────────────► AI VALIDATOR (O(N_uncertain))  ── keep / drop
       └── s < θ_on ────────────────────► never emitted (shadow-logged for eval only)
                     │
                     ▼
                 Tier-2 (O(N_accepted), expensive) ──► MemoryRecord
                     │
                     ▼
                 SQLite (source of truth) + FAISS HNSW (derived cache)
                     │
                     ▼
                 retrieve(): dense ⊕ KG → RRF(k=60) → constraints → top-k
```

---

## 1. Non-negotiables

These are hard invariants. Violating one is a bug even if the tests pass.

1. **No fabricated numbers. Ever.** Every figure, table, or benchmark this repo emits must
   come from a measurement the code actually performed. If a benchmark has not been run, print
   `NOT MEASURED` — do not interpolate, do not "estimate", do not carry a plausible-looking
   number forward. Any constant that is an *assumption* must be tagged `# ASSUMED` in the config
   with a note on where it came from. This project has been burned by a fabricated analysis
   before; the rule is absolute.
2. **No hard-coded thresholds.** Every threshold, weight, cadence, and window size lives in
   `configs/*.yaml`. Code reads config. A grep for a bare float in a detector is a defect.
3. **`d = 384` is frozen.** The MiniLM-384 embedding dimension is the single agreed interface
   between the Tier-2 owner (P3) and the store owner (P4). `contracts.py` asserts it at runtime.
   Any code path that would change it must raise, not adapt.
4. **SQLite is the source of truth; FAISS is a derived cache.** FAISS `IndexHNSWFlat` has no
   in-place vector removal, so deletion is: tombstone in SQLite (one transaction) → retrieval
   filters out any `embedding_id` whose segment is no longer `live` → index rebuilt from live rows
   on a schedule. Never attempt `remove_ids` on the HNSW index. This cross-store deletion
   consistency is the project's headline privacy argument — it must actually work.
5. **On-device is the default; cloud is opt-in.** `on_device_only: true` in `configs/default.yaml`.
   In that mode nothing on the gating or validation path may open a socket. The cloud validator
   is enabled only by an explicit profile (`configs/cloud.yaml`) and prints a consent banner.
6. **Every stage writes an artifact and is independently re-runnable.** `detect` → `out/scores/`,
   `gate` → `out/segments/`, `validate` → `out/verdicts/`, `process` → `out/records/`,
   `ingest` → `out/store/`. No monolith. Re-running a stage must not require re-running the one
   before it.
7. **Determinism.** Same input + same config + same seed ⇒ byte-identical segments. Evaluation is
   meaningless otherwise. Seed everything; pin model revisions.
8. **Ablations are first-class.** `--validator null`, `--dense-only`, `--kg-only` must always
   work. The report needs these; they are not debug flags.

---

## 2. The source abstraction (why this repo does not care about the glasses)

The team currently has **Meta Ray-Ban glasses** and may switch to a **GoPro**. The prototype
must not care. A `Source` yields two time-aligned streams on **one monotonic clock**:

```python
Frame(t_ms: int, rgb: np.ndarray)              # HxWx3, uint8
AudioChunk(t_ms: int, pcm: np.ndarray)         # int16 mono, 16 kHz, 20 ms
```

Implementations, in priority order:

| Source | Status | Notes |
|---|---|---|
| `ClipSource` | **primary** | any MP4/MOV — GoPro, Meta, iPhone, phone-on-a-lanyard. This is what you build against. |
| `WebcamSource` | secondary | live webcam + mic, for fast iteration |
| `SyntheticSource` | tests | generated frames/audio, deterministic |
| `GpmfSource` | optional | GoPro telemetry (gyro/accel/GPS) muxed in the MP4 — see §7 |

Everything downstream of `Source` is source-agnostic. If the team switches hardware, exactly one
file changes. Do not leak `gopro`/`meta` conditionals past `hindsight/sources/`.

---

## 3. Tier-1: the first guard (cheap, deterministic, runs on everything)

Six detectors, each emitting a score in `[0, 1]` per **0.5 s decision window**, plus an
**availability mask** bit (a detector that could not run — no audio track, stale cadence — must
be masked out, not scored 0).

| # | Detector | Modality | Cadence | Method |
|---|---|---|---|---|
| D1 | `scene_change` | video | every frame | HSV histogram correlation drop between sampled frames (`cv2.compareHist`, CORREL), downscale to 160×90 first |
| D2 | `motion` | video | every frame | **ego-compensated residual motion** — estimate dominant global translation (`cv2.phaseCorrelate`), warp, then `absdiff`. See §3.1. |
| D3 | `face_presence` | video | **1 Hz** (every 24th frame @ 24 fps) | OpenCV DNN res10 SSD or MediaPipe Face Detection. Presence only, no identity. Zero-order hold between evaluations; mask marks staleness. |
| D4 | `text_presence` | video | **0.5 Hz** (every 48th frame) | EAST text detector, or MSER + stroke-width heuristic. Box count + text-area fraction + mean confidence. |
| D5 | `voice_activity` | audio | every window | `webrtcvad` (30 ms frames @ 16 kHz, aggressiveness configurable) with an **RMS-energy fallback** if webrtcvad is unavailable or errors. Window score = fraction of voiced chunks. **Highest fusion weight.** |
| D6 | `audio_energy` | audio | every window | short-time RMS in dBFS through a robust normaliser (running 5th/95th percentile) |
| D7 | `dwell` | IMU | optional | GoPro-only, see §7. Availability-masked off everywhere else. |

### 3.1 The one thing that is different about first-person video

**Ego-motion dominates.** In POV footage the camera is on someone's head. A naive frame-difference
motion detector fires continuously while the wearer walks and says nothing about whether anything
interesting is happening. So:

- **D2 measures *residual* motion** — motion left over *after* compensating for the dominant global
  camera motion. That isolates object/person motion from head motion.
- **D7 measures *dwell*** — the wearer's angular velocity dropping below a threshold *after* a period
  above it. "Stopped and looked at something" is one of the strongest salience cues a wearable has,
  and it is nearly free.

Both are prototype-stage improvements over the submitted design. They are logged in
`docs/DESIGN_DELTAS.md` — they need the Tier-1 owner's sign-off before they enter the report.

### 3.2 Fusion

Weighted linear sum over the six normalised scores, with the **availability mask applied and the
weights renormalised over the available detectors** (so an audio-only or video-only segment does not
collapse to a near-zero score). Weights are config, not code. They are a *starting point for a
sweep*, not a result.

### 3.3 Gating (the hysteresis FSM)

Four states. Parameters are from the design report, Table 3.2-8 — **use these exact values as the
starting point**, and put every one of them in config:

| Parameter | Value | Purpose |
|---|---|---|
| Decision window | 0.5 s | align detector outputs |
| θ_on (activation) | 0.58 | require clear evidence before opening a segment |
| θ_off (deactivation) | 0.38 | permit temporary dips without closing (0.20 hysteresis gap) |
| Activation persistence | 2 windows (1.0 s) | reject isolated spikes |
| Deactivation persistence | 3 windows (1.5 s) | reduce fragmentation |
| Pre-roll | 2.0 s | recover the event onset |
| Post-roll | 3.0 s | preserve the end of speech |
| Min duration | 3.0 s **on the ACTIVE span** (not the retained span) | reject unusable fragments |
| Merge gap | 2.0 s | join brief pauses within one event |
| Max duration | 60 s | bound Tier-2 work per segment |

**Structural invariant** (assert this in code): the shortest possible ACTIVE span is 1.5 s
(three windows to leave ACTIVE), so with a 2.0 s pre-roll and 3.0 s post-roll **no retained segment
can ever be shorter than 6.5 s**. That floor is what bounds the segment count: with a ≥ 80 %
reduction target, a 30-minute session retains ≤ 360 s, hence **≤ 55 segments**. Assert both.

---

## 4. The second guard: the AI validator (this is the new part)

The gate is tuned for **recall**. It will over-produce. The validator's job is to buy back
**precision** — and it is the only place a model is allowed to look at raw media.

```
s = segment.salience_peak
if   s >= tau_hi   : AUTO_ACCEPT      # confident. no AI call. free.
elif s >= theta_on : VALIDATE(seg)    # the uncertain band. one AI call.
else               : (gate never emitted it)
```

### The asymmetry that governs the whole design

> **The validator can only remove candidates. It can never add one.**
> Therefore it can *only* improve precision and can *only* hurt recall.
> **Recall is entirely Tier-1's job.** If the gate misses an event, no model recovers it.

Two consequences you must respect:

1. **Re-tune the gate for recall.** With a validator behind it, θ_on should be *lower* than it would
   be for a bare Tier-1. The operating point is a **joint sweep over (θ_on, θ_off, τ_hi)**, not
   independent tuning. `eval/gating.py` must support this.
2. **Fail open.** If the validator errors, times out, blows its budget, or returns unparseable
   output — `keep = True`. Never lose a memory to a flaky API. Degrading to fail-open degrades the
   system to Tier-1-only, which is a known-good state.

### Validators (pluggable, selected by config)

| Validator | Where it runs | Use |
|---|---|---|
| `null` | — | accepts everything. **The ablation baseline = Tier-1 alone.** |
| `clip_local` | on-device | **default.** CLIP ViT-B/32 zero-shot: score keyframes against a positive prompt bank (conversation, whiteboard, screen with text, face close-up, reading a document…) vs a negative bank (empty hallway, blurry wall, floor, ceiling, walking down a corridor…). `keep = (max_pos − max_neg) > margin`. No network. NFS4-safe. |
| `llm_claude` | cloud, **opt-in** | Claude (`claude-sonnet-4-6`), multimodal. Strict JSON out. Highest quality, costs money, requires explicit consent. |

*(CLIP was scored and rejected as the primary **embedder** in the design report. Using it here as a
zero-shot **validator** is a different job and not a contradiction — but say so explicitly in any
write-up, or it will look like one.)*

### Evidence pack (bounded, so cost is bounded)

The validator sees, and only sees:

- **≤ 3 keyframes**, ≤ 512 px long edge, JPEG q70: one at `t_start + pre_roll`, one at peak salience,
  one at `t_end − post_roll`.
- **A cheap transcript** — `whisper tiny.en` run on the segment audio, **only for uncertain-band
  segments**. Cache it by content hash: Tier-2 reuses it for accepted segments (and upgrades to
  `base.en`), so this work is never thrown away.
- **The detector evidence vector**: the 6 scores, the mask, duration, peak and mean salience.

### Verdict schema (frozen — see `contracts.py`)

```python
Verdict(keep: bool, confidence: float, reason: str, tags: list[str],
        trim: tuple[float, float] | None,   # may only SHRINK the segment, never grow it
        validator: str, cost_usd: float, latency_ms: float)
```

### Budget + cache (mandatory)

- Cache every verdict by `sha256(segment_bytes + config_hash)`. Re-running an eval must cost $0.
- `cascade.budget.max_calls_per_session` and `max_cost_usd`. On breach: log, stop calling,
  fail open for the remainder.

### The cost argument (this is a headline result — measure it, do not assert it)

```
N_naive  = ceil(T * f_sample)        # what "just let the AI watch the video" costs
N_unc    = |{ c : theta_on <= s(c) < tau_hi }|      # what the cascade actually costs
speedup  = N_naive / N_unc
```

`eval/cascade.py` must print `N_cand`, `N_unc`, `N_auto`, `N_naive` at `f_sample ∈ {0.2, 1.0}` Hz,
the measured `$` and wall-clock, and the measured ΔP / ΔR / ΔF1 versus the `null` baseline.
**That table is the justification for the entire architecture.** It is also the most likely thing to
be wrong, so it must be measured on the real corpus, not projected.

### Shadow band (evaluation only)

With `gating.emit_shadow: true`, also log — but do **not** retain — candidates in
`[θ_shadow, θ_on)` to `out/shadow.jsonl`. This is the only way to see the events the gate *missed*,
which is the only way to know whether recall is actually Tier-1-limited. Off in production.

---

## 5. Tier-2 (per surviving segment)

Phase A is **parallel** (three pure functions of the raw segment); Phase B is **sequential**.
Models are the ones the design report chose — do not substitute without logging a delta.

| Stage | Model (Track A) | Output field |
|---|---|---|
| ASR | `faster-whisper base.en`, int8 | `transcript` |
| Captioning | BLIP-base (ViT-B, CapFilt-L) | `caption` |
| OCR | Tesseract (Track A only; Track B uses Apple Vision) | `ocr_text` |
| NER | spaCy `en_core_web_sm` | `entities[] = [{text, type, span}]` |
| Summarisation | Extractive TextRank (k=2, d=0.85, ≤30 iters) — **default, zero-hallucination** | `summary` |
| Embedding | `all-MiniLM-L6-v2`, **d = 384**, L2-normalised | `embedding[384]` |

```python
async def process_segment(seg) -> MemoryRecord:
    r = MemoryRecord(seg.id, seg.t_start, seg.t_end, seg.source, on_device=True)
    r.transcript, r.caption, r.ocr_text = await asyncio.gather(   # Phase A — concurrent
        run_asr(seg.audio), run_captioning(seg.keyframe), run_ocr(seg.keyframes))
    text        = f"{r.transcript}\n{r.caption}\n{r.ocr_text}"
    r.entities  = run_ner(text)                                   # Phase B — sequential
    r.summary   = run_extractive_summary(text, r.entities)
    r.embedding = run_embedding(r.summary)                        # len == 384. assert it.
    return r
```

An abstractive summariser (DistilBART) and a cloud LLM are *opportunistic upgrades*, not defaults —
same adaptive-tier pattern as the validator. Extractive can't phrase a novel sentence, but it also
cannot hallucinate a memory that never happened, which matters more here.

---

## 6. The store and retrieval (the core contribution — the supervisor endorsed this specifically)

**Two files, zero servers**: one SQLite DB (metadata + graph tables) beside one serialised FAISS
index. Schema in `hindsight/store/schema.sql`.

- **Vector index**: `faiss.IndexHNSWFlat`, `d=384`, `M=16`, `efConstruction=200`, `efSearch=64`,
  inner product over L2-normalised vectors (cosine ≡ IP). Wrap in `IndexIDMap2` so `embedding_id`
  is ours. **Exact `IndexFlat` is kept as the recall oracle** — it is how recall@k is measured.
- **Knowledge graph**: `ENTITY` (with a `canonical_name` unique index for O(1) resolution),
  `MENTION` (SEGMENT↔ENTITY, many-to-many, carries modality), `RELATION` (typed, weighted edges).
  `CO_OCCURS` is derived free from shared mentions and is always populated. `WORKS_AT` / `LOCATED_IN`
  need relation extraction beyond NER — populate them opportunistically with an entity-type-gated
  rule pass over co-mentions, and **leave them empty rather than fake them**. Materialise the
  relevant neighbourhood into an in-memory NetworkX view at query time for bounded traversal.

```
retrieve(query, k):
    q       = embed(query)                       # MiniLM 384-d, L2-normalised
    ents    = extract_entities(query)            # spaCy
    vec     = hnsw.search(q, ef=64, n=50)        # dense channel
    seeds   = [kg.expand(kg.resolve(e), hops=2) for e in ents]   # symbolic channel, bounded BFS
    kg_hits = rank(seeds, by=recency * degree)
    score   = defaultdict(float)                 # Reciprocal Rank Fusion, k=60
    for r, (s, _) in enumerate(vec):     score[s] += 1 / (60 + r)
    for r, (s, _) in enumerate(kg_hits): score[s] += 1 / (60 + r)
    fused   = apply_constraints(sort_desc(score), query)   # time / person / place
    return hydrate(fused[:k])
```

RRF fuses **ranks, not scores**, because cosine distance and graph recency×degree are
incommensurable. That is the point — no fragile per-channel normalisation.

**Deletion (non-negotiable #4, restated because it is the differentiator):** tombstone in SQLite
in one transaction; retrieval filters any `embedding_id` not `live`; rebuild the index periodically.
There must be a test that a deleted memory is unretrievable **before** the index is rebuilt.

---

## 7. GoPro notes (only relevant inside `hindsight/sources/`)

Verified against Open GoPro's public docs — this is materially more open than the Meta toolkit:
the API is MIT-licensed with no approval process, works over BLE / Wi-Fi / USB, and documents
several video-streaming paths. GoPro also muxes a **GPMF** metadata track into every MP4 containing
gyroscope, accelerometer, GPS and some computed metrics.

Two things matter for us:

1. **GPMF is readable only *after* the file is written**, not while recording. So the `dwell`
   detector (D7) is an **offline-path detector only**. It cannot be relied on for the real-time
   gating claim. Availability-mask it and never let it into the essential path.
2. A GoPro would trivially clear the video and audio capture specs that the Meta glasses strain
   against (the 8 kHz beamformed HFP mic is why the iPhone mic was selected). But **whether a phone
   app can hold a continuous frame stream for a passive all-day session is the open question** —
   verify it against Open GoPro's compatibility tables before anyone commits. `docs/DESIGN_DELTAS.md`
   lists what a hardware switch would reopen in the report.

Neither question blocks this repo. That is the entire reason the source is file-first.

---

## 8. Commands

```bash
make setup                                    # env + models + spaCy + tesseract check
hindsight probe   data/corpus/clip01.mp4      # res, fps, sr, duration, measured A/V drift
hindsight detect  clip01 --config configs/default.yaml   # -> out/scores/clip01.csv
hindsight gate    clip01                      # -> out/segments/clip01.jsonl + salience plot
hindsight validate clip01 --validator clip_local         # -> out/verdicts/clip01.jsonl
hindsight process clip01                      # -> out/records/clip01.jsonl  (Tier-2)
hindsight ingest  clip01                      # -> out/store/{hindsight.db, index.faiss}
hindsight ask     "who did I talk to about the data contract"
hindsight label   clip01                      # human ground-truth labelling UI
hindsight eval    gating|cascade|retrieval|latency|index
make figures                                  # regenerate every report figure from measurements
```

---

## 9. Style

- Python 3.11+. `ruff` + `black`. Type hints everywhere; `mypy --strict` on `contracts.py`.
- Small, pure functions. Detectors are `(window) -> float`; they do not know about fusion.
- Log timings for every stage, always, into `out/timings.jsonl`. The report needs them.
- Prefer a boring, measured implementation over a clever, unmeasured one.
- When you are unsure whether something is a design decision or an implementation detail:
  it is a design decision, and it goes in `docs/DESIGN_DELTAS.md`.
