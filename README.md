# Hindsight — Track A prototype

Passive AI memory from first-person video. Cheap detectors find the interesting parts;
an AI validator confirms them; only the survivors get processed and stored.

**Start here:** `CLAUDE.md` (context + invariants) → `docs/BUILD_BRIEF.md` (what to build, in order)
→ `docs/STATUS.md` (what is built vs pending).

## Quickstart (no corpus, no heavy deps needed)

```bash
pip install -e .            # core only: numpy, PyYAML, typer
hindsight --help            # every subcommand
hindsight detect demo       # deterministic SyntheticSource -> out/scores/
hindsight gate   demo       # fusion + hysteresis -> out/segments/ (+ salience trace)
hindsight validate demo --validator null   # the second guard -> out/verdicts/
hindsight process demo      # Tier-2 -> out/records/
hindsight ingest  demo      # SQLite + FAISS sidecar -> out/store/
hindsight ask "conversation near the wearer"
hindsight eval cascade demo && hindsight eval figures
pytest -q                   # optional-backend integration tests may skip in a core-only environment
```

The special clip id **`demo`** runs the whole funnel on a deterministic synthetic source, so the
pipeline works with only numpy/PyYAML/typer installed. Real footage: `pip install -e .[full]`
(OpenCV/torch/faiss/av/spaCy/whisper/…), register a clip in `data/manifest.yaml`, then
`hindsight probe <file>` and use the clip id.

## Sample videos (set up locally — not in git)

Raw footage is **gitignored** (`/sample-videos/`, `data/corpus/`) — raw media never leaves the
device, and GitHub blocks files over 100 MB. **Each developer must set up their own footage:**

```bash
mkdir -p sample-videos          # not tracked; create it yourself
# drop your own MP4/MOV clips in here (GoPro, iPhone, Meta glasses, …)
pip install av                  # decode backend (or opencv-python)
hindsight probe sample-videos/<your-clip>.mp4
```

Then register each clip in `data/manifest.yaml` (that file *is* tracked; the `.mp4`s are not) and
drive the pipeline by its `id`. The `sample`/`clip01` ids in the manifest expect files you provide.

## The funnel, on a real clip

```bash
hindsight probe   data/corpus/clip01.mp4   # res, fps, sr, MEASURED A/V drift
hindsight detect  clip01                    # 6 detectors -> out/scores/
hindsight gate    clip01                    # fusion + hysteresis -> out/segments/
hindsight validate clip01 --validator clip_local
hindsight process clip01                    # Tier-2 -> out/records/
hindsight ingest  clip01                    # SQLite (truth) + FAISS (cache)
hindsight ask     "who did I talk to about the data contract"
hindsight serve                             # localhost query/delete API
```

## Design invariants held in code

- **No fabricated numbers.** A stage whose backend is absent prints `NOT MEASURED` / masks the
  detector off — it never invents a value. `hindsight eval figures` tabulates every spec ID as
  MEASURED / NOT MEASURED.
- **Source-agnostic:** `ClipSource` takes any MP4 — GoPro, Meta Ray-Ban, iPhone. Switch hardware,
  one file changes.
- **d=384 frozen; SQLite is truth, FAISS is a derived cache;** deletion tombstones in SQLite and is
  filtered from retrieval *before* the index rebuilds (see `test_deleted_memory_unretrievable_before_rebuild`).
- **On-device by default;** the cloud validator is opt-in (`configs/cloud.yaml`) behind a consent banner.
