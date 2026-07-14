# MIGRATION — inventory of the pre-existing tree and where every file went

M0 requires this: *"for every existing file — what it does, does it work, where it goes, or why it
is deleted."* Below is the state of the repo **before** this milestone and the disposition of each
path. "Works" means it did something useful; a `print("Hello, World!")` stub does not.

## Pre-existing files

| Path (before) | What it was | Works? | Disposition |
|---|---|---|---|
| `CLAUDE.md` | Project context + invariants (the spec) | n/a (doc) | **Kept** at repo root, unchanged. It is the source of truth. |
| `README.md` | Quickstart | n/a (doc) | **Kept**, unchanged. |
| `LICENSE` | MIT license | n/a | **Kept**. |
| `Makefile` | `setup/lint/test/pipeline/eval/figures` targets | targets valid, commands not yet implemented | **Kept**. Targets now resolve as modules land. |
| `requirements.txt` | Full runtime dep list | n/a | **Kept**. Mirrored into `pyproject.toml [full]`. |
| `.gitignore` | Ignores `out/ data/corpus/ *.db *.faiss …` | yes | **Kept + extended** (`.DS_Store`, caches, `*.egg-info`). |
| `hindsight/__init__.py` | Empty package marker | yes | **Kept** (package version added). |
| `hindsight/contracts.py` | **FROZEN** dataclasses / protocols (the interface) | yes | **Kept verbatim.** Everything imports from here; nothing edits it. |
| `hindsight/store/schema.sql` | SQLite schema (source of truth) | yes (DDL) | **Kept verbatim**, wired to `store/sqlite_store.py`. |
| `configs/default.yaml` | Full parameterisation w/ provenance tags | yes | **Kept**, loaded by `config.py`. |
| `configs/onDevice.yaml` | On-device profile (`extends: default.yaml`) | yes | **Kept**; `extends` now resolved by `config.py`. |
| `configs/cloud.yaml` | Opt-in cloud profile (`extends: default.yaml`) | yes | **Kept**; consent banner enforced in CLI. |
| `configs/prompts.yaml` | CLIP validator prompt bank | yes | **Kept**, read by `cascade/validators/clip_local.py`. |
| `data/manifest.yaml` | One entry per corpus clip | yes | **Kept**, read by config/corpus tooling. |
| `data/queries/queries.yaml` | Labelled query set for retrieval eval | yes | **Kept**, read by `eval/retrieval.py`. |
| `data/corpus/`, `data/labels/` | Empty dirs (gitignored corpus, tracked labels) | n/a | **Kept** (empty). |
| `docs/BUILD_BRIEF.md`, `docs/CORPUS.md`, `docs/DESIGN_DELTAS.md`, `docs/SPEC_MAP.md` | Planning docs | n/a | **Kept**. |
| `tests/__init__.py` | Empty | yes | **Kept**; real tests added under `tests/`. |
| `sample-videos/sample.mp4` (25 MB) | Sample POV clip | usable footage | **Kept in place** (see note). |
| `sample-videos/plane_1.MP4` (879 MB), `plane_2.MP4` (484 MB) | User-recorded POV footage | usable footage | **Kept in place** (see note). |
| `fusion/main.py` | `print("Hello, World!")` scaffold stub | **no** | **Deleted.** Real fusion lives in `hindsight/fusion/linear.py`. |
| `video/main.py` | `print("Hello, World!")` scaffold stub | **no** | **Deleted.** Real video detectors live in `hindsight/detectors/video/`. |
| `audio/main.py` | `print("Hello, World!")` scaffold stub | **no** | **Deleted.** Real audio detectors live in `hindsight/detectors/audio/`. |
| `.DS_Store` (repo root + parent) | macOS Finder metadata | n/a (junk) | **Deleted** + gitignored. |
| `{configs,docs,hindsight,tests,data` (+ nested) | Empty dirs from a failed brace-expansion `mkdir` in a non-brace shell | **no** | **Deleted.** |
| `out/` | Empty output dir (gitignored) | n/a | **Kept**; run-time subdirs (`scores/ segments/ …`) created on demand. |

## Note on `sample-videos/`

These three MP4s are **real footage I did not create** — most likely the team's own POV test
clips (`plane_1`, `plane_2` look like GoPro/POV recordings). They are **not** deleted. They are the
raw material for the M4 corpus. Recommended next step (not done automatically, to avoid moving a
user's only copy): register one as `clip01` in `data/manifest.yaml` and point `ClipSource` at it, or
copy it into the gitignored `data/corpus/`. `hindsight probe sample-videos/sample.mp4` accepts an
explicit path today.

## New scaffolding created in M0

The target tree from `BUILD_BRIEF.md` now exists under `hindsight/` (`sources/`, `detectors/{video,
audio,imu}/`, `fusion/`, `gating/`, `cascade/{validators}/`, `tier2/`, `store/`, `retrieval/`,
`api/`, `eval/`, `cli.py`) plus `pyproject.toml`. Modules whose backends are heavy (CV/ML/decode)
import those backends lazily and availability-mask themselves when a backend is absent, so
`hindsight --help` and `pytest -q` pass in a minimal environment.
