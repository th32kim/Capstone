# DESIGN DELTAS — prototype vs. the submitted Detailed Design report

The 30-page Detailed Design report is submitted and its Section 2 spec list is **frozen**. This
prototype deviates from it in the places listed below. Every one of these is a *design decision*,
not an implementation detail, and needs the relevant owner's sign-off before it enters the 498B
report. Nothing here should be silently absorbed.

Owners: **P1** Yuyoung Kim (capture / data contract) · **P2** Andy Lim (Tier-1 gating) ·
**P3** Tae Hong Kim (Tier-2) · **P4** Youngdo Jang (store / retrieval / app).

---

## D-1 — Two-stage cascade: an AI validator behind the Tier-1 gate  ★ major, new

**Owner to sign off:** P2 (primary), P3 (secondary — it borrows `whisper tiny.en`).

**What changed.** The submitted design has one gating decision: Tier-1 fuses six detector scores and
a hysteresis gate emits interest segments, full stop. The prototype adds a **second guard**: segments
whose salience lands in an uncertain band `[θ_on, τ_hi)` are handed to a model (CLIP zero-shot
on-device, or Claude in the opt-in cloud profile) which returns keep/drop with a reason. Segments
above `τ_hi` are auto-accepted with no model call at all.

**Why.** Because running a model over the whole video is the thing we are specifically trying not to
do. The cascade makes model cost `O(N_uncertain)` instead of `O(T)`. On a 30-minute session that is
roughly *tens* of model calls instead of *hundreds to thousands*, and the exact ratio is measurable
(`hindsight eval cascade`).

**The consequence that actually matters, and it changes P2's tuning target:**

> The validator can only **remove** candidates. It can never add one.
> So it can only raise precision, and can only lower recall.
> **Recall remains entirely Tier-1's responsibility.**

Therefore θ_on should be tuned **lower** than it would be for a bare Tier-1 — deliberately
over-producing candidates and letting the validator buy precision back. The operating point becomes a
**joint sweep over (θ_on, θ_off, τ_hi)** rather than P2 tuning thresholds in isolation. **This is the
change that most affects P2's section** and it should not be made behind Andy's back.

**Report impact:** FS5 (F1 ≥ 0.70) is now satisfied by a two-stage cascade, not one gate. It needs a
new decision matrix (validator: none / local CLIP / cloud LLM), a new figure (the cascade ablation),
and the cost-funnel QTA. That is a genuinely strong addition to §4.3 — but it must be *earned by the
ablation*, exactly as RRF was.

**Risk:** if the measured ΔF1 from the validator is small, the cascade is complexity we cannot
justify and it should be cut. `--validator null` must always work so that we can cut it cleanly.

---

## D-2 — Ego-motion compensation in the motion detector  ★ moderate

**Owner:** P2.

**What changed.** D2 (`motion`) now measures **residual** motion after compensating for the dominant
global camera translation, rather than raw frame-difference.

**Why.** First-person video is head-mounted. Raw frame-diff fires continuously while the wearer walks
and carries almost no information about whether anything interesting is happening. Residual motion
isolates *object/person* motion from *head* motion. Without this, the motion detector is close to a
walking-detector.

**Cost:** one `cv2.phaseCorrelate` + warp per frame on a downscaled grayscale image. Cheap, but it
does change the compute budget — which M2 measures rather than assumes.

**Report impact:** D2's algorithm description and its row in the compute-budget table both change.

---

## D-3 — New optional detector: `dwell` (D7), from IMU  ★ moderate, GoPro-only

**Owner:** P2, and P1 if it touches the data contract.

**What.** A seventh detector: angular velocity dropping below a threshold *after* a period above it —
"the wearer stopped walking and is now looking at something." Sourced either from the ego-motion
estimate already computed in D2, or, on GoPro, from the **GPMF** telemetry track (gyro/accel/GPS)
muxed into every MP4.

**Why.** For a wearable, dwell is one of the strongest and cheapest salience cues available, and it is
orthogonal to the six existing detectors.

**Hard caveat, verified against Open GoPro's docs:** GPMF metadata **cannot be read while the camera
is recording** — only after the file is written. So a GPMF-sourced `dwell` is an **offline-path
detector only** and cannot support the real-time gating claim (FS4 / NFS1). Availability-mask it. The
ego-motion-derived variant *is* available in real time and is the one to prefer if D7 is adopted.

**Report impact:** either a seventh row in Tier-1 (and a re-weighted fusion), or explicitly scoped as
"prototype-only, not in the essential path." Do not let it drift in silently.

---

## D-4 — CLIP reappears, in a different role

**Owner:** P3/P4 to note.

The report's embedding decision matrix scored CLIP-512 at 7.43 and **rejected it as the primary
embedder** in favour of MiniLM-384. The prototype uses CLIP again — but as a **zero-shot salience
validator**, which is a different job. This is not a contradiction, but if it appears in the report
without a sentence saying so, a reader will think we forgot our own decision matrix. Say it out loud.

---

## D-5 — Spec-ID hygiene: the sections' local IDs are not the frozen §2 IDs

**Owner:** P1 (integrator).

P3's section drafted against FS4–FS8 / NFS1–NFS7 and P4's against FS-9/10/11 / NFS-5/6/7. Neither
matches the frozen §2 master list. The prototype uses **only the frozen §2 IDs** — see
`docs/SPEC_MAP.md`. Two specific traps:

- **"30-day retention" is not NFS7.** In the frozen list NFS7 is *platform compatibility*
  (iOS 17+ / macOS / Linux). Retention/storage is **NFS6** (≥ 8 h of retained memories within 2 GB).
- **Retrieval latency is NFS2 (≤ 2 s at ≥ 10 000 memories)**, not the "≤ 500 ms" that P4's section
  used locally. The measured ~40 ms round-trip clears both, so nothing breaks — but cite the right ID.

---

## D-6 — Tightening the segment-count bound (optional, not an error)

The report derives **≤ 55 segments per 30-minute session** from the 6.5 s *structural* floor
(1.5 s minimum ACTIVE span + 2.0 s pre-roll + 3.0 s post-roll) against a ≥ 80 % reduction target:
`floor(0.20 × 1800 / 6.5) = 55`.

But the gate *also* enforces a **3.0 s minimum ACTIVE span**, which makes the *effective* retained
floor 8.0 s, giving `floor(360 / 8.0) = 45`. The report's 55 is therefore a **valid but conservative
upper bound**, not a mistake. If you want the tighter number, it is available and derivable — but it
requires stating that both rules bind. Either is defensible; pick one and be consistent.

---

## D-7 — If the team actually switches to GoPro, this is what reopens

Not a prototype delta — a **report** delta, listed here so it is not discovered in week 10.
Verified against Open GoPro's public documentation (MIT-licensed API, no approval process, control
over BLE / Wi-Fi / USB, several documented video-streaming paths):

| Frozen spec | Meta Ray-Ban (as submitted) | GoPro | Action if we switch |
|---|---|---|---|
| **FS1** POV video ≥ 720p, ≥ 24 fps | strained — relaxed to ≥ 15 fps to match the SDK's documented floor | clears it easily | **the FS1 relaxation should be reverted**, and the reason for relaxing it re-examined |
| **FS2** audio ≥ 16 kHz, sync ≤ 100 ms | fails on the glasses mic (8 kHz mono HFP, beamformed *toward the wearer* — adversarial to conversational recall). This is why DM-1 selected the iPhone mic. | in-container 48 kHz, natively A/V-synced | **DM-1's entire premise changes.** The decision matrix must be re-scored, not patched. |
| **FS4 / NFS1** real-time gating during capture | the DAT streams live to the phone | **the open question.** Streaming paths are documented, but whether a phone app can hold a continuous frame stream for a passive, all-day session is unverified. The realistic fallback is record-then-offload, which turns Tier-1 from *real-time gating* into *post-hoc gating*. | **verify against Open GoPro's compatibility tables before committing.** If it is offload-only, FS4's "while capture is ongoing" and NFS1 both need rewording, and the passive/always-on value proposition weakens. |
| **NFS4** no raw media to third parties | satisfied | satisfied — GoPro streams to its own Wi-Fi AP / USB, i.e. locally. **Materially better than DJI**, which was disqualified precisely because RTMP routing would have violated NFS4. | none |
| **NFS5** ≥ 30 min continuous | plausible | GoPro battery/thermal under continuous streaming is the risk | measure it |
| **NFS8** cost ≤ CAD $1 000 | ~$636 | a HERO is cheaper than the glasses; likely fine | recompute the BOM |

**Bottom line:** GoPro is *more* open than Meta, not less — the API is MIT-licensed with no approval
gate, and it ships IMU/GPS telemetry the glasses do not expose. The one thing to verify before
committing is continuous live frame access to a phone app. **None of this blocks the prototype**,
because the source is file-first. That was the point.

---

## D-8 — face_presence / text_presence backend selection made honest and config-driven

**Owner:** P2 (Video-2).

**What changed.**

1. `face_presence` previously ignored its own `backend` config value and unconditionally tried
   MediaPipe first, falling back to nothing. It now branches on `backend` (`mediapipe` |
   `opencv_dnn`) as configured, so config and behaviour agree. `configs/default.yaml`'s default
   moved from `opencv_dnn` (an unimplemented stub — the res10 SSD model files are not bundled, so
   this setting could never do anything but mask off) to `mediapipe`, the only backend with a
   real implementation.
2. **MediaPipe's legacy `mp.solutions.face_detection` API** (what this detector is written
   against) was removed upstream in favour of the new Tasks API. **Measured boundary**: the
   win_amd64 wheels for 0.10.18, 0.10.20 and 0.10.21 all still ship
   `mediapipe/python/solutions/face_detection.py` (wheel contents inspected); 0.10.30 — the next
   published release after 0.10.21 — no longer does. Runtime-verified endpoints: 0.10.14 builds
   `FaceDetection`; 0.10.35 raises `AttributeError: module 'mediapipe' has no attribute
   'solutions'`. An unpinned `mediapipe>=0.10` (as `requirements.txt` had it) resolves to the
   latest release and silently leaves face_presence availability-masked off forever.
   `requirements.txt` now pins `mediapipe>=0.10.9,<0.10.30` with a `python_version < "3.13"`
   marker — no release in the pinned range publishes cp313 wheels, and an unsatisfiable pin
   would otherwise hard-fail the entire requirements install on Python 3.13. On 3.13 the
   backend is simply absent and `detect` reports the reason (see below). The pin is mirrored
   in `pyproject.toml`'s `[full]` extra. When the detector cannot build, it now records a
   cause (`unavailable_reason`: not installed / legacy API removed / build failed) that
   `hindsight detect` prints — the AttributeError no longer vanishes into a generic mask-off.
3. `text_presence`'s `backend` config (`east` | `mser_swt`) was likewise accepted but never
   branched on — it always ran MSER regardless. A real EAST branch now exists
   (`cv2.dnn.readNet` + the standard EAST score/geometry decode, vectorised), gated on a
   configurable `detectors.text_presence.east_model_path` (relative paths resolve against the
   **repo root**, not the CWD, so results cannot vary by launch directory). The frozen EAST
   `.pb` model is **not bundled** — there is no canonical, trustworthy URL for it to hardcode
   a download from — so with no model file on disk (the default in this repo), `backend: east`
   **honestly falls back to MSER** and records both the backend that ran and the actual reason
   (`notes["text_backend"]` / `notes["text_backend_reason"]`, printed by `hindsight detect`),
   mirroring the existing VAD energy-fallback pattern. It never silently claims EAST ran, and
   it never prints a guessed cause.
4. **One backend per run.** The EAST and MSER score formulas are incommensurable (different
   count saturations and fusion weights), so the backend is decided once at construction and a
   run never mixes the two formulas in one score column: a per-window EAST inference error
   masks that window (availability mask, the pre-existing honest degrade) and is counted in
   `notes["text_east_errors"]`; after `east.max_consecutive_errors` failures the run stops
   paying for doomed EAST attempts and stays masked. EAST's box coverage is computed as a
   clipped **union** on the network input grid, so overlapping/out-of-frame boxes cannot
   inflate the area term.
5. **All new scoring constants live in config** (CLAUDE.md §1 non-negotiable 2): the EAST NMS
   IoU, count saturation, fusion weights and input size, and the pre-existing MSER count
   saturation and weights, are now `detectors.text_presence.east.*` / `.mser.*` `[TUNE]` keys.
   Backend names are validated in `config._validate` (and again in the detector constructors),
   so a typo'd backend fails loudly instead of silently running something else.

**Why.** CLAUDE.md §1 non-negotiable 2 ("No hard-coded thresholds... Code reads config.") and
non-negotiable 1 ("No fabricated numbers. Ever."), plus §3's availability-mask rule ("a detector
that could not run ... must be masked out, not scored 0"). Both detectors previously had a config
knob that did nothing, which is the same failure class as a hard-coded threshold — the config
lies about what the code will do.

**Report impact:** none to the frozen design; this is implementation catching up to what
`configs/default.yaml` already claimed. If EAST is later benchmarked against MSER for FS8/D4's
precision, note that the corpus run in this repo used the MSER fallback (no EAST weights present)
unless `east_model_path` is explicitly populated.
