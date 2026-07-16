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

**Validation (Video-1, 2026-07-14, on `plane_1.MP4` — 4K/59.94 fps aircraft-cabin POV: looking
around the cabin + out the window (ego motion) and reading a menu, hand turning pages (independent
object motion). Awaiting P2 sign-off).** All figures measured, none assumed:

- **Ego-compensation helps, and the numbers say by how much.** Versus a naive frame-diff, residual
  motion cuts the mean response 25 % (21.0 → 15.6, arbitrary 8-bit units) and, on the top-quartile
  *high-camera-motion* windows, 30 % (37.4 → 26.2). Correlation of the emitted score with global
  camera-shift magnitude drops from +0.57 to +0.46. So D2 fires on the menu-page/hand motion, not on
  the head pans — which is the whole point of the delta.
- **cv2's subpixel `phaseCorrelate` is materially better than the numpy fallback** (mean residual
  15.6 vs 24.8; high-ego 26.2 vs 42.3), even though the shift is rounded to an integer roll — its
  centroid refinement lands a better integer. OpenCV is a core dep, so the good path is the default
  path; the numpy path is a real but weaker fallback (they disagree on the integer shift ~88 % of
  the time). This validates the "use the cv2 subpixel path" call.
- **Two tempting refinements measured and REJECTED** (kept out to avoid unmeasured cleverness,
  CLAUDE.md §9): a Hanning window on the FFT (no gain, marginally worse); a subpixel `warpAffine`
  instead of the integer `np.roll` (<3 % residual change). The docstring that claimed the transform
  was "Hanning-windowed" was simply wrong and is corrected.
- **Ceiling, stated honestly:** because D2 compares one frame per 0.5 s window (2 Hz-effective,
  independent of decode fps — see below), the frames it differences are ~0.5 s apart, where head
  motion is large and non-translational (rotation/parallax/blur). That is why the residual is only
  *partly* decorrelated from ego motion (+0.46, not ~0).

**Walking-footage validation (Video-1, 2026-07-16, on `walk_1` = `IMG_6983.MOV`, 1080p/30 fps indoor
walking POV — the case the task specifically asked for, and the case D-2 exists for). This is the
finding that most matters for P2.** On *walking* footage, ego-compensation at the current 2 Hz sampling
cuts the motion response **only 12 %** (raw 46.8 → residual 41.2), versus 25–30 % on the seated clip —
even though the camera motion is 5× larger (mean shift 26 px vs 5.6 px). So **D2 as implemented is
still close to a walking-detector on walking footage**, which is exactly the failure D-2 claims to
cure. The cause is purely the 0.5 s frame spacing; a decode-fps sweep on the same clip proves it:

| motion frame spacing | mean inter-frame shift | ego-comp reduction |
|---|---|---|
| 500 ms (**2 Hz — D2 today**) | 26 px | **12 %** |
| 167 ms (6 Hz) | 15 px | 36 % |
| 67 ms (15 Hz) | 5 px | 46 % |
| 33 ms (30 Hz, adjacent frames) | 2.6 px | **49 %** |

At adjacent-frame spacing, translational phase correlation is accurate (small, un-blurred shift) and
ego-comp removes ~half the walking response — genuinely isolating object motion. **This makes the
"difference adjacent decoded frames, aggregate per window" change a measured necessity for the
walking case, not an optional nicety** — it roughly quadruples ego-comp effectiveness there. It
changes D2's cost row and interacts with the memory-driven `target_fps` cap (adjacent differencing
wants ≥15 fps for motion; the budget wants ≤4 fps), so it needs P2's sign-off and an architecture
call (e.g. let motion difference the two frames already present in a window rather than one). Flagged,
not smuggled in. *Mitigation already in place:* the causal running-percentile normaliser adapts to
sustained walking, so motion is not saturated on `walk_1` (mean 0.39), and the clip correctly yields
0 segments — an empty-kitchen walk carries no conversational/text salience.

**Decode tuning that came out of the same pass (for `data/manifest.yaml` / `configs/*`):** runtime
is decode-bound, so `target_fps` is a *memory* lever, not a speed one — the runner holds the whole
frame list (~2.8 MB/frame at 1280 px). Native 60 fps on a 4K clip ≈ 13 GB; `target_fps: 4` keeps peak
RSS ~1.2 GB with no loss of D1/D2 signal (they are 2 Hz-effective). `hindsight detect plane_1 --fps 4`
runs in ~73 s (80 s clip) with no OOM. **Caveat for P2:** decode subsampling also stretches the
*frame*-specified cadences of D3/D4 (face `cadence: 24` becomes ~6 s, not 1 Hz, at 4 fps) — harmless
while those detectors are model-gated/masked, but if they come online, cadence should be expressed in
seconds, not frames. That touches `base.CadencePolicy` (shared) and is flagged, not changed here.

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

## D-8 — Face detector backend: forced substitution to YuNet  ★ minor, backend-only

**Owner to sign off:** the D3/face detector owner.

**What changed.** When D3 (`face_presence`) is enabled it now uses **OpenCV YuNet**
(`cv2.FaceDetectorYN`, an ONNX model in `models/face/`) instead of the report's "res10 SSD or
MediaPipe Face Detection." It is **opt-in** (`configs/faces.yaml`); the default profile carries no
`model_path`, so face masks off and the gate keeps its Table 3.2-8 operating point (see the
operating-point note below for why enabling it is deferred to P2's sweep).

**Why (forced, not preferred).** On the current stack *neither* named backend is usable:
- **res10 SSD** is a Caffe model; **OpenCV 5 removed the Caffe importer** (`cv2.dnn.readNetFromCaffe`
  is gone), so the caffemodel cannot be loaded at all.
- **MediaPipe** ships an arm64-macOS wheel with only the **Tasks API** — `mp.solutions` (the legacy
  API D3 called) does not exist in it.

YuNet is OpenCV's own, officially-recommended res10 successor, ~230 KB, no extra heavy deps (cv2 is
already core). The presence score keeps the existing formula `f(max_conf, largest_face_area_frac)`;
only the detector under it changed. Score semantics are unchanged, so the fusion contract is intact.

**Report impact:** D3's method cell changes ("res10/MediaPipe" → "YuNet"). No spec is affected; it is
presence-only, no identity, same output range.

**Verification finding that matters more than the substitution (for the D5/voice + P2/gating owners).**
Enabling the real detectors on `plane_1_720p.mp4` (720p aircraft-cabin POV) exposed that **`webrtcvad`
is the wrong VAD for this footage**: the constant broadband cabin/engine bed is classified as speech
at **~87 % of frames even at aggressiveness 3** (0=99.9 %, 2=88 %, 3=87 %) — saturated, not
discriminative. Consequence in the funnel, measured:

| detector state on plane_1_720p | segments | reduction (target ≥ 0.80) |
|---|---|---|
| energy fallback + face masked | 0 | 1.000 — under-fires |
| webrtcvad + face masked | 1 × 60 s (max cap) | 0.255 — over-fires |
| webrtcvad + YuNet face (face correctly 0.0) | 1 × 24.5 s | 0.696 — still FAIL |

Face at a genuine 0.0 (checked, none present — this clip is not face-heavy) dampens the over-firing,
but voice saturation keeps salience above θ_off across the menu-reading stretch. **The gate is not at
fault; the VAD is.** Recommend either a noise-robust VAD (e.g. Silero) for wearable audio, or
validating gating on footage with clean conversation rather than a plane cabin. This is a strong
argument for the D-1 cascade too: a saturated Tier-1 cue is exactly what a validator would prune.

**Operating-point consequence (for P2 — this is the important one).** Turning face *on* is not free
at the gate: on face-less footage its 0.0 contribution at weight 0.20 uniformly lowers fused salience,
so θ_on (tuned in the masked-face regime) becomes too high. Measured on the deterministic `demo`
source in a full-deps env: **3 segments → 0**, fused max **0.725 → 0.580** (right at θ_on = 0.58).
The minimal-env reference (no backends → face masks off) is unchanged at 4 segments, so nothing
regressed there. The lesson is the D-1 lesson restated: **adding a detector requires re-sweeping the
operating point, not just enabling the backend.** Recommend P2 fold "face available" and "real VAD"
into the joint (θ_on, θ_off, weights) sweep before either is relied on. Until then, enabling these
backends changes *where* the gate fires, not *whether* it is correct.
