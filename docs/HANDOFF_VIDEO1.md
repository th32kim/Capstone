# HANDOFF — open items from Video-1 (deterministic CV + decode)

Notice board from the **Video-1** owner (sources/clip.py, detectors/video/{motion,scene_change}).
Everything here is *outside* Video-1's remit — it needs another owner's decision, sign-off, a shared
fix, or a setup/permission step this owner cannot land alone. Grouped by who acts. Evidence is
measured (see `docs/DESIGN_DELTAS.md` D-2 / D-8 for the numbers).

Last updated: 2026-07-16.

---

## For P2 (Andy — Tier-1 gating)

1. **Sign off D-2 (ego-motion compensation).** Validated on real footage. Headline finding you need to
   see: on **walking** footage (`walk_1`) ego-comp cuts the motion response **only 12 %** at the
   current 2 Hz window sampling — D2 today is still close to a walking-detector on the very case it
   exists for. An fps sweep proves the cause is the 0.5 s frame spacing (12 % → 49 % at adjacent
   frames). **Decision needed:** adopt the "difference adjacent decoded frames, aggregate per window"
   change to D2. It changes D2's cost row and interacts with the memory-driven `target_fps` cap
   (motion wants ≥15 fps, the budget wants ≤4 fps) — so it's an architecture call, not a Video-1 patch.
   One low-cost option: let motion difference the *two* frames already present in a 4 fps window
   instead of one. See D-2 table.

2. **Re-sweep the operating point (θ_on, θ_off, weights) with the new detectors folded in.** Enabling
   face (`configs/faces.yaml`) shifts the operating point: on face-less footage its 0.0 at weight 0.20
   lowers fused salience, so θ_on=0.58 is now too high (measured: `demo` 3 → 0 segments, fused max
   0.725 → 0.580). Real VAD does the opposite (over-fires — see audio item). This is the D-1 joint
   sweep; it must run on the labelled corpus before face or webrtcvad is relied on by default.

3. **Sign off D-8** (face backend substitution to YuNet — mostly your concern via the operating point).

## For the audio / VAD owner (D5 voice_activity)

4. **webrtcvad is unsuitable for constant-broadband wearable audio.** On `plane_1` cabin noise it
   marks **~87 % of frames voiced even at aggressiveness 3** (0=99.9 %, 2=88 %, 3=87 %) — saturated,
   not discriminative — which makes the gate over-fire (reduction 0.255, FAIL). Recommend evaluating a
   noise-robust VAD (e.g. Silero). Evidence in D-8.
5. **webrtcvad is install-gated, not config-gated.** `voice_activity` uses webrtcvad whenever the
   package is importable, regardless of `detectors.voice_activity.backend`. So the default profile is
   *not* reproducible across machines (energy fallback vs webrtcvad depending on install). Add a real
   config toggle so the default behaviour is deterministic team-wide.

## For the D3/D4 detector owner (face / text)

6. **`test_unavailable_detectors_masked_not_zeroed` fails in a full-deps env.** Its premise — face and
   text never have a backend — is now false: with cv2 installed `text_presence` (MSER/SWT) runs, and
   with the YuNet model present `face_presence` runs, so neither masks off. Make the test
   backend-independent, e.g. assert the mask-not-zero invariant on a genuinely-unavailable modality
   (an audio-less window → voice/audio must mask). It passes in the canonical minimal env today.

## For the shared Tier-1 runner owner (`hindsight/detectors/__init__.py`)

7. **`run_detectors` materialises the whole frame list** (`list(source.frames())`), so a long 4K clip
   OOMs even at `target_fps: 4` (a 30-min clip ≈ 25 GB). `target_fps` bounds short/medium clips
   (~1.2 GB for the 80 s clips); hours-long passive sessions need windowed/streaming processing.
8. **Cadence is expressed in frames, so decode subsampling stretches D3/D4's effective rate** (face
   `cadence: 24` becomes ~6 s, not 1 Hz, at 4 fps). Consider cadence-in-seconds so the reduced-rate
   detectors are decode-fps-agnostic like D1/D2.

## Setup / environment (flags & provisioning others will hit)

9. **`setuptools<81` is required** — webrtcvad imports `pkg_resources`, removed from setuptools 81+.
   Pinned in `requirements.txt`. Without it, `import webrtcvad` fails at load.
10. **res10 / MediaPipe face backends are dead on this stack.** OpenCV 5 removed the Caffe importer
    (`cv2.dnn.readNetFromCaffe` gone); the arm64-macOS mediapipe wheel ships only the Tasks API (no
    `mp.solutions`). The working path is **YuNet** (opencv_zoo ONNX, ~230 KB) — fetched by
    `make setup`, enabled via `configs/faces.yaml`. See D-8.
11. **Heavy Tier-2 stack is not installed in the detect-stage venv** (torch / faster-whisper / faiss /
    spaCy / tesseract) — deliberate; Video-1 only needs numpy/cv2/av/PyYAML/typer. `process` → `ingest`
    → `ask` need the full `make setup`.

## Video-1's own remaining call — RESOLVED (2026-07-16)

12. **cv2-vs-numpy determinism gap in `motion.py`.** ✅ Resolved by *loud fallback*, not a hard cv2
    requirement (requiring cv2 would break the canonical minimal-env funnel that STATUS.md certifies).
    `MotionDetector.last_backend` now records `cv2_phasecorr` | `numpy_phasecorr` | `raw_framediff`;
    the runner surfaces it as `notes["motion_backend"]` and `hindsight detect` prints a warning when
    the numpy fallback ran, so a numpy-computed run is never silently compared against a cv2 run —
    same contract as `voice_activity`'s VAD-backend logging. Within a fixed backend, motion stays
    byte-deterministic. (The same cv2/numpy split also lives in the shared `downscale()` resampler, so
    for report-grade numbers everyone should run with cv2 — which is a core dep — installed.)
