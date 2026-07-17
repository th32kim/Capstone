"""Detector bank: build from config, run over aligned windows, emit DetectorScores.

Cadence + zero-order-hold policy (base.CadencePolicy) lives in the runner, not the
detectors, so detectors stay small pure `score(window)` functions (CLAUDE.md §9). The
runner also:
  * measures per-detector compute cost (the M2 budget line is *measured*, not assumed),
  * records the VAD backend actually used and any availability-masked detector into `notes`
    (a silent degrade is the worst outcome — M2),
  * wires D7 dwell's angular-velocity source to D2's ego-motion estimate when both are on.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..clock import align_windows
from ..config import Config
from ..contracts import DECISION_WINDOW_S, DetectorScores, DETECTOR_NAMES
from .audio.audio_energy import AudioEnergyDetector
from .audio.voice_activity import VoiceActivityDetector
from .base import CadencePolicy
from .imu.dwell import DwellDetector
from .video.face_presence import FacePresenceDetector
from .video.motion import MotionDetector
from .video.scene_change import SceneChangeDetector
from .video.text_presence import TextPresenceDetector


@dataclass
class DetectResult:
    windows: list[DetectorScores]
    detector_names: tuple[str, ...]
    fps: float
    duration_s: float
    timings_s: dict[str, float] = field(default_factory=dict)   # total score() seconds per detector
    n_evals: dict[str, int] = field(default_factory=dict)       # windows actually evaluated
    notes: dict[str, object] = field(default_factory=dict)


def build_detectors(cfg: Config) -> list[object]:
    """Instantiate the enabled detectors, in the frozen DETECTOR_NAMES order."""
    d = cfg.get("detectors")
    built: dict[str, object] = {}
    if d.get("scene_change", {}).get("enabled"):
        c = d["scene_change"]
        built["scene_change"] = SceneChangeDetector(
            cadence=c["cadence"], downscale_wh=tuple(c["downscale"]), hist_bins=tuple(c["hist_bins"]))
    if d.get("motion", {}).get("enabled"):
        c = d["motion"]
        built["motion"] = MotionDetector(
            cadence=c["cadence"], downscale_wh=tuple(c["downscale"]),
            ego_compensate=c.get("ego_compensate", True), norm_percentiles=tuple(c["norm_percentiles"]))
    if d.get("face_presence", {}).get("enabled"):
        c = d["face_presence"]
        built["face_presence"] = FacePresenceDetector(
            cadence=c["cadence"], backend=c.get("backend", "opencv_dnn"),
            min_confidence=c.get("min_confidence", 0.6), area_weight=c.get("area_weight", 0.3))
    if d.get("text_presence", {}).get("enabled"):
        c = d["text_presence"]
        built["text_presence"] = TextPresenceDetector(
            cadence=c["cadence"], backend=c.get("backend", "east"),
            min_confidence=c.get("min_confidence", 0.5))
    if d.get("voice_activity", {}).get("enabled"):
        c = d["voice_activity"]
        built["voice_activity"] = VoiceActivityDetector(
            cadence=c["cadence"],
            aggressiveness=c.get("aggressiveness", 2),
            frame_ms=c.get("frame_ms", 30),
            fallback=c.get("fallback", "energy"),
            fallback_norm_percentiles=tuple(c["fallback_norm_percentiles"]),
            fallback_threshold=c["fallback_threshold"],
            fallback_min_dbfs=c["fallback_min_dbfs"],
            silence_floor_dbfs=c["silence_floor_dbfs"],
        )
    if d.get("audio_energy", {}).get("enabled"):
        c = d["audio_energy"]
        built["audio_energy"] = AudioEnergyDetector(
            cadence=c["cadence"],
            norm_percentiles=tuple(c["norm_percentiles"]),
            silence_floor_dbfs=c["silence_floor_dbfs"],
        )
    if d.get("dwell", {}).get("enabled"):
        c = d["dwell"]
        built["dwell"] = DwellDetector(
            cadence=c["cadence"], settle_threshold_dps=c.get("settle_threshold_dps", 15.0),
            settle_hold_s=c.get("settle_hold_s", 1.0),
            window_s=cfg.get("clock.decision_window_s", DECISION_WINDOW_S))
    # frozen order
    return [built[name] for name in DETECTOR_NAMES if name in built]


def run_detectors(source, cfg: Config) -> DetectResult:
    """Decode the source, align to windows, score every detector on each window."""
    window_s = cfg.get("clock.decision_window_s", DECISION_WINDOW_S)
    frames = list(source.frames())
    chunks = list(source.audio())
    windows = align_windows(frames, chunks, window_s=window_s, duration_s=source.duration_s())

    detectors = build_detectors(cfg)
    names = tuple(getattr(det, "name") for det in detectors)
    native_fps = float(getattr(source, "fps", 0.0) or 0.0)
    # effective fps = frames actually decoded / duration (respects target_fps subsampling); this is
    # what cadence and the compute-budget line must use, since it is what we actually processed.
    duration = source.duration_s() or 0.0
    fps = (len(frames) / duration) if duration > 0 else native_fps

    # cadence policies (reduced-rate detectors hold + go stale between evals)
    policies = {det.name: CadencePolicy(det.cadence, fps or 24.0, window_s) for det in detectors}

    # wire D7 dwell to D2 ego-motion, if both present
    by_name = {det.name: det for det in detectors}
    if "dwell" in by_name and "motion" in by_name:
        motion = by_name["motion"]
        by_name["dwell"].wire_velocity(lambda m=motion: m.raw_global_motion)

    timings = {n: 0.0 for n in names}
    n_evals = {n: 0 for n in names}
    masked_off = {n: True for n in names}  # flip False once a detector ever produces a value
    out: list[DetectorScores] = []

    for w in windows:
        scores: dict[str, float] = {}
        mask: dict[str, bool] = {}
        stale: dict[str, bool] = {}
        for det in detectors:
            pol = policies[det.name]
            fresh = None
            if pol.due(w.index):
                t0 = time.perf_counter()
                fresh = det.score(w)
                timings[det.name] += time.perf_counter() - t0
                n_evals[det.name] += 1
            value, is_stale = pol.apply(w.index, fresh)
            if value is None:
                mask[det.name] = False
            else:
                scores[det.name] = float(value)
                mask[det.name] = True
                stale[det.name] = is_stale
                masked_off[det.name] = False
        out.append(DetectorScores(window_index=w.index, t_start=w.t_start, t_end=w.t_end,
                                   scores=scores, mask=mask, stale=stale))

    notes: dict[str, object] = {
        "vad_backend": getattr(by_name.get("voice_activity"), "last_backend", "n/a"),
        "vad_fallback_reason": getattr(by_name.get("voice_activity"), "fallback_reason", None),
        "masked_off_all_windows": sorted(n for n, off in masked_off.items() if off),
        "n_frames": len(frames),
        "n_chunks": len(chunks),
        "native_fps": native_fps,
        "effective_fps": round(fps, 2),
    }
    return DetectResult(windows=out, detector_names=names, fps=fps,
                        duration_s=source.duration_s(), timings_s=timings, n_evals=n_evals, notes=notes)
