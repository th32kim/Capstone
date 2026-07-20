"""D1 scene_change + D2 motion — core behaviour, ego-motion mechanism, determinism.

Backend-agnostic: windows are built from numpy frames, so these run in the canonical minimal
env (no cv2/av) on the numpy phase-correlation path AND in the full env on cv2. They lock in the
behaviour Video-1 validated by hand (DESIGN_DELTAS D-2); the deep ego-comp *effectiveness* numbers
live there, measured on real footage. Frames are 90x160 so the detectors' 160x90 downscale is a
no-op and the injected pans/cuts are exact at the working resolution.
"""

from __future__ import annotations

import numpy as np

from hindsight.contracts import Frame, Window
from hindsight.detectors.video.motion import (
    MotionDetector, _numpy_phase_shift, _overlap_residual)
from hindsight.detectors.video.scene_change import SceneChangeDetector


def _win(rgb: np.ndarray, i: int) -> Window:
    return Window(index=i, t_start=i * 0.5, t_end=i * 0.5 + 0.5,
                  frames=(Frame(t_ms=i * 500, rgb=rgb),), chunks=())


def _pan_frames(dx_per_win: int, n: int = 10, seed: int = 5) -> list[np.ndarray]:
    """A pure CAMERA pan (ego motion) over a large static scene: each window crops a window
    shifted by dx_per_win px. Reveals genuinely new content at the leading edge (not a cyclic
    roll), so it exercises the overlap-residual path the way real footage does."""
    scene = np.random.default_rng(seed).integers(0, 255, size=(200, 400, 3), dtype=np.uint8)
    return [scene[40:130, 20 + i * dx_per_win: 20 + i * dx_per_win + 160].copy() for i in range(n)]


def _bg(seed: int = 7) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 255, size=(90, 160, 3), dtype=np.uint8)


def _run(det, frames):
    return [det.score(_win(f, i)) for i, f in enumerate(frames)]


# --------------------------------------------------------------------------- motion (D2)
def test_motion_first_window_masked():
    # no predecessor -> None (masked), never a fabricated 0.0
    assert MotionDetector().score(_win(_bg(), 0)) is None


def test_motion_static_scene_scores_zero():
    bg = _bg()
    scores = [s for s in _run(MotionDetector(), [bg.copy() for _ in range(14)]) if s is not None]
    assert max(scores) < 0.05  # identical frames -> no residual motion


def test_motion_fires_on_moving_object():
    bg = _bg()
    frames = []
    for i in range(14):
        f = bg.copy()
        x = (i * 9) % 140
        f[40:52, x:x + 14] = 250  # a bright block translating across a static background
        frames.append(f)
    scores = [s for s in _run(MotionDetector(), frames) if s is not None]
    assert np.mean(scores) > 0.15  # object motion is picked up (static baseline is ~0)


def test_motion_ego_estimates_global_pan():
    # D-2 core: with ego_compensate on, phase correlation recovers the dominant camera translation.
    md = MotionDetector(ego_compensate=True)
    bg = _bg()
    md.score(_win(bg, 0))
    md.score(_win(np.roll(bg, 6, axis=1), 1))  # pure 6 px horizontal pan
    assert abs(md.raw_global_motion - 6.0) <= 1.5


def test_motion_no_ego_leaves_global_motion_unmeasured():
    md = MotionDetector(ego_compensate=False)
    bg = _bg()
    md.score(_win(bg, 0))
    md.score(_win(np.roll(bg, 6, axis=1), 1))
    assert md.raw_global_motion == 0.0  # the ego branch is the only writer of raw_global_motion


def test_motion_deterministic():
    bg = _bg()
    frames = [np.roll(bg, i * 3, axis=1).copy() for i in range(12)]
    assert _run(MotionDetector(), frames) == _run(MotionDetector(), frames)


def test_motion_ego_suppresses_pure_camera_pan():
    # THE headline D-2 claim: a pure ego pan (no object motion) must read LOW residual WITH
    # ego-comp and HIGH without. Guards the sign + overlap fixes (a regression to either makes
    # ego-comp a no-op and this fails). Uses a real pan that reveals new scene at the edge.
    frames = _pan_frames(dx_per_win=18)
    on = [s for s in _run(MotionDetector(ego_compensate=True), frames) if s is not None]
    off = [s for s in _run(MotionDetector(ego_compensate=False), frames) if s is not None]
    assert np.mean(on) < 0.15                    # ego-comp cancels the pan -> near-zero residual
    assert np.mean(off) > 0.30                   # uncompensated, the same pan reads as motion
    assert np.mean(on) < 0.5 * np.mean(off)      # and the reduction is large, not marginal


def test_numpy_phase_shift_has_aligning_sign():
    # The numpy fallback must return the shift that MAPS prev ONTO gray (same convention as
    # cv2.phaseCorrelate) so the overlap residual cancels a pan. A negated sign (the old bug)
    # leaves the residual at the uncompensated baseline.
    rng = np.random.default_rng(11)
    prev = rng.integers(0, 255, (90, 160)).astype(np.float32)
    gray = np.roll(np.roll(prev, 4, 0), 7, 1).astype(np.float32)
    dy, dx = _numpy_phase_shift(prev, gray)
    assert (dy, dx) == (4, 7)                     # aligning sign, not (-4, -7)
    assert _overlap_residual(prev, gray, dy, dx) < 1.0


def test_motion_large_pan_no_border_artifact():
    # A shift larger than a quarter-frame (fast head turn) must not leave np.roll wrap/edge
    # artefacts in the residual. Overlap-slicing compares only in-bounds pixels, so a correctly
    # compensated large pan reads ~0 even though the leading edge shows fresh content.
    rng = np.random.default_rng(11)
    prev = rng.integers(0, 255, (90, 160)).astype(np.float32)
    gray = np.roll(np.roll(prev, 30, 0), 55, 1).copy()   # shift > 90//4 and > 160//4
    gray[:30, :] = rng.integers(0, 255, (30, 160))       # fresh revealed band, top
    gray[:, :55] = rng.integers(0, 255, (90, 55))        # fresh revealed band, left
    assert _overlap_residual(prev, gray, 30, 55) < 1.0


def test_motion_records_ego_backend():
    # the phase-correlation path is recorded so a numpy-fallback run is never silently mixed with a
    # cv2 run (they disagree ~88% of the time; DESIGN_DELTAS D-2). Which one is env-dependent.
    from hindsight._deps import optional
    md = MotionDetector(ego_compensate=True)
    bg = _bg()
    md.score(_win(bg, 0))
    md.score(_win(np.roll(bg, 4, axis=1), 1))
    assert md.last_backend == ("cv2_phasecorr" if optional("cv2") is not None else "numpy_phasecorr")


def test_motion_raw_framediff_backend():
    md = MotionDetector(ego_compensate=False)
    bg = _bg()
    md.score(_win(bg, 0))
    md.score(_win(bg.copy(), 1))
    assert md.last_backend == "raw_framediff"


# --------------------------------------------------------------------------- scene_change (D1)
def test_scene_first_window_masked():
    assert SceneChangeDetector().score(_win(_bg(), 0)) is None


def test_scene_same_palette_scores_low():
    bg = _bg()
    sc = SceneChangeDetector()
    sc.score(_win(bg, 0))
    assert sc.score(_win(bg.copy(), 1)) < 0.05  # identical histogram -> correl ~1 -> ~0


def test_scene_fires_on_hard_cut():
    # distinct COLOUR PALETTES (histogram method is palette-based): blue-ish -> red-ish
    blue = np.zeros((90, 160, 3), np.uint8)
    blue[..., 2], blue[..., 0] = 200, 40
    red = np.zeros((90, 160, 3), np.uint8)
    red[..., 0], red[..., 1] = 200, 40
    sc = SceneChangeDetector()
    sc.score(_win(blue, 0))
    same = sc.score(_win(blue.copy(), 1))
    cut = sc.score(_win(red, 2))
    assert cut > 0.3 and cut > same


def test_scene_deterministic():
    frames = [_bg(s) for s in (1, 2, 3, 4, 5)]
    assert _run(SceneChangeDetector(), frames) == _run(SceneChangeDetector(), frames)
