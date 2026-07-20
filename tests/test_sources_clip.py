"""ClipSource decode — downscale cap, target_fps subsampling, timestamps, no-audio path.

The `_downscale_long_edge` tests are pure numpy and always run. The decode tests need a real
container, so they generate a tiny H.264 clip with PyAV and skip cleanly if PyAV or the encoder
is unavailable (the canonical minimal env has neither).
"""

from __future__ import annotations

import numpy as np
import pytest

from hindsight.sources.clip import ClipSource, _downscale_long_edge


# --------------------------------------------------------------------------- pure: downscale cap
def test_downscale_noop_under_cap():
    rgb = np.zeros((90, 160, 3), np.uint8)  # long edge 160
    assert _downscale_long_edge(rgb, 1280).shape == (90, 160, 3)


def test_downscale_caps_long_edge_and_keeps_aspect():
    rgb = np.zeros((200, 400, 3), np.uint8)  # long edge 400, 2:1
    out = _downscale_long_edge(rgb, 100)
    h, w = out.shape[:2]
    assert max(h, w) <= 100
    assert abs((w / h) - 2.0) < 0.05  # aspect preserved


def test_downscale_none_cap_is_noop():
    rgb = np.zeros((200, 400, 3), np.uint8)
    assert _downscale_long_edge(rgb, None).shape == (200, 400, 3)


# --------------------------------------------------------------------------- decode (needs PyAV)
W, H, FPS, SECS = 320, 240, 10, 2.0


@pytest.fixture(scope="module")
def clip_path(tmp_path_factory):
    av = pytest.importorskip("av")
    path = tmp_path_factory.mktemp("clip") / "t.mp4"
    try:
        out = av.open(str(path), "w")
        st = out.add_stream("libx264", rate=FPS)
        st.width, st.height, st.pix_fmt = W, H, "yuv420p"
        st.options = {"crf": "28", "preset": "ultrafast"}
        for i in range(int(FPS * SECS)):
            arr = np.zeros((H, W, 3), np.uint8)
            x = (i * 13) % (W - 20)
            arr[:, x:x + 20] = 200        # a moving bar so frames differ
            arr[..., 1] = (i * 7) % 255    # and a changing tint
            vf = av.VideoFrame.from_ndarray(arr, format="rgb24").reformat(format="yuv420p")
            for p in st.encode(vf):
                out.mux(p)
        for p in st.encode():
            out.mux(p)
        out.close()
    except Exception as exc:  # noqa: BLE001 — no libx264 in this build
        pytest.skip(f"cannot encode a test clip: {exc}")
    return path


def test_meta_matches_encoded(clip_path):
    m = ClipSource(str(clip_path)).meta()
    assert (m.width, m.height) == (W, H)
    assert abs(m.fps - FPS) < 0.5
    assert abs(m.duration_s - SECS) < 0.5


def test_frames_are_rgb_uint8_and_timestamps_monotonic(clip_path):
    frames = list(ClipSource(str(clip_path)).frames())
    assert len(frames) > 0
    for f in frames:
        assert f.rgb.dtype == np.uint8 and f.rgb.ndim == 3 and f.rgb.shape[2] == 3
    ts = [f.t_ms for f in frames]
    assert ts == sorted(ts)  # non-decreasing on one monotonic clock


def test_target_fps_subsamples(clip_path):
    native = list(ClipSource(str(clip_path)).frames())
    sub = list(ClipSource(str(clip_path), target_fps=4).frames())
    assert len(sub) < len(native)
    assert 4 <= len(sub) <= 12  # ~4 fps over 2 s, allowing for boundary rounding


def test_max_long_edge_downscales(clip_path):
    frames = list(ClipSource(str(clip_path), max_long_edge_px=160).frames())
    assert frames and all(max(f.rgb.shape[:2]) <= 160 for f in frames)


def test_no_audio_stream_yields_no_chunks(clip_path):
    # the test clip is video-only -> audio() degrades to empty, not an error
    assert list(ClipSource(str(clip_path)).audio()) == []
