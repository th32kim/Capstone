"""Config (extends/validate/freeze/hash) + clock (alignment/drift) + source determinism."""

from __future__ import annotations

import numpy as np
import pytest

from hindsight import contracts
from hindsight.clock import align_windows, av_drift_ms, window_index
from hindsight.config import load_config
from hindsight.contracts import AudioChunk, Frame
from hindsight.sources import demo_source


def test_contracts_frozen():
    assert contracts.EMBEDDING_DIM == 384
    assert abs(contracts.STRUCTURAL_MIN_RETAINED_S - 6.5) < 1e-9


def test_extends_and_hash():
    default = load_config("default")
    on_dev = load_config("onDevice")
    cloud = load_config("cloud")
    assert on_dev.get("cascade.validator") == "clip_local"
    assert on_dev.get("clock.decision_window_s") == 0.5  # inherited from default
    # onDevice only re-states values that are already default -> resolved config is identical
    assert on_dev.config_hash == default.config_hash
    # cloud genuinely differs (on_device_only + validator) -> different hash
    assert cloud.config_hash != default.config_hash
    # deterministic re-load
    assert load_config("default").config_hash == default.config_hash


def test_cloud_profile_allows_llm():
    cloud = load_config("cloud")
    assert cloud.get("on_device_only") is False
    assert cloud.get("cascade.validator") == "llm_claude"


def test_on_device_forbids_cloud_validator():
    default = load_config("default")
    with pytest.raises(ValueError):
        default.with_overrides(**{"cascade.validator": "llm_claude"})


def test_overrides_change_hash():
    c = load_config("default")
    c2 = c.with_overrides(**{"gating.theta_on": 0.4})
    assert c2.get("gating.theta_on") == 0.4
    assert c2.config_hash != c.config_hash
    assert c.get("gating.theta_on") == 0.58  # original untouched


def test_window_alignment():
    frames = [Frame(t_ms=t, rgb=np.zeros((4, 4, 3), np.uint8)) for t in (0, 300, 600, 900)]
    chunks = [AudioChunk(t_ms=t, pcm=np.zeros(320, np.int16)) for t in (0, 500, 1000)]
    ws = align_windows(frames, chunks, window_s=0.5)
    assert window_index(600, 0.5) == 1
    assert len(ws) == 3  # windows 0,1,2
    assert len(ws[0].frames) == 2 and ws[0].t_start == 0.0  # 0ms,300ms
    assert len(ws[1].frames) == 2  # 600ms,900ms


def test_av_drift_measured():
    frames = [Frame(t_ms=1000, rgb=np.zeros((4, 4, 3), np.uint8))]
    chunks = [AudioChunk(t_ms=1080, pcm=np.zeros(320, np.int16))]
    assert av_drift_ms(frames, chunks) == 80


def test_synthetic_determinism():
    a = list(demo_source().frames())
    b = list(demo_source().frames())
    assert len(a) == len(b)
    assert np.array_equal(a[100].rgb, b[100].rgb)
