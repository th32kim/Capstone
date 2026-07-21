"""eval/text_backend (V2-2): label loading + honest NOT MEASURED / unavailable paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from hindsight.config import load_config
from hindsight.eval import text_backend as tb


def test_load_text_labels_missing_file_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(tb, "LABELS_DIR", tmp_path)
    assert tb.load_text_labels("no_such_clip") == []


def test_load_text_labels_parses_real_shipped_labels():
    # data/labels/plane_1_text.csv is checked in (V2-2 ground truth) -- exercise the real file
    labels = tb.load_text_labels("plane_1")
    assert len(labels) > 0
    assert any(lb.text_present for lb in labels)
    assert any(not lb.text_present for lb in labels)
    assert all(lb.t_end > lb.t_start for lb in labels)


def test_label_for_matches_midpoint():
    labels = [tb.TextLabel(0.0, 1.0, False), tb.TextLabel(1.0, 2.0, True)]
    assert tb._label_for(0.0, 1.0, labels) is labels[0]
    assert tb._label_for(1.0, 2.0, labels) is labels[1]
    assert tb._label_for(5.0, 6.0, labels) is None


def test_benchmark_not_measured_without_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(tb, "LABELS_DIR", tmp_path)
    cfg = load_config("default")
    result = tb.benchmark(source=None, cfg=cfg, clip="no_labels_for_this_clip")
    assert "NOT MEASURED" in result


def test_run_backend_honest_when_east_unprovisioned():
    # default.yaml points east_model_path at models/east/... -- if that file happens to be
    # absent (fresh checkout, no `make setup` run yet), the east branch must say so honestly
    # rather than silently reporting MSER's numbers under the "east" label.
    cfg = load_config("default")
    east_path = cfg.get("detectors.text_presence.east_model_path")
    if Path(east_path).exists():
        pytest.skip("EAST model is provisioned on this machine; fallback path not exercised")
    from hindsight.sources.synthetic import demo_source

    labels = [tb.TextLabel(0.0, 240.0, True)]
    result = tb.run_backend(demo_source(), cfg, "east", labels, threshold=0.5)
    assert result.available is False
    assert result.precision == "NOT MEASURED (EAST unavailable: see reason)"


def test_benchmark_end_to_end_on_synthetic_source(tmp_path, monkeypatch):
    monkeypatch.setattr(tb, "LABELS_DIR", tmp_path)
    (tmp_path / "synthclip_text.csv").write_text(
        "t_start,t_end,text_present\n0.0,120.0,0\n120.0,240.0,1\n", encoding="utf-8"
    )
    from hindsight.sources.synthetic import demo_source

    cfg = load_config("default")
    result = tb.benchmark(demo_source(), cfg, "synthclip", threshold=0.5)
    assert result["n_labels"] == 2
    assert "east" in result and "mser_swt" in result
