"""Tier-2 (summary/embed) + eval gating (P/R/F1, the assisted-recall refusal, kappa)."""

from __future__ import annotations

import numpy as np
import pytest

from hindsight.eval.gating import AssistedLabelsError, Interval, prf, recall_labels
from hindsight.tier2.embed import Embedder
from hindsight.tier2.summarize import extractive_summary


def test_extractive_summary_selects_sentences(cfg):
    text = ("The data contract is frozen. Mike explained the interface. "
            "The weather was mild. We agreed on the embedding dimension. Lunch was late.")
    summ = extractive_summary(text, k=2)
    # returns exactly some of the source sentences, never invents text
    for sent in summ.split(". "):
        assert sent.strip(". ") in text


def test_embed_dim_and_norm(cfg):
    e = Embedder(cfg)
    v = np.array(e.embed("hello world"))
    assert v.shape == (384,)
    assert abs(np.linalg.norm(v) - 1.0) < 1e-5  # L2-normalised


def test_asr_builds_from_old_schema_config():
    # a config written before tier2.asr.device existed must still build (Config.get raises
    # on missing keys, so every read in Asr must pass an explicit default)
    from hindsight.config import _freeze
    from hindsight.tier2.asr import Asr

    cfg = _freeze({"tier2": {"asr": {"model": "base.en", "compute_type": "int8"}}}, source="test")
    a = Asr(cfg)  # must not raise KeyError
    assert a.device == "cpu"


def test_prf_basic():
    labels = [Interval(10, 20, "dense"), Interval(50, 60, "dense")]
    pred = [(10, 20), (100, 110)]  # one hit, one false positive; one label missed
    m = prf(pred, labels, iou_thr=0.5, allow_recall=True)
    assert m.tp == 1 and m.fp == 1 and m.fn == 1
    assert abs(m.precision - 0.5) < 1e-9
    assert abs(m.recall - 0.5) < 1e-9


def test_recall_refuses_assisted_labels():
    assisted = [Interval(10, 20, "assisted"), Interval(30, 40, "assisted")]
    with pytest.raises(AssistedLabelsError):
        recall_labels(assisted)
    # but dense present -> returns the dense subset
    mixed = assisted + [Interval(50, 60, "dense")]
    assert len(recall_labels(mixed)) == 1
