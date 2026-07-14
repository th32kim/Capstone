"""Shared fixtures."""

from __future__ import annotations

import pytest

from hindsight.config import load_config


@pytest.fixture
def cfg():
    return load_config("default")


@pytest.fixture
def embedder(cfg):
    from hindsight.tier2.embed import Embedder

    return Embedder(cfg)


def make_record(seg_id, text, embedder, *, t_start=0.0, t_end=10.0, entities=None):
    from hindsight.contracts import MemoryRecord

    r = MemoryRecord(seg_id, t_start, t_end, "test", on_device=True)
    r.transcript = text
    r.summary = text
    r.entities = entities or []
    r.embedding = embedder.embed(text)
    return r
