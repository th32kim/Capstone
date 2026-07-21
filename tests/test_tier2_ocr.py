"""Tier-2 OCR (Tesseract, CLAUDE.md §5/V2-1): honest degrade + real text extraction."""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from hindsight.config import _freeze
from hindsight.tier2.ocr import Ocr

# The Windows dev machine installs Tesseract outside PATH; point at the winget install
# location if PATH lookup fails, so these tests exercise the real binary when present.
_WIN_TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def _tesseract_cmd() -> str | None:
    import os

    if shutil.which("tesseract"):
        return None  # PATH already resolves it; no override needed
    if os.path.exists(_WIN_TESSERACT):
        return _WIN_TESSERACT
    return "unavailable"  # sentinel: no real binary on this machine


def _text_image(text: str) -> np.ndarray:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (400, 100), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((10, 30), text, fill=(0, 0, 0))
    return np.array(img)


def test_sample_keyframes_spread_across_segment():
    # a 200-frame segment asking for 3 keyframes must not silently collapse to the first
    # three (V2-1 finding: on-screen text is often only in view partway through a long
    # segment, so sampling only the segment's start structurally misses it)
    cfg = _freeze({"tier2": {"ocr": {"max_keyframes": 3}}}, source="test")
    ocr = Ocr(cfg)
    frames = list(range(200))  # stand-ins; _sample only indexes, never inspects content
    picked = ocr._sample(frames)
    assert picked == [0, 99, 199] or picked == [0, 100, 199]  # linspace rounding
    assert picked[0] == 0 and picked[-1] == 199


def test_sample_keyframes_shorter_than_max_returns_all():
    cfg = _freeze({"tier2": {"ocr": {"max_keyframes": 5}}}, source="test")
    ocr = Ocr(cfg)
    frames = [1, 2, 3]
    assert ocr._sample(frames) == frames


def test_ocr_honest_empty_when_pytesseract_not_installed(monkeypatch):
    import hindsight.tier2.ocr as ocr_mod

    monkeypatch.setattr(ocr_mod, "optional", lambda module: None)
    cfg = _freeze({"tier2": {"ocr": {"enabled": True}}}, source="test")
    ocr = Ocr(cfg)
    assert ocr.backend == "unavailable"
    frame = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    assert ocr.read([frame]) == ""


def test_ocr_honest_empty_when_binary_missing():
    # pytesseract package present, but the binary path is bogus -> caught at call time
    cfg = _freeze(
        {"tier2": {"ocr": {"enabled": True, "tesseract_cmd": "C:/nonexistent/tesseract.exe"}}},
        source="test",
    )
    ocr = Ocr(cfg)
    if ocr.backend == "unavailable":
        pytest.skip("pytesseract package not installed in this environment")
    frame = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    assert ocr.read([frame]) == ""


def test_ocr_reads_real_text():
    cmd = _tesseract_cmd()
    if cmd == "unavailable":
        pytest.skip("tesseract binary not installed on this machine")
    cfg_dict = {"tier2": {"ocr": {"enabled": True, "max_keyframes": 1}}}
    if cmd:
        cfg_dict["tier2"]["ocr"]["tesseract_cmd"] = cmd
    cfg = _freeze(cfg_dict, source="test")
    ocr = Ocr(cfg)
    if ocr.backend == "unavailable":
        pytest.skip("pytesseract package not installed in this environment")
    text = ocr.read([_text_image("HINDSIGHT MEMORY TEST")])
    assert "HINDSIGHT" in text.upper()
