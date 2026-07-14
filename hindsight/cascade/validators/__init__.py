"""Pluggable validators, selected by config (cascade.validator)."""

from __future__ import annotations

from .clip_local import ClipLocalValidator
from .llm_claude import LlmClaudeValidator
from .null import NullValidator


def build_validator(cfg):
    name = cfg.get("cascade.validator", "clip_local")
    if name == "null":
        return NullValidator()
    if name == "clip_local":
        return ClipLocalValidator(cfg)
    if name == "llm_claude":
        if cfg.get("on_device_only", True):
            raise ValueError("llm_claude requires on_device_only: false (configs/cloud.yaml)")
        return LlmClaudeValidator(cfg)
    raise ValueError(f"unknown validator {name!r}")


__all__ = ["NullValidator", "ClipLocalValidator", "LlmClaudeValidator", "build_validator"]
