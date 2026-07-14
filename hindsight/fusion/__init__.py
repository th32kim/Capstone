"""Tier-1 fusion: weighted linear sum with the availability mask applied."""

from .linear import FusedWindow, fuse_all, fuse_window

__all__ = ["FusedWindow", "fuse_window", "fuse_all"]
