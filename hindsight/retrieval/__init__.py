"""Retrieval: dense ⊕ KG fused with RRF, constrained, top-k."""

from .constraints import Constraints, apply_constraints, parse_constraints
from .engine import RetrievalEngine
from .rrf import rrf_fuse

__all__ = ["RetrievalEngine", "rrf_fuse", "parse_constraints", "apply_constraints", "Constraints"]
