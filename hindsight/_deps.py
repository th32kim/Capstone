"""Lazy optional-dependency loader.

The heavy stack (OpenCV, torch, faiss, av, spaCy, whisper, webrtcvad, tesseract) is
not required to import the package or run the CLI. Any module that needs one of these
imports it *through here*, at call time, so:

  * `import hindsight` works with only numpy + PyYAML installed;
  * a stage whose backend is missing degrades honestly (availability mask off, or a
    printed `NOT MEASURED` / `backend unavailable`) instead of crashing at import or,
    worse, silently fabricating a result.

This directly serves CLAUDE.md §1.1 (no fabricated numbers) and §3 (a detector that
cannot run is masked out, not scored 0).
"""

from __future__ import annotations

import importlib
from functools import lru_cache


class MissingDependency(RuntimeError):
    """Raised (or caught) when an optional backend is not installed."""


@lru_cache(maxsize=None)
def available(module: str) -> bool:
    """True if `module` can be imported. Cached; safe to call in hot loops."""
    try:
        importlib.import_module(module)
        return True
    except Exception:  # noqa: BLE001 — a broken install is "unavailable" too
        return False


def require(module: str, *, feature: str) -> object:
    """Import and return `module`, or raise MissingDependency with a useful message."""
    try:
        return importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001
        raise MissingDependency(
            f"{feature} needs '{module}', which is not installed. "
            f"Install the full stack with `pip install -e .[full]` (see requirements.txt)."
        ) from exc


def optional(module: str):
    """Import and return `module`, or None if unavailable. Never raises."""
    try:
        return importlib.import_module(module)
    except Exception:  # noqa: BLE001
        return None
