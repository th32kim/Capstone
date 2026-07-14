"""Config loading, validation, freezing, and hashing.

CLAUDE.md §1.2: *no hard-coded thresholds*. Every threshold, weight, cadence, and window
size lives in `configs/*.yaml`; code reads config. This module is the only place that
reads YAML. It:

  * resolves the `extends:` chain (onDevice/cloud extend default) with a deep merge,
  * validates the derived invariants that must never be typed twice (the 6.5 s structural
    floor; fusion weights are a subset of the detector names; cloud validator implies
    `on_device_only: false`),
  * freezes the result (dotted read-only access; mutation is not offered),
  * computes a deterministic `config_hash` — the cache key ingredient (CLAUDE.md §4) and
    the value stored in the store's `meta` table.

`with_overrides()` returns a *new* frozen config; it is how eval sweeps vary θ_on / τ_hi
without mutating anything (CLAUDE.md §1.7 determinism).
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from . import contracts

# configs/ lives at the repo root, a sibling of the hindsight/ package.
CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"

_MISSING = object()


def _deep_merge(base: dict, override: dict) -> dict:
    """Return base updated by override; dict values merge recursively, others replace."""
    out = copy.deepcopy(base)
    for key, val in override.items():
        if key == "extends":
            continue
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def _resolve_path(name_or_path: str | Path) -> Path:
    p = Path(name_or_path)
    if p.exists():
        return p
    # bare name like "default" or "onDevice" -> configs/<name>.yaml
    cand = CONFIG_DIR / (p.name if p.suffix else f"{p.name}.yaml")
    if not cand.suffix:
        cand = cand.with_suffix(".yaml")
    if cand.exists():
        return cand
    raise FileNotFoundError(f"config not found: {name_or_path!r} (looked in {CONFIG_DIR})")


def _load_raw(name_or_path: str | Path, _seen: set[Path] | None = None) -> dict:
    path = _resolve_path(name_or_path)
    seen = _seen or set()
    if path in seen:
        raise ValueError(f"circular `extends` chain at {path}")
    seen.add(path)
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config {path} did not parse to a mapping")
    parent_name = data.get("extends")
    if parent_name:
        parent = _load_raw(parent_name, seen)
        data = _deep_merge(parent, data)
    return data


@dataclass(frozen=True)
class Config:
    """A frozen, validated configuration.

    Read with `cfg.get("gating.theta_on")` (dotted) or `cfg["gating"]` (top-level section
    as a plain dict copy). There is no setter; use `with_overrides()`.
    """

    _data: dict
    source: str
    config_hash: str

    # -- access ------------------------------------------------------------
    def get(self, dotted: str, default: Any = _MISSING) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            elif default is not _MISSING:
                return default
            else:
                raise KeyError(f"config key not found: {dotted!r}")
        return copy.deepcopy(node)

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def as_dict(self) -> dict:
        return copy.deepcopy(self._data)

    # -- derivation --------------------------------------------------------
    def with_overrides(self, **dotted: Any) -> "Config":
        """New Config with dotted keys replaced. e.g. with_overrides(**{'gating.theta_on': 0.4})."""
        data = copy.deepcopy(self._data)
        for dotted_key, value in dotted.items():
            node = data
            parts = dotted_key.split(".")
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value
        return _freeze(data, source=f"{self.source}+overrides")


def _hash_config(data: dict) -> str:
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate(data: dict) -> None:
    """Assert the invariants that must hold regardless of which profile was loaded."""
    # Clock must match the frozen contract (these are duplicated in config for readability;
    # if they drift, that is a bug — fail loudly, never silently prefer one).
    win = data.get("clock", {}).get("decision_window_s")
    if win is not None and abs(win - contracts.DECISION_WINDOW_S) > 1e-9:
        raise ValueError(
            f"clock.decision_window_s={win} disagrees with the frozen "
            f"contracts.DECISION_WINDOW_S={contracts.DECISION_WINDOW_S}"
        )
    sr = data.get("clock", {}).get("audio_sample_rate")
    if sr is not None and sr != contracts.AUDIO_SAMPLE_RATE:
        raise ValueError(f"clock.audio_sample_rate={sr} != frozen {contracts.AUDIO_SAMPLE_RATE}")

    # Fusion weights are a subset of the known detector names.
    weights = data.get("fusion", {}).get("weights", {})
    unknown = set(weights) - set(contracts.DETECTOR_NAMES)
    if unknown:
        raise ValueError(f"fusion.weights names not in DETECTOR_NAMES: {sorted(unknown)}")

    # Embedding dim is frozen.
    dim = data.get("tier2", {}).get("embed", {}).get("dim")
    if dim is not None and dim != contracts.EMBEDDING_DIM:
        raise ValueError(
            f"tier2.embed.dim={dim} != frozen EMBEDDING_DIM={contracts.EMBEDDING_DIM}. "
            "d=384 is the P3<->P4 interface; fix the producer, do not resize."
        )

    # On-device profile must not select the cloud validator (CLAUDE.md §1.5, NFS4).
    on_device = data.get("on_device_only", True)
    validator = data.get("cascade", {}).get("validator", "clip_local")
    if on_device and validator == "llm_claude":
        raise ValueError(
            "on_device_only=true forbids the cloud validator 'llm_claude'. "
            "Use configs/cloud.yaml (on_device_only: false) for the opt-in cloud path."
        )

    # Gating hysteresis gap must be non-negative (θ_off <= θ_on).
    g = data.get("gating", {})
    if "theta_on" in g and "theta_off" in g and g["theta_off"] > g["theta_on"] + 1e-9:
        raise ValueError(f"gating.theta_off ({g['theta_off']}) > theta_on ({g['theta_on']})")


def _freeze(data: dict, *, source: str) -> Config:
    _validate(data)
    return Config(_data=copy.deepcopy(data), source=source, config_hash=_hash_config(data))


def load_config(name_or_path: str | Path = "default") -> Config:
    """Load, resolve `extends`, validate, and freeze a config. This is the public entry point."""
    raw = _load_raw(name_or_path)
    return _freeze(raw, source=str(name_or_path))
