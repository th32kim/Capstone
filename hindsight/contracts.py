"""
hindsight.contracts — THE FROZEN DATA CONTRACT.

This module is the interface. Everything else in the repo imports from here and nothing
here imports from anywhere else in the repo. If you find yourself wanting to change a
field in this file, stop: that is a cross-subsystem design change, and it goes through
docs/DESIGN_DELTAS.md and the owning team member first.

Two contracts in here are load-bearing across people:

  * EMBEDDING_DIM = 384 is the sole frozen interface between Tier-2 (P3) and the
    store (P4). It is asserted, not adapted. A record with the wrong dimension is a
    hard error, never a warning.

  * Frame.t_ms and AudioChunk.t_ms share ONE monotonic clock. Video and audio are
    independently paced but commensurably timestamped. Everything downstream --
    windowing, gating, segment bounds, the store's start_ts/end_ts -- assumes this.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, Sequence, runtime_checkable

import numpy as np

# ---------------------------------------------------------------------------
# Frozen constants. Do not "make these configurable."
# ---------------------------------------------------------------------------

EMBEDDING_DIM: int = 384          # MiniLM-L6-v2. The P3<->P4 interface. FROZEN.
AUDIO_SAMPLE_RATE: int = 16_000   # FS2 requires >= 16 kHz.
AUDIO_CHUNK_MS: int = 20
DECISION_WINDOW_S: float = 0.5    # Tier-1 decision window.

# Structural floor on a retained segment, derived (never typed twice):
#   min possible ACTIVE span (3 windows to leave ACTIVE) = 1.5 s
#   + pre_roll (2.0) + post_roll (3.0)                    = 6.5 s
MIN_ACTIVE_SPAN_S: float = 3 * DECISION_WINDOW_S          # 1.5
STRUCTURAL_MIN_RETAINED_S: float = MIN_ACTIVE_SPAN_S + 2.0 + 3.0   # 6.5

DETECTOR_NAMES: tuple[str, ...] = (
    "scene_change",
    "motion",
    "face_presence",
    "text_presence",
    "voice_activity",
    "audio_energy",
    "dwell",          # optional (D-3); availability-masked off by default
)

Modality = Literal["video", "audio", "imu"]
EntityType = Literal["PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "DATE", "TIME", "OTHER"]


# ---------------------------------------------------------------------------
# Capture layer -- what a Source yields
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Frame:
    """One video frame on the master monotonic clock."""
    t_ms: int
    rgb: np.ndarray               # (H, W, 3) uint8

    def __post_init__(self) -> None:
        if self.rgb.ndim != 3 or self.rgb.shape[2] != 3:
            raise ValueError(f"Frame.rgb must be HxWx3, got {self.rgb.shape}")


@dataclass(frozen=True, slots=True)
class AudioChunk:
    """One audio chunk on the master monotonic clock. PCM16 mono @ 16 kHz."""
    t_ms: int
    pcm: np.ndarray               # (N,) int16

    def __post_init__(self) -> None:
        if self.pcm.dtype != np.int16 or self.pcm.ndim != 1:
            raise ValueError(f"AudioChunk.pcm must be 1-D int16, got {self.pcm.dtype}/{self.pcm.ndim}D")


@runtime_checkable
class Source(Protocol):
    """A swappable capture source. The glasses are ONE implementation of this, not the API."""
    name: str
    fps: float
    sample_rate: int

    def frames(self) -> "Sequence[Frame]": ...
    def audio(self) -> "Sequence[AudioChunk]": ...
    def duration_s(self) -> float: ...


# ---------------------------------------------------------------------------
# Tier-1 -- detectors, fusion, gating
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Window:
    """One 0.5 s decision window. The unit Tier-1 reasons about."""
    index: int
    t_start: float
    t_end: float
    frames: tuple[Frame, ...]
    chunks: tuple[AudioChunk, ...]


@dataclass(frozen=True, slots=True)
class DetectorScores:
    """Six (or seven) scores + an availability mask, for one window.

    A detector that could not run -- no audio track, cadence not due and no held value,
    backend unavailable -- is MASKED OUT (mask bit False). It is NOT scored 0.0.
    Scoring an unavailable detector 0.0 silently drags the fused salience down and is
    the single easiest way to destroy recall without noticing.
    """
    window_index: int
    t_start: float
    t_end: float
    scores: dict[str, float]          # name -> [0,1]
    mask: dict[str, bool]             # name -> available?
    stale: dict[str, bool] = field(default_factory=dict)   # value held from an earlier cadence tick

    def __post_init__(self) -> None:
        for k, v in self.scores.items():
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"detector {k!r} emitted {v}, outside [0,1]")


@runtime_checkable
class Detector(Protocol):
    name: str
    modality: Modality
    cadence: int                       # 1 = every frame/window; 24 = every 24th frame

    def score(self, w: Window) -> float | None:
        """Return a salience score in [0,1], or None if unavailable (-> mask bit False)."""
        ...


GateState = Literal["IDLE", "ARMING", "ACTIVE", "CLOSING"]


@dataclass(frozen=True, slots=True)
class Segment:
    """A candidate interest segment emitted by the gate. Not yet validated, not yet processed."""
    segment_id: str
    session_id: str
    source: str
    t_start: float                     # retained bounds, i.e. INCLUDING pre/post-roll
    t_end: float
    active_start: float                # the ACTIVE span, EXCLUDING pre/post-roll
    active_end: float
    salience_peak: float
    salience_mean: float
    keyframe_ts: tuple[float, ...]     # <= 3, chosen by cascade.evidence
    detector_evidence: dict[str, float] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start

    @property
    def active_duration(self) -> float:
        return self.active_end - self.active_start

    def __post_init__(self) -> None:
        if self.duration < STRUCTURAL_MIN_RETAINED_S - 1e-6:
            raise ValueError(
                f"segment {self.segment_id} is {self.duration:.2f}s, below the {STRUCTURAL_MIN_RETAINED_S}s "
                "structural floor (min ACTIVE 1.5s + pre-roll 2.0s + post-roll 3.0s). "
                "Either the gate is wrong or the floor is."
            )


# ---------------------------------------------------------------------------
# Cascade -- the second guard
# ---------------------------------------------------------------------------

CascadeRoute = Literal["auto_accept", "validate", "shadow"]


@dataclass(frozen=True, slots=True)
class Evidence:
    """The BOUNDED evidence pack handed to a validator. Bounded evidence == bounded cost."""
    segment_id: str
    keyframes_jpeg: tuple[bytes, ...]           # <= 3, <= 512px long edge
    transcript: str | None                      # cheap ASR (tiny.en), uncertain band only
    detector_evidence: dict[str, float]
    duration_s: float
    salience_peak: float
    salience_mean: float


@dataclass(frozen=True, slots=True)
class Verdict:
    """A validator's decision on one candidate segment.

    A validator may only SHRINK a segment (via `trim`) or DROP it. It can never create one
    and it can never extend one. This asymmetry is the whole reason recall stays Tier-1's
    responsibility -- see docs/DESIGN_DELTAS.md D-1.
    """
    segment_id: str
    keep: bool
    confidence: float                            # [0,1]
    reason: str                                  # human-readable, <= 200 chars
    tags: tuple[str, ...]
    validator: str                               # "null" | "clip_local" | "llm_claude"
    route: CascadeRoute
    trim: tuple[float, float] | None = None      # (new_start, new_end); may only shrink
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    failed_open: bool = False                    # True if we kept it because the validator errored

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence {self.confidence} outside [0,1]")
        if len(self.reason) > 200:
            raise ValueError("Verdict.reason must be <= 200 chars")


@runtime_checkable
class Validator(Protocol):
    name: str

    def validate(self, seg: Segment, ev: Evidence) -> Verdict:
        """MUST NOT raise. On any internal failure, return keep=True with failed_open=True."""
        ...


# ---------------------------------------------------------------------------
# Tier-2 -- the memory record
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Entity:
    text: str
    type: EntityType
    span: tuple[int, int]                        # char offsets into the aggregated text


@dataclass(slots=True)
class MemoryRecord:
    """Tier-2's output and the store's input. The shared P3<->P4 shape."""
    segment_id: str
    t_start: float
    t_end: float
    source: str
    on_device: bool

    # Phase A -- parallel, each a pure function of the raw segment
    transcript: str = ""
    caption: str = ""
    ocr_text: str = ""

    # Phase B -- sequential
    entities: list[Entity] = field(default_factory=list)
    summary: str = ""
    embedding: list[float] = field(default_factory=list)     # len == EMBEDDING_DIM. FROZEN.

    # provenance -- carried so the store can explain WHY a memory exists
    salience_peak: float = 0.0
    validator_verdict: str = ""                  # "auto_accept" | "ai_keep" | "null"
    validator_confidence: float = 0.0
    validator_reason: str = ""

    def validate(self) -> None:
        """Call before handing to the store. Loud failure, never silent coercion."""
        if len(self.embedding) != EMBEDDING_DIM:
            raise ValueError(
                f"embedding has dim {len(self.embedding)}, expected {EMBEDDING_DIM}. "
                "d=384 is the FROZEN P3<->P4 interface -- fix the producer, do not resize here."
            )
        if self.t_end <= self.t_start:
            raise ValueError(f"segment {self.segment_id}: t_end <= t_start")


# ---------------------------------------------------------------------------
# Retrieval -- what comes back out
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class MemoryHit:
    """One result. This is what a MemoryCard renders."""
    segment_id: str
    score: float                                  # fused RRF score
    dense_rank: int | None                        # None = this channel did not surface it
    kg_rank: int | None
    t_start: float
    t_end: float
    summary: str
    entities: tuple[str, ...]
    thumbnail_ref: str | None


@dataclass(frozen=True, slots=True)
class QueryResult:
    query: str
    hits: tuple[MemoryHit, ...]
    latency_ms: dict[str, float]                  # per-stage. this IS Table 3.4-7. measure it.
