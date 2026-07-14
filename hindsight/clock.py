"""One monotonic timebase; window alignment.

Frame.t_ms and AudioChunk.t_ms share ONE clock (contracts.py). This module is where
the two independently-paced streams are cut into aligned 0.5 s decision windows — the
unit Tier-1 reasons about. Window k covers `[k*W, (k+1)*W)` seconds and collects every
frame and chunk whose timestamp falls in that half-open interval.

Nothing here peeks at the future: a window is emitted only from samples at or before its
end. That is what lets the online claim (FS4, "while capture is ongoing") mean something.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from .contracts import DECISION_WINDOW_S, AudioChunk, Frame, Window


def window_index(t_ms: int, window_s: float = DECISION_WINDOW_S) -> int:
    """The 0-based decision-window index a timestamp falls into."""
    return int((t_ms / 1000.0) // window_s)


def align_windows(
    frames: Iterable[Frame],
    chunks: Iterable[AudioChunk],
    *,
    window_s: float = DECISION_WINDOW_S,
    duration_s: float | None = None,
) -> list[Window]:
    """Cut two time-aligned streams into contiguous decision windows.

    Windows are contiguous from index 0 to the last one containing any sample (or up to
    `duration_s` if given). A window with no frames/chunks still exists (empty tuples) so
    downstream indexing stays dense — detectors mask themselves out on empty input.
    """
    frames = sorted(frames, key=lambda f: f.t_ms)
    chunks = sorted(chunks, key=lambda c: c.t_ms)

    by_frame: dict[int, list[Frame]] = {}
    by_chunk: dict[int, list[AudioChunk]] = {}
    last_idx = 0
    for f in frames:
        k = window_index(f.t_ms, window_s)
        by_frame.setdefault(k, []).append(f)
        last_idx = max(last_idx, k)
    for c in chunks:
        k = window_index(c.t_ms, window_s)
        by_chunk.setdefault(k, []).append(c)
        last_idx = max(last_idx, k)

    if duration_s is not None:
        last_idx = max(last_idx, int(duration_s // window_s) - 1 if duration_s > 0 else 0)

    windows: list[Window] = []
    for k in range(last_idx + 1):
        windows.append(
            Window(
                index=k,
                t_start=k * window_s,
                t_end=(k + 1) * window_s,
                frames=tuple(by_frame.get(k, ())),
                chunks=tuple(by_chunk.get(k, ())),
            )
        )
    return windows


def iter_windows(
    frames: Iterable[Frame],
    chunks: Iterable[AudioChunk],
    *,
    window_s: float = DECISION_WINDOW_S,
    duration_s: float | None = None,
) -> Iterator[Window]:
    """Generator form of :func:`align_windows` for streaming callers."""
    yield from align_windows(frames, chunks, window_s=window_s, duration_s=duration_s)


def av_drift_ms(frames: Iterable[Frame], chunks: Iterable[AudioChunk]) -> float:
    """Measured A/V drift = |last video t_ms − last audio t_ms| after independent decode.

    This is what `hindsight probe` reports against FS2 (<= 100 ms). It is *measured*, not
    assumed — the caller decodes video and audio on separate paths and hands both here.
    """
    frames = list(frames)
    chunks = list(chunks)
    if not frames or not chunks:
        raise ValueError("cannot measure A/V drift without both a video and an audio stream")
    last_frame_ms = max(f.t_ms for f in frames)
    last_chunk_ms = max(c.t_ms for c in chunks)
    return abs(last_frame_ms - last_chunk_ms)
