"""Evidence pack builder — BOUNDED evidence == bounded cost (CLAUDE.md §4).

The validator sees, and only sees: <= 3 keyframes (<= 512 px long edge, JPEG q70) at
{t_start+pre_roll, peak, t_end-post_roll}; a cheap `whisper tiny.en` transcript for
uncertain-band segments only, cached by content hash; and the detector evidence vector.

Keyframe JPEG encoding uses Pillow or OpenCV if present; the ASR transcript uses
faster-whisper if present. When a backend is absent that field is simply empty — the
validator then works from whatever evidence it does have (and clip_local, needing frames,
fails open). No fabrication.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._deps import optional
from ..contracts import Evidence, Frame, Segment


def _nearest_frame(frames: list[Frame], t_s: float) -> Frame | None:
    if not frames:
        return None
    return min(frames, key=lambda f: abs(f.t_ms / 1000.0 - t_s))


def _encode_jpeg(rgb: np.ndarray, max_long_edge: int, quality: int) -> bytes | None:
    h, w = rgb.shape[:2]
    scale = min(1.0, max_long_edge / max(h, w))
    if scale < 1.0:
        nh, nw = int(h * scale), int(w * scale)
        ys = np.linspace(0, h - 1, nh).astype(int)
        xs = np.linspace(0, w - 1, nw).astype(int)
        rgb = rgb[ys][:, xs]
    PIL = optional("PIL.Image") or optional("PIL")
    if PIL is not None:
        from PIL import Image
        import io

        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
        return buf.getvalue()
    cv2 = optional("cv2")
    if cv2 is not None:
        ok, enc = cv2.imencode(".jpg", rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
        return enc.tobytes() if ok else None
    return None


def build_evidence(seg: Segment, frames: list[Frame], cfg,
                   transcript: str | None = None) -> Evidence:
    ev_cfg = cfg.get("cascade.evidence")
    n_kf = int(ev_cfg["n_keyframes"])
    max_edge = int(ev_cfg["max_long_edge_px"])
    quality = int(ev_cfg["jpeg_quality"])

    ts = list(seg.keyframe_ts)[:n_kf] or [seg.active_start, seg.salience_peak, seg.active_end]
    jpegs: list[bytes] = []
    for t in ts:
        fr = _nearest_frame(frames, t)
        if fr is None:
            continue
        enc = _encode_jpeg(fr.rgb, max_edge, quality)
        if enc is not None:
            jpegs.append(enc)

    return Evidence(
        segment_id=seg.segment_id,
        keyframes_jpeg=tuple(jpegs),
        transcript=transcript,
        detector_evidence=dict(seg.detector_evidence),
        duration_s=seg.duration,
        salience_peak=seg.salience_peak,
        salience_mean=seg.salience_mean,
    )
