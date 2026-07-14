"""ClipLocalValidator — CLIP ViT-B/32 zero-shot. On-device. NFS4-safe. THE DEFAULT.

keep = (max_pos - max_neg) > margin, over the prompt bank in configs/prompts.yaml (prompts
in config, not code — they are swept). No network is ever opened.

MUST NOT raise (contract). If torch/open-clip are missing, or there are no keyframes, it
FAILS OPEN: keep=True, failed_open=True — degrading to Tier-1-only, the known-good state.
Model + preprocess are built lazily and cached on the instance.
"""

from __future__ import annotations

import time

import numpy as np
import yaml

from ..._deps import optional
from ...contracts import Evidence, Segment, Verdict


class ClipLocalValidator:
    name = "clip_local"

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        c = cfg.get("cascade.clip_local")
        self.model_name = c.get("model", "ViT-B/32")
        self.margin = float(c.get("margin", 0.05))
        self.prompts_file = c.get("prompts_file", "configs/prompts.yaml")
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._text_feats: dict | None = None

    def _lazy_build(self) -> bool:
        if self._model is not None:
            return True
        torch = optional("torch")
        open_clip = optional("open_clip")
        if torch is None or open_clip is None:
            return False
        arch = self.model_name.replace("/", "-")
        model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained="openai")
        model.eval()
        self._model, self._preprocess = model, preprocess
        self._tokenizer = open_clip.get_tokenizer(arch)
        bank = yaml.safe_load(open(self.prompts_file))
        with torch.no_grad():
            pos = self._encode_text(bank["positive"], torch)
            neg = self._encode_text(bank["negative"], torch)
        self._text_feats = {"pos": pos, "neg": neg}
        return True

    def _encode_text(self, prompts, torch):
        toks = self._tokenizer(list(prompts))
        feats = self._model.encode_text(toks)
        return feats / feats.norm(dim=-1, keepdim=True)

    def validate(self, seg: Segment, ev: Evidence) -> Verdict:
        t0 = time.perf_counter()
        try:
            if not ev.keyframes_jpeg:
                return self._fail_open(seg, "no keyframes to score", t0)
            if not self._lazy_build():
                return self._fail_open(seg, "torch/open-clip unavailable", t0)
            keep, margin, tag = self._score(ev)
            return Verdict(
                segment_id=seg.segment_id, keep=keep, confidence=float(min(abs(margin) * 5, 1.0)),
                reason=f"clip_local: (max_pos-max_neg)={margin:.3f} vs margin {self.margin} -> {tag}",
                tags=(tag,), validator="clip_local", route="validate",
                cost_usd=0.0, latency_ms=(time.perf_counter() - t0) * 1000,
            )
        except Exception as exc:  # noqa: BLE001 — contract: never raise
            return self._fail_open(seg, f"error: {type(exc).__name__}", t0)

    def _score(self, ev: Evidence):
        import io

        torch = optional("torch")
        from PIL import Image

        imgs = [self._preprocess(Image.open(io.BytesIO(b)).convert("RGB")) for b in ev.keyframes_jpeg]
        batch = torch.stack(imgs)
        with torch.no_grad():
            feats = self._model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            max_pos = (feats @ self._text_feats["pos"].T).max().item()
            max_neg = (feats @ self._text_feats["neg"].T).max().item()
        margin = max_pos - max_neg
        keep = margin > self.margin
        return keep, margin, ("interesting" if keep else "background")

    def _fail_open(self, seg: Segment, why: str, t0: float) -> Verdict:
        return Verdict(
            segment_id=seg.segment_id, keep=True, confidence=0.0,
            reason=f"clip_local fail-open ({why}); kept (Tier-1-only)"[:200],
            tags=("fail_open",), validator="clip_local", route="validate",
            cost_usd=0.0, latency_ms=(time.perf_counter() - t0) * 1000, failed_open=True,
        )
