"""LlmClaudeValidator — Claude multimodal, strict JSON, one retry, then FAIL OPEN.

Cloud, OPT-IN only (configs/cloud.yaml, on_device_only: false). Highest quality, costs
money, requires explicit consent (the CLI prints a banner before the first call). Strict
JSON out; on a parse failure retry once; on a second failure or any error, keep=True,
failed_open=True — never lose a memory to a flaky API (CLAUDE.md §4).

Cost is estimated from the model's published token pricing in config and the actual token
usage returned by the API — it is measured, never assumed. If the SDK/key is missing it
fails open.
"""

from __future__ import annotations

import json
import time

from ..._deps import optional
from ...contracts import Evidence, Segment, Verdict

_SYSTEM = (
    "You are a strict validator for a passive memory system. Given keyframes and a short "
    "transcript of a candidate 'interesting' segment, decide whether it is worth remembering "
    "(a conversation, reading, a demonstration, a notable object/place) or is a false positive "
    "(walking, idle, empty scene). You may only DROP or KEEP; you cannot invent events. "
    'Reply with STRICT JSON: {"keep": bool, "confidence": 0..1, "reason": "<=200 chars", "tags": []}'
)


class LlmClaudeValidator:
    name = "llm_claude"

    def __init__(self, cfg) -> None:
        c = cfg.get("cascade.llm_claude")
        self.model = c.get("model", "claude-sonnet-4-6")
        self.max_tokens = int(c.get("max_tokens", 512))
        self.retries = int(c.get("retries_on_parse_failure", 1))
        # pricing is config so the $ figure is auditable, not magic. USD per 1M tokens.
        self.price_in = float(c.get("price_per_mtok_in", 3.0))
        self.price_out = float(c.get("price_per_mtok_out", 15.0))
        self._client = None

    def _client_or_none(self):
        if self._client is not None:
            return self._client
        anthropic = optional("anthropic")
        if anthropic is None:
            return None
        try:
            self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        except Exception:  # noqa: BLE001
            return None
        return self._client

    def validate(self, seg: Segment, ev: Evidence) -> Verdict:
        t0 = time.perf_counter()
        client = self._client_or_none()
        if client is None or not ev.keyframes_jpeg:
            return self._fail_open(seg, "anthropic sdk/key or keyframes missing", t0)
        content = self._build_content(ev)
        last_err = "parse failure"
        for _ in range(self.retries + 1):
            try:
                resp = client.messages.create(
                    model=self.model, max_tokens=self.max_tokens, system=_SYSTEM,
                    messages=[{"role": "user", "content": content}],
                )
                cost = self._cost(resp)
                parsed = self._parse(resp)
                if parsed is not None:
                    return Verdict(
                        segment_id=seg.segment_id, keep=bool(parsed["keep"]),
                        confidence=float(parsed.get("confidence", 0.5)),
                        reason=str(parsed.get("reason", ""))[:200],
                        tags=tuple(parsed.get("tags", [])), validator="llm_claude", route="validate",
                        cost_usd=cost, latency_ms=(time.perf_counter() - t0) * 1000,
                    )
            except Exception as exc:  # noqa: BLE001
                last_err = f"{type(exc).__name__}"
        return self._fail_open(seg, last_err, t0)

    def _build_content(self, ev: Evidence) -> list:
        import base64

        blocks: list = []
        for jpeg in ev.keyframes_jpeg:
            blocks.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg",
                           "data": base64.b64encode(jpeg).decode()},
            })
        text = f"Transcript: {ev.transcript or '(none)'}\nDetector evidence: {ev.detector_evidence}"
        blocks.append({"type": "text", "text": text})
        return blocks

    @staticmethod
    def _parse(resp) -> dict | None:
        try:
            text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            return json.loads(text[text.index("{") : text.rindex("}") + 1])
        except Exception:  # noqa: BLE001
            return None

    def _cost(self, resp) -> float:
        u = getattr(resp, "usage", None)
        if u is None:
            return 0.0
        return (u.input_tokens / 1e6) * self.price_in + (u.output_tokens / 1e6) * self.price_out

    def _fail_open(self, seg: Segment, why: str, t0: float) -> Verdict:
        return Verdict(
            segment_id=seg.segment_id, keep=True, confidence=0.0,
            reason=f"llm_claude fail-open ({why}); kept"[:200],
            tags=("fail_open",), validator="llm_claude", route="validate",
            cost_usd=0.0, latency_ms=(time.perf_counter() - t0) * 1000, failed_open=True,
        )
