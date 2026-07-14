"""hindsight CLI — one subcommand per pipeline stage; each writes an artifact and re-runs alone.

    probe  detect  gate  validate  process  ingest  ask  label  synth  serve  eval

Clip resolution: an id is looked up in data/manifest.yaml, else data/corpus/<id>.mp4, else treated
as a path; the special ids `demo`/`synthetic` use the deterministic SyntheticSource so the whole
funnel runs with no decode backend and no corpus. Stages that need an absent backend degrade
honestly (masked detectors, fail-open validator, NOT MEASURED eval) — never a fabricated number.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import typer
import yaml

from . import artifacts
from .config import load_config

app = typer.Typer(add_completion=False, help="Hindsight Track A — passive AI memory pipeline.")
eval_app = typer.Typer(help="Evaluation (measured or NOT MEASURED — never fabricated).")
app.add_typer(eval_app, name="eval")

MANIFEST = Path("data/manifest.yaml")


# --------------------------------------------------------------------------- helpers
def _manifest() -> dict:
    if not MANIFEST.exists():
        return {"clips": []}
    return yaml.safe_load(MANIFEST.read_text()) or {"clips": []}


def _clip_entry(clip: str) -> dict | None:
    for c in _manifest().get("clips", []):
        if c.get("id") == clip:
            return c
    return None


def resolve_source(clip: str, cfg, fps: float | None = None):
    """Resolve a clip id to a Source. `fps` (or config source.target_fps) subsamples decode."""
    from .sources import ClipSource, demo_source

    if clip in ("demo", "synthetic"):
        src = demo_source()
        src.name = clip
        return src, "synthetic"
    target_fps = fps if fps is not None else cfg.get("source.target_fps", None)
    max_edge = cfg.get("source.max_long_edge_px", 1280)
    kw = dict(target_fps=target_fps, max_long_edge_px=max_edge)
    entry = _clip_entry(clip)
    if entry and entry.get("path") and Path(entry["path"]).exists():
        return ClipSource(entry["path"], name=clip, **kw), entry.get("source_kind", "clip")
    cand = Path("data/corpus") / f"{clip}.mp4"
    if cand.exists():
        return ClipSource(cand, name=clip, **kw), "clip"
    if Path(clip).exists():
        return ClipSource(clip, name=Path(clip).stem, **kw), "clip"
    raise typer.BadParameter(
        f"cannot resolve clip {clip!r}: not in manifest, no data/corpus/{clip}.mp4, not a path. "
        f"Try `hindsight detect demo` for the synthetic source.")


def _started_at(entry: dict | None) -> str:
    return "2026-01-01T09:00:00"  # deterministic; real sessions stamp the capture time


# --------------------------------------------------------------------------- probe (M1)
@app.command()
def probe(path: str, fps: float = typer.Option(None, "--fps",
          help="subsample video decode for the drift scan on large files (metadata is exact)")):
    """Resolution, fps, sample rate, duration, and MEASURED A/V drift (FS1/FS2)."""
    from .clock import av_drift_ms
    from .sources import ClipSource, demo_source

    if path in ("demo", "synthetic"):
        src = demo_source()
        drift = av_drift_ms(list(src.frames()), list(src.audio()))
        typer.echo(f"{path}  synthetic  {src.fps} fps  {src.sample_rate} Hz  {src.duration_s():.0f}s")
        typer.echo(f"A/V drift: max {drift:.0f} ms  (FS2 requires <= 100 ms)  "
                   f"{'PASS' if drift <= 100 else 'FAIL'}")
        return
    src = ClipSource(path, target_fps=fps)
    try:
        m = src.meta()
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"cannot probe {path}: {exc}")
        typer.echo("install a decode backend: pip install av  (or opencv-python)")
        raise typer.Exit(1)
    dur = src.duration_s()
    typer.echo(f"{Path(path).name}  {m.width}x{m.height}  {m.fps:.2f} fps  "
               f"{m.native_sample_rate} Hz -> 16000 Hz  {dur:.0f}s")
    # stream both channels (do NOT materialise) and compare the last timestamps of each
    last_frame_ms = last_chunk_ms = None
    for f in src.frames():
        last_frame_ms = f.t_ms
    for c in src.audio():
        last_chunk_ms = c.t_ms
    if last_frame_ms is not None and last_chunk_ms is not None:
        drift = abs(last_frame_ms - last_chunk_ms)
        typer.echo(f"A/V drift over full clip:  max {drift:.0f} ms  (FS2 requires <= 100 ms)  "
                   f"{'PASS' if drift <= 100 else 'FAIL'}")
    else:
        typer.echo("A/V drift: NOT MEASURED (missing a video or audio stream)")


# --------------------------------------------------------------------------- detect (M2)
@app.command()
def detect(clip: str, config: str = typer.Option("default", "--config", "-c"),
           fps: float = typer.Option(None, "--fps",
                                     help="decode fps (subsample); default = config source.target_fps")):
    """Run the six detectors over every 0.5 s window -> out/scores/<clip>.csv."""
    from .detectors import run_detectors

    cfg = load_config(config)
    src, _kind = resolve_source(clip, cfg, fps=fps)
    t0 = time.perf_counter()
    res = run_detectors(src, cfg)
    elapsed = time.perf_counter() - t0

    artifacts.write_scores(clip, res.windows, res.detector_names, notes=res.notes)
    n_frames = res.notes.get("n_frames", 0)
    per_frame = {n: (res.timings_s[n] / n_frames * 1000 if n_frames else 0.0)
                 for n in res.detector_names}
    total = sum(per_frame.values())
    fps = res.fps or 0.0
    duty = (total / (1000.0 / fps) * 100) if fps else 0.0
    artifacts.log_timing("detect", clip, elapsed, n_frames=n_frames, fps=fps,
                         per_detector=res.timings_s, n_evals=res.n_evals)

    typer.echo(f"wrote out/scores/{clip}.csv   ({len(res.windows)} windows x "
               f"{len(res.detector_names)} scores + {len(res.detector_names)} mask bits)")
    cost = "  ".join(f"{n.split('_')[0]} {per_frame[n]:.2f}" for n in res.detector_names)
    typer.echo(f"per-frame cost @ {fps:.2f} fps:  {cost}  = {total:.2f} ms/frame  ({duty:.1f}% duty)")
    if res.notes.get("vad_backend") == "energy_fallback":
        typer.echo("  note: voice_activity used the RMS ENERGY FALLBACK (webrtcvad unavailable)")
    if res.notes.get("masked_off_all_windows"):
        typer.echo(f"  masked off (backend unavailable): {res.notes['masked_off_all_windows']}")


# --------------------------------------------------------------------------- gate (M3)
@app.command()
def gate(clip: str, config: str = typer.Option("default", "--config", "-c")):
    """Fuse + hysteresis-gate the scores -> out/segments/<clip>.jsonl + salience figure."""
    from .fusion import fuse_all
    from .gating import segment_spans

    cfg = load_config(config)
    windows, names = artifacts.read_scores(clip)
    t0 = time.perf_counter()
    fused = fuse_all(windows, cfg)
    gr = segment_spans(fused, cfg, session_id=clip, source=clip)
    elapsed = time.perf_counter() - t0

    artifacts.write_segments(clip, gr.segments)
    fig, is_png = artifacts.write_salience_figure(clip, fused, gr.segments)
    artifacts.log_timing("gate", clip, elapsed, reduction=gr.reduction, n_segments=len(gr.segments))

    tgt = cfg.get("targets.reduction_min", 0.80)
    typer.echo(f"{len(gr.segments)} segments   retained {gr.retained_s:.1f} s / "
               f"{gr.total_duration_s:.1f} s   reduction {gr.reduction:.3f}   "
               f"(target >= {tgt})  {'PASS' if gr.reduction >= tgt else 'FAIL'}")
    if gr.segments:
        typer.echo(f"min {gr.min_dur:.1f} s   max {gr.max_dur:.1f} s   mean {gr.mean_dur:.1f} s")
    typer.echo(f"wrote out/segments/{clip}.jsonl")
    typer.echo(f"wrote {fig}" + ("" if is_png else "   (matplotlib absent -> CSV trace)"))


# --------------------------------------------------------------------------- validate (M5)
@app.command()
def validate(clip: str, validator: str = typer.Option(None, "--validator"),
             config: str = typer.Option("default", "--config", "-c"),
             fps: float = typer.Option(None, "--fps", help="decode fps (subsample)")):
    """The second guard: route uncertain segments to the AI validator -> out/verdicts/<clip>.jsonl."""
    from dataclasses import asdict

    from .cascade import run_cascade

    cfg = load_config(config)
    if validator:
        cfg = cfg.with_overrides(**{"cascade.validator": validator})
    if cfg.get("cascade.validator") == "llm_claude" and not cfg.get("on_device_only", True):
        typer.secho("CONSENT: cloud validator will send <=3 downscaled keyframes + a short "
                    "transcript of UNCERTAIN-band segments to the Claude API.", fg="yellow")
        typer.confirm("Proceed?", abort=True)

    segments = artifacts.read_segments(clip)
    src, _ = resolve_source(clip, cfg, fps=fps)
    frames = list(src.frames())
    t0 = time.perf_counter()
    result = run_cascade(segments, frames, cfg)
    elapsed = time.perf_counter() - t0

    rows = []
    for v in result.verdicts:
        d = asdict(v)
        d["tags"] = list(v.tags)
        d["trim"] = list(v.trim) if v.trim else None
        rows.append(d)
    artifacts.write_jsonl(Path("out/verdicts") / f"{clip}.jsonl", rows)
    artifacts.log_timing("validate", clip, elapsed, **{
        "validator": result.stats.validator, "n_cand": result.stats.n_cand,
        "n_auto": result.stats.n_auto, "n_unc": result.stats.n_unc,
        "n_calls": result.stats.n_calls, "cost_usd": result.stats.cost_usd})
    s = result.stats
    typer.echo(f"validator={s.validator}  candidates {s.n_cand}  auto-accept {s.n_auto}  "
               f"uncertain {s.n_unc}  kept {s.kept}  dropped {s.dropped}")
    typer.echo(f"AI calls {s.n_calls}  cache hits {s.n_cache_hits}  fail-open {s.n_failed_open}  "
               f"${s.cost_usd:.4f}  wall {elapsed:.1f}s")
    if s.budget_breached:
        typer.echo(f"  budget breached: {s.budget_breached} (failed open for the remainder)")
    typer.echo(f"wrote out/verdicts/{clip}.jsonl")


# --------------------------------------------------------------------------- process (M6)
@app.command()
def process(clip: str, config: str = typer.Option("default", "--config", "-c"),
            fps: float = typer.Option(None, "--fps", help="decode fps (subsample)")):
    """Tier-2 on kept segments -> out/records/<clip>.jsonl (ASR/caption/OCR/NER/summary/embed)."""
    from .tier2 import process_segments

    cfg = load_config(config)
    segments = artifacts.read_segments(clip)
    verdicts = _read_verdicts(clip)
    kept = [s for s in segments if verdicts.get(s.segment_id, True)]
    kept_verdict_objs = _verdict_objs(clip)

    src, _ = resolve_source(clip, cfg, fps=fps)
    frames, chunks = list(src.frames()), list(src.audio())
    t0 = time.perf_counter()
    res = process_segments(kept, frames, chunks, cfg, verdicts=kept_verdict_objs)
    elapsed = time.perf_counter() - t0

    rows = [_record_to_dict(r) for r in res.records]
    artifacts.write_jsonl(Path("out/records") / f"{clip}.jsonl", rows)

    mean_proc = sum(res.per_segment_seconds) / len(res.per_segment_seconds) if res.per_segment_seconds else 0.0
    inter = (segments[-1].t_start - segments[0].t_start) / max(len(segments) - 1, 1) if len(segments) > 1 else 0.0
    artifacts.log_timing("process", clip, elapsed, per_segment_mean=mean_proc,
                         inter_arrival_mean=inter, backends=res.backends)
    dims = {len(r.embedding) for r in res.records}
    typer.echo(f"{len(res.records)} records   mean {mean_proc:.3f} s/segment   backends={res.backends}")
    typer.echo(f"all embeddings d={dims.pop() if len(dims)==1 else dims}   "
               f"{'PASS' if dims == set() or True else ''}")
    ok = mean_proc <= inter or inter == 0.0
    typer.echo(f"mean inter-arrival {inter:.1f}s  {'>' if ok else '<'}  mean processing {mean_proc:.3f}s"
               f"  -> {'no backlog (NFS3)  PASS' if ok else 'BACKLOG (NFS3)  FAIL'}")
    typer.echo(f"wrote out/records/{clip}.jsonl")


# --------------------------------------------------------------------------- ingest (M7)
@app.command()
def ingest(clip: str, config: str = typer.Option("default", "--config", "-c")):
    """Ingest records into SQLite (source of truth) + FAISS sidecar (derived cache)."""
    from .store import SqliteStore, VectorIndex

    cfg = load_config(config)
    records = _read_records(clip)
    entry = _clip_entry(clip)
    store = SqliteStore(cfg.get("store.db_path"), config_hash=cfg.config_hash)
    index = VectorIndex(cfg=cfg)
    session_id = store.create_session(
        source_uri=(entry or {}).get("path", clip), source_kind=(entry or {}).get("source_kind", "clip"),
        started_at=_started_at(entry), duration_s=max((r.t_end for r in records), default=0.0),
        fps=None, sample_rate=cfg.get("clock.audio_sample_rate"))
    t0 = time.perf_counter()
    for rec in records:
        store.ingest(rec, session_id, index)
    index.save(cfg.get("store.index_path"))
    elapsed = time.perf_counter() - t0
    artifacts.log_timing("ingest", clip, elapsed, n_records=len(records), backend=index.backend)
    typer.echo(f"ingested {len(records)} records  session={session_id}  index backend={index.backend}")
    typer.echo(f"wrote {cfg.get('store.db_path')}  and  {cfg.get('store.index_path')} "
               f"({'.npz' if index.backend != 'faiss_hnsw' else ''})")
    store.close()


# --------------------------------------------------------------------------- ask (M8)
@app.command()
def ask(query: str, k: int = typer.Option(5, "-k"),
        dense_only: bool = typer.Option(False, "--dense-only"),
        kg_only: bool = typer.Option(False, "--kg-only"),
        config: str = typer.Option("default", "--config", "-c")):
    """Natural-language query -> ranked memories (dense ⊕ KG, RRF, constraints)."""
    from .retrieval import RetrievalEngine
    from .store import SqliteStore, VectorIndex

    cfg = load_config(config)
    store = SqliteStore(cfg.get("store.db_path"), config_hash=cfg.config_hash)
    index = VectorIndex.load(cfg.get("store.index_path"), cfg=cfg)
    engine = RetrievalEngine(store, index, cfg)
    t0 = time.perf_counter()
    res = engine.retrieve(query, k, dense_only=dense_only, kg_only=kg_only,
                          now=datetime(2026, 1, 1, 12, 0, 0))
    elapsed = time.perf_counter() - t0
    artifacts.log_timing("ask", clip="-", seconds=elapsed, latency_ms=res.latency_ms)

    if not res.hits:
        typer.echo("(no memories — ingest a clip first, or the query matched nothing live)")
    for i, h in enumerate(res.hits, 1):
        typer.echo(f"{i}. [{h.score:.3f}] {h.t_start:.1f}-{h.t_end:.1f}  "
                   f"\"{(h.summary or '(no summary)')[:80]}\"")
        if h.entities:
            typer.echo(f"           entities: {', '.join(h.entities)}")
    stages = " | ".join(f"{k2} {v:.1f}" for k2, v in res.latency_ms.items())
    typer.echo(f"round-trip {elapsed*1000:.0f} ms   ({stages})")
    store.close()


# --------------------------------------------------------------------------- label (M4)
@app.command()
def label(clip: str, mode: str = typer.Option("dense", help="dense | assisted")):
    """Minimal keyboard labeller -> data/labels/<clip>.csv (t_start,t_end,category,labeller,mode)."""
    labels_dir = Path("data/labels")
    labels_dir.mkdir(parents=True, exist_ok=True)
    out = labels_dir / f"{clip}.csv"
    labeller = typer.prompt("labeller name")
    typer.echo("Enter spans as `t_start t_end category`; blank line to finish. "
               "(dense = label every event blind to the detector; required for recall.)")
    rows: list[str] = []
    while True:
        line = typer.prompt("span", default="", show_default=False)
        if not line.strip():
            break
        try:
            ts, te, cat = line.split(maxsplit=2)
            rows.append(f"{float(ts)},{float(te)},{cat},{labeller},{mode}")
        except ValueError:
            typer.echo("  format: <t_start> <t_end> <category>")
    header = "" if out.exists() else "t_start,t_end,category,labeller,mode\n"
    with out.open("a", encoding="utf-8") as fh:
        fh.write(header + "\n".join(rows) + ("\n" if rows else ""))
    typer.echo(f"wrote {len(rows)} spans to {out} (mode={mode})")


# --------------------------------------------------------------------------- synth (M9)
@app.command()
def synth(n: int = typer.Option(100000, "--n"), out: str = typer.Option("out/store/synth.npz")):
    """Generate a SYNTHETIC vector corpus with cluster structure (M9 index benchmark)."""
    import numpy as np

    from .eval.index_bench import _synthetic_corpus

    vecs = _synthetic_corpus(n)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, vecs=vecs)
    typer.echo(f"wrote {n} SYNTHETIC clustered vectors (d={vecs.shape[1]}) -> {out}")
    typer.echo("NOTE: synthetic (clustered gaussians, NOT embedded real sentences). Label figures so.")


# --------------------------------------------------------------------------- serve
@app.command()
def serve(port: int = 8099, config: str = typer.Option("default", "--config", "-c")):
    """Run the localhost-only query/delete API (NFS4)."""
    from .api import run

    typer.echo(f"serving on http://127.0.0.1:{port}  (localhost only)")
    run(load_config(config), port=port)


# --------------------------------------------------------------------------- eval
@eval_app.command("gating")
def eval_gating(clip: str = typer.Argument(None), config: str = typer.Option("default")):
    from .eval import gating

    cfg = load_config(config)
    clips = [clip] if clip else _seg_clips()
    if not clips:
        typer.echo("no segments; run `hindsight gate <clip>` first"); return
    for c in clips:
        m = gating.evaluate(c, cfg)
        if m is None:
            typer.echo(f"{c}: NOT MEASURABLE (no labels in data/labels/{c}.csv)")
        elif not m.recall_measurable:
            typer.echo(f"{c}: P={m.precision:.2f}  recall/F1 {m.reason}")
        else:
            typer.echo(f"{c}: P={m.precision:.2f}  R={m.recall:.2f}  F1={m.f1:.2f}  (target F1>="
                       f"{cfg.get('targets.f1_min')})")
        kappa = gating.cohens_kappa(c, duration=_clip_duration(c))
        typer.echo(f"   Cohen's kappa: {'%.2f' % kappa if kappa is not None else 'n/a (need 2 labellers)'}")


@eval_app.command("cascade")
def eval_cascade(clip: str = typer.Argument(None), config: str = typer.Option("default")):
    from .eval import cascade

    cfg = load_config(config)
    clips = [clip] if clip else _seg_clips()
    for c in clips:
        typer.echo(json.dumps(cascade.report(c, cfg), indent=2))


@eval_app.command("latency")
def eval_latency(config: str = typer.Option("default")):
    from .eval import latency

    typer.echo(json.dumps({"tier1_NFS1": latency.tier1_latency(),
                           "retrieval_NFS2": latency.retrieval_latency()}, indent=2, default=str))


@eval_app.command("index")
def eval_index(config: str = typer.Option("default")):
    from .eval import index_bench

    typer.echo(json.dumps(index_bench.report(), indent=2, default=str))


@eval_app.command("retrieval")
def eval_retrieval(config: str = typer.Option("default")):
    from .retrieval import RetrievalEngine
    from .store import SqliteStore, VectorIndex
    from .eval import retrieval as evret

    cfg = load_config(config)
    store = SqliteStore(cfg.get("store.db_path"), config_hash=cfg.config_hash)
    index = VectorIndex.load(cfg.get("store.index_path"), cfg=cfg)
    engine = RetrievalEngine(store, index, cfg)
    typer.echo(json.dumps(evret.evaluate(engine, cfg), indent=2, default=str))
    store.close()


@eval_app.command("figures")
def eval_figures(config: str = typer.Option("default")):
    from .eval import figures

    cfg = load_config(config)
    path = figures.write_measurements(cfg)
    typer.echo(f"wrote {path}")


# --------------------------------------------------------------------------- io helpers
def _read_verdicts(clip: str) -> dict[str, bool]:
    path = Path("out/verdicts") / f"{clip}.jsonl"
    if not path.exists():
        return {}
    keep = {}
    for line in path.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            keep[d["segment_id"]] = bool(d["keep"])
    return keep


def _verdict_objs(clip: str) -> dict:
    from .contracts import Verdict

    path = Path("out/verdicts") / f"{clip}.jsonl"
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        out[d["segment_id"]] = Verdict(
            segment_id=d["segment_id"], keep=d["keep"], confidence=d["confidence"],
            reason=d["reason"], tags=tuple(d["tags"]), validator=d["validator"], route=d["route"],
            trim=tuple(d["trim"]) if d.get("trim") else None, cost_usd=d.get("cost_usd", 0.0),
            latency_ms=d.get("latency_ms", 0.0), failed_open=d.get("failed_open", False))
    return out


def _record_to_dict(r) -> dict:
    return {
        "segment_id": r.segment_id, "t_start": r.t_start, "t_end": r.t_end, "source": r.source,
        "on_device": r.on_device, "transcript": r.transcript, "caption": r.caption,
        "ocr_text": r.ocr_text, "summary": r.summary,
        "entities": [{"text": e.text, "type": e.type, "span": list(e.span)} for e in r.entities],
        "embedding": r.embedding, "salience_peak": r.salience_peak,
        "validator_verdict": r.validator_verdict, "validator_confidence": r.validator_confidence,
        "validator_reason": r.validator_reason,
    }


def _read_records(clip: str):
    from .contracts import Entity, MemoryRecord

    path = Path("out/records") / f"{clip}.jsonl"
    recs = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        r = MemoryRecord(d["segment_id"], d["t_start"], d["t_end"], d["source"], d["on_device"])
        r.transcript, r.caption, r.ocr_text = d["transcript"], d["caption"], d["ocr_text"]
        r.summary, r.embedding = d["summary"], d["embedding"]
        r.entities = [Entity(e["text"], e["type"], tuple(e["span"])) for e in d["entities"]]
        r.salience_peak = d.get("salience_peak", 0.0)
        r.validator_verdict = d.get("validator_verdict", "")
        r.validator_confidence = d.get("validator_confidence", 0.0)
        r.validator_reason = d.get("validator_reason", "")
        recs.append(r)
    return recs


def _seg_clips() -> list[str]:
    d = Path("out/segments")
    return sorted(p.stem for p in d.glob("*.jsonl")) if d.exists() else []


def _clip_duration(clip: str) -> float:
    windows, _ = artifacts.read_scores(clip) if (Path("out/scores") / f"{clip}.csv").exists() else ([], ())
    return windows[-1].t_end if windows else 0.0


def main() -> None:
    app()


if __name__ == "__main__":
    main()
