"""Distance sweeps over one model, optionally resumable through a results JSON."""
import json
import time
from pathlib import Path

import torch

from recall_bench.copy_task import DNABackground, TextBackground, run_point
from recall_bench.paths import data_dir

DISTANCES = (64, 256, 1024, 4096, 16384, 65536, 262144)
DEFAULTS = {"variant": "exact", "mut_rate": 0.0, "control": "native"}


def default_background(adapter):
    if adapter.domain == "dna":
        p = data_dir() / "hg38_test.npz"
        if not p.exists():
            raise FileNotFoundError(f"{p} missing: run `recall-bench prepare dna`")
        return DNABackground(p)
    p = data_dir() / "wikitext103_test.txt"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run `recall-bench prepare text`")
    return TextBackground(p, adapter)


def protocol_key(r):
    """Identity of a sweep point; rows written before the protocol options existed are exact/native."""
    return (r["ctx"], r["dist"], r["seg"], r["trials"], r["seed"],
            *(r.get(k, v) for k, v in DEFAULTS.items()))


def _atomic_write(path, text):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def evaluate(adapter, distances=DISTANCES, *, seg=None, trials=128, seed=0, ctx=0,
             variant="exact", mut_rate=0.0, control="native", background=None, out=None,
             verbose=True):
    """Copy advantage vs distance for one model. Returns a list of row dicts.

    ctx=0 picks, per distance, the smallest power-of-two window that fits the probe; a fixed
    ctx evaluates every distance at that one window length instead. With `out` (a JSON path)
    finished points are reused and each trial is checkpointed, so an interrupted sweep resumes
    exactly where it stopped. After one out-of-memory point, longer ones are skipped."""
    seg = seg or (256 if adapter.domain == "dna" else 64)
    bg = background or default_background(adapter)
    opts = {"variant": variant, "mut_rate": float(mut_rate) if variant == "mutate" else 0.0,
            "control": control}
    rows, done = [], set()
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        rows = json.loads(out.read_text())["rows"] if out.exists() else []
        done = {protocol_key(r) for r in rows}
    tag = "" if opts == DEFAULTS else f"_{variant}{opts['mut_rate'] or ''}_{control}"
    result, oom = [], False
    for d in map(int, distances):
        c = ctx or 1 << (d + 2 * seg + 512 - 1).bit_length()
        if ctx and d + 2 * seg + 64 > ctx:
            continue
        base = {"model": adapter.name, "ctx": c, "dist": d, "seg": seg, "trials": trials,
                "seed": seed, **opts}
        key = protocol_key(base)
        if key in done:
            r = next(r for r in rows if protocol_key(r) == key)
            result.append(r)
            if verbose:
                print(f"reused: ctx={c} d={d}", flush=True)
            continue
        t0 = time.time()
        part = out.parent / f"partial_{c}_{d}_{seg}_{trials}_{seed}{tag}.json" if out else None
        if oom:
            r = {"status": "skip: OOM at shorter ctx"}
        else:
            try:
                prev = json.loads(part.read_text()) if part and part.exists() else []
                if prev and verbose:
                    print(f"resuming ctx={c} d={d} at trial {len(prev)}/{trials}", flush=True)
                r = run_point(adapter, bg, c, d, seg, trials, seed, prev,
                              (lambda ds: _atomic_write(part, json.dumps(ds))) if part else None,
                              **opts)
                if part:
                    part.unlink(missing_ok=True)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache(); oom = True; r = {"status": "skip: OOM"}
        if adapter.domain == "dna" and hasattr(bg, "contiguous"):
            r["bg_contiguous"] = bg.contiguous(c)
        r.update(base, seconds=round(time.time() - t0, 1))
        result.append(r)
        if out is not None:
            rows.append(r)
            out.write_text(json.dumps({"model": adapter.name, "rows": rows}, indent=1))
        if verbose:
            adv = (f"{r['copy_advantage']:+.4f} ± {r['stderr']:.4f}" if r["status"] == "ok"
                   else r["status"])
            print(f"ctx={c:>8} d={d:>7}  copy_advantage={adv}  ({r['seconds']}s)", flush=True)
    return result
