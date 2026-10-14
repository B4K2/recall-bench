"""Run the copy-recall probe on ONE model. Resumable: finished points in
results/<name>/copy.json are reused.

    uv run python scripts/run_copy.py hyenadna-small-32k
    uv run python scripts/run_copy.py mnemos-dna --distances 64,1024,16384 --trials 32

Default sweep: for each distance d, ctx = smallest power of two >= d + 2*seg + 512, so the
curve reads recall vs distance at minimal cost. --ctx fixes one context for every d instead.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch

from recall_bench import ROOT
from recall_bench.adapters import load_adapter
from recall_bench.copy_task import DNABackground, TextBackground, run_point

DISTANCES = "64,256,1024,4096,16384,65536,262144"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--distances", default=DISTANCES)
    ap.add_argument("--ctx", type=int, default=0)
    ap.add_argument("--seg", type=int, default=0, help="default 256 bp (dna) / 64 tokens (text)")
    ap.add_argument("--trials", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    ad = load_adapter(a.name, a.device)
    seg = a.seg or (256 if ad.domain == "dna" else 64)
    bg = (DNABackground(ROOT / "data/hg38_test.npz") if ad.domain == "dna"
          else TextBackground(ROOT / "data/wikitext103_test.txt", ad))
    out = ROOT / "results" / a.name / "copy.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = json.loads(out.read_text())["rows"] if out.exists() else []
    done = {(r["ctx"], r["dist"], r["seg"], r["trials"], r["seed"]) for r in rows}
    oom = False
    for d in map(int, a.distances.split(",")):
        ctx = a.ctx or 1 << (d + 2 * seg + 512 - 1).bit_length()
        key = (ctx, d, seg, a.trials, a.seed)
        if key in done:
            print(f"reused: ctx={ctx} d={d}"); continue
        if a.ctx and d + 2 * seg + 64 > ctx:
            continue
        t0 = time.time()
        if oom:
            r = {"status": "skip: OOM at shorter ctx"}
        else:
            try:
                r = run_point(ad, bg, ctx, d, seg, a.trials, a.seed)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache(); oom = True; r = {"status": "skip: OOM"}
        if ad.domain == "dna":
            r["bg_contiguous"] = bg.contiguous(ctx)
        r.update(model=a.name, ctx=ctx, dist=d, seg=seg, trials=a.trials, seed=a.seed,
                 seconds=round(time.time() - t0, 1))
        rows.append(r)
        out.write_text(json.dumps({"model": a.name, "rows": rows}, indent=1))
        adv = f"{r['copy_advantage']:+.4f} ± {r['stderr']:.4f}" if r["status"] == "ok" else r["status"]
        print(f"ctx={ctx:>8} d={d:>7}  copy_advantage={adv}  ({r['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
