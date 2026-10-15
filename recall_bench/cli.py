"""`recall-bench` command line.

    recall-bench list                                   # packaged model registry
    recall-bench prepare dna                            # held-out hg38 background (~1 GB download)
    recall-bench run hyenadna-small-32k --fetch         # download if needed, sweep, save JSON
    recall-bench run mnemos-dna --variant mutate --mut-rate 0.05
    recall-bench run-hf InstaDeepAI/nucleotide-transformer-v2-50m-multi-species \\
        --masked --max-ctx 2048 --align 6
    recall-bench report results/                        # copy advantage table
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

from recall_bench.copy_task import CONTROLS, VARIANTS
from recall_bench.paths import model_dir, registry
from recall_bench.runner import DEFAULTS, DISTANCES


def _sweep_args(p):
    p.add_argument("--distances", default=",".join(map(str, DISTANCES)))
    p.add_argument("--ctx", type=int, default=0, help="fixed window length (default: per distance)")
    p.add_argument("--seg", type=int, default=0, help="default 256 bp (dna) / 64 tokens (text)")
    p.add_argument("--trials", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--variant", choices=VARIANTS, default="exact")
    p.add_argument("--mut-rate", type=float, default=0.0, help="substitution rate for --variant mutate")
    p.add_argument("--control", choices=CONTROLS, default="native")
    p.add_argument("--device", default=None)
    p.add_argument("--results", default="results", help="results/<model>/copy.json is written here")


def _run(ad, a, name):
    from recall_bench.runner import evaluate
    if a.variant == "mutate" and not a.mut_rate:
        sys.exit("--variant mutate needs --mut-rate > 0")
    evaluate(ad, [int(d) for d in a.distances.split(",")], seg=a.seg or None, trials=a.trials,
             seed=a.seed, ctx=a.ctx, variant=a.variant, mut_rate=a.mut_rate, control=a.control,
             out=Path(a.results) / name / "copy.json")


def _device(a):
    import torch
    return a.device or ("cuda" if torch.cuda.is_available() else "cpu")


def cmd_list(a):
    for n, e in registry().items():
        print(f"{n:24s} {e['group']:10s} {e['family']:12s} {e['max_ctx']:>8}  {e['hf_id']}")


def cmd_download(a):
    from recall_bench.adapters import download
    for n in a.names:
        print(download(n))


def cmd_prepare(a):
    from recall_bench import data
    for w in a.what:
        {"dna": data.prepare_dna, "text": data.prepare_text}[w]()


def cmd_run(a):
    from recall_bench.adapters import load_adapter
    ad = load_adapter(a.name, _device(a), fetch=a.fetch)
    _run(ad, a, a.name)
    if a.cleanup:
        shutil.rmtree(model_dir(a.name), ignore_errors=True)


def cmd_run_hf(a):
    from recall_bench.adapters import from_hf
    ad = from_hf(a.hf_id, causal=not a.masked, domain=a.domain, max_ctx=a.max_ctx,
                 align=a.align, device=_device(a))
    _run(ad, a, a.name or a.hf_id.replace("/", "--"))


def cmd_report(a):
    want = {"variant": a.variant, "mut_rate": a.mut_rate if a.variant == "mutate" else 0.0,
            "control": a.control}
    table, dists = {}, set()
    for f in sorted(Path(a.results).glob("*/copy.json")):
        rows = json.loads(f.read_text())["rows"]
        for r in rows:
            if any(r.get(k, DEFAULTS[k]) != v for k, v in want.items()):
                continue
            cell = table.setdefault(f.parent.name, {})
            prev = cell.get(r["dist"])
            if r["status"] != "ok":
                cell.setdefault(r["dist"], "×" if "max_ctx" in r["status"] else "oom")
            elif not isinstance(prev, dict) or r["n"] >= prev["n"]:
                cell[r["dist"]] = r
            dists.add(r["dist"])
    if not table:
        sys.exit("no matching results")
    ds = sorted(dists)
    print(f"copy advantage (nats/bp); variant={a.variant} control={a.control}; * = 95% CI excludes 0")
    print(f"{'model':24s}" + "".join(f"{d:>10}" for d in ds))
    for m, cell in table.items():
        out = []
        for d in ds:
            r = cell.get(d, "-")
            if isinstance(r, dict):
                sig = "*" if r["ci_lo"] > 0 or r["ci_hi"] < 0 else " "
                r = f"{r['copy_advantage']:+.4f}{sig}"
            out.append(f"{r:>10}")
        print(f"{m:24s}" + "".join(out))


def main(argv=None):
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    ap = argparse.ArgumentParser(prog="recall-bench", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="show the model registry").set_defaults(fn=cmd_list)
    p = sub.add_parser("download", help="fetch registry models")
    p.add_argument("names", nargs="+"); p.set_defaults(fn=cmd_download)
    p = sub.add_parser("prepare", help="build background corpora")
    p.add_argument("what", nargs="+", choices=["dna", "text"]); p.set_defaults(fn=cmd_prepare)
    p = sub.add_parser("run", help="sweep one registry model")
    p.add_argument("name"); _sweep_args(p)
    p.add_argument("--fetch", action="store_true", help="download the model if missing")
    p.add_argument("--cleanup", action="store_true", help="delete the weights afterwards")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("run-hf", help="sweep any Hugging Face causal or masked LM")
    p.add_argument("hf_id"); _sweep_args(p)
    p.add_argument("--masked", action="store_true", help="AutoModelForMaskedLM (default causal)")
    p.add_argument("--domain", choices=["dna", "text"], default="dna")
    p.add_argument("--max-ctx", type=int, default=2048)
    p.add_argument("--align", type=int, default=1, help="k for k-mer tokenizers")
    p.add_argument("--name", default=None, help="results subdirectory (default: the id)")
    p.set_defaults(fn=cmd_run_hf)
    p = sub.add_parser("report", help="print a results table")
    p.add_argument("results", nargs="?", default="results")
    p.add_argument("--variant", choices=VARIANTS, default="exact")
    p.add_argument("--mut-rate", type=float, default=0.0)
    p.add_argument("--control", choices=CONTROLS, default="native")
    p.set_defaults(fn=cmd_report)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
