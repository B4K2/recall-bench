"""Regenerate paper figures/tables from results/<model>/copy.json.

    uv run python paper/make_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "figures"

# (key, label) in table/legend order; memory arms first, then KDA controls, then baselines.
MODELS = [
    ("mnemos-dna", "Mnemos-DNA (D06)"),
    ("mnemos-dna-c", "Mnemos-DNA-C (D07)"),
    ("pure-kda-d01", "Pure KDA (D01)"),
    ("kda-local-d02", "KDA+Local (D02)"),
    ("hyenadna-small-32k", "HyenaDNA-small-32k"),
    ("hyenadna-medium-450k", "HyenaDNA-medium-450k"),
    ("hyenadna-large-1m", "HyenaDNA-large-1m"),
    ("caduceus-ph-131k", "Caduceus-Ph-131k"),
    ("caduceus-ps-131k", "Caduceus-PS-131k"),
    ("hyenadna-tiny-1k", "HyenaDNA-tiny-1k"),
]
DISTS = [64, 256, 1024, 4096, 16384, 65536, 262144]
MEMORY = {"mnemos-dna", "mnemos-dna-c"}
# Validated categorical order (dataviz reference palette, light mode).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
GRAY = "#b5b4ad"


def load():
    """{model: {dist: row}}; keeps only the full-trial row per distance (drops n=16 pilots)."""
    res = {}
    for key, _ in MODELS:
        f = ROOT / "results" / key / "copy.json"
        if not f.exists():
            continue
        by_d = {}
        for r in json.loads(f.read_text())["rows"]:
            d = r.get("dist")
            if r["status"] != "ok":
                by_d.setdefault(d, r)
            elif r.get("n", 0) >= by_d.get(d, {}).get("n", 0):
                by_d[d] = r
        res[key] = by_d
    return res


def figure(res):
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#6f6e69", "axes.labelcolor": "#2b2b29",
                         "xtick.color": "#6f6e69", "ytick.color": "#6f6e69",
                         "pdf.fonttype": 42, "font.family": "serif"})
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.5), constrained_layout=True)
    others = [k for k, _ in MODELS if k in res and k not in MEMORY
              and any(r["status"] == "ok" for r in res[k].values())]
    label = dict(MODELS)

    def series(k):
        pts = [(d, res[k][d]) for d in DISTS if d in res[k] and res[k][d]["status"] == "ok"]
        return ([d for d, _ in pts], [r["copy_advantage"] for _, r in pts],
                [r["ci_lo"] for _, r in pts], [r["ci_hi"] for _, r in pts])

    # (a) full scale: memory arms in color, everything else gray.
    for k in others:
        x, y, _, _ = series(k)
        a.plot(x, y, color=GRAY, lw=1.2, zorder=1)
    for i, k in enumerate(m for m in ("mnemos-dna", "mnemos-dna-c") if m in res):
        x, y, lo, hi = series(k)
        a.fill_between(x, lo, hi, color=PALETTE[i], alpha=0.18, lw=0)
        a.plot(x, y, color=PALETTE[i], lw=2, marker=MARKERS[i], ms=4, label=label[k], zorder=3)
    a.plot([], [], color=GRAY, lw=1.2, label="All other DNA LMs")
    a.set_ylim(-0.05, 1.0)
    a.set_title("(a) Full scale", loc="left", fontsize=8.5)

    # (b) zoom on models without content-addressed memory.
    for i, k in enumerate(others):
        x, y, lo, hi = series(k)
        c = (PALETTE[2:] + ["#4a4a46"])[i]  # slots 1-2 stay reserved for the memory arms
        x = [v * 1.06 ** (i - len(others) / 2) for v in x]  # dodge overlapping error bars
        b.errorbar(x, y, yerr=[[v - l for v, l in zip(y, lo)], [h - v for v, h in zip(y, hi)]],
                   color=c, lw=1.2, marker=MARKERS[i % len(MARKERS)], ms=3.5, capsize=1.5,
                   elinewidth=0.7, label=label[k])
    b.set_ylim(-0.015, 0.006)
    b.set_title("(b) Zoom: models without k-mer memory", loc="left", fontsize=8.5)

    for ax in (a, b):
        ax.set_xscale("log", base=2)
        ax.set_xticks(DISTS, ["64", "256", "1K", "4K", "16K", "64K", "256K"])
        ax.axhline(0, color="#6f6e69", lw=0.6, ls="--", zorder=0)
        ax.grid(axis="y", color="#e6e5df", lw=0.5)
        ax.set_xlabel("Distance $d$ between copies (bp)")
    a.set_ylabel("Copy advantage (nats/bp)")
    a.legend(frameon=False, fontsize=7, loc="lower left", bbox_to_anchor=(0, 0.1))
    b.legend(frameon=False, fontsize=6, ncol=2, loc="lower right")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "recall_vs_distance.pdf")
    fig.savefig(OUT / "recall_vs_distance.png", dpi=200)


def table(res):
    """LaTeX body rows: copy advantage x 100 (centi-nats/bp) per distance."""
    lines = []
    for k, lab in MODELS:
        if k not in res:
            continue
        cells = []
        for d in DISTS:
            r = res[k].get(d)
            if r is None:
                cells.append("--")
            elif r["status"] != "ok":
                cells.append(r"\textsc{oom}" if "OOM" in r["status"] else r"$\times$")
            else:
                v = 100 * r["copy_advantage"]
                sig = r["ci_lo"] > 0 or r["ci_hi"] < 0
                s = f"{v:+.2f}" if abs(v) < 10 else f"{v:+.1f}"
                cells.append(rf"\textbf{{{s}}}" if sig and k in MEMORY else (s + ("$^*$" if sig else "")))
        lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
        if k == "kda-local-d02":
            lines.append(r"\midrule")
    (OUT / "recall_table.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    r = load()
    figure(r)
    table(r)
    print(f"wrote {OUT}")
