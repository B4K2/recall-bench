"""Regenerate paper figures/tables from results/<model>/copy.json.

    uv run python paper/make_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
DEFAULT = {"variant": "exact", "mut_rate": 0.0, "control": "native"}
MUT_RATES = [0.0, 0.01, 0.05, 0.1, 0.2]
# One-hue sequential ramp (reference palette blue, light -> dark) for mutation rate.
RAMP = ["#0d366b", "#1c5cab", "#3987e5", "#6da7ec", "#9ec5f4"]
PROTO_DISTS = [64, 1024, 16384, 65536]


def load(variant="exact", mut_rate=0.0, control="native"):
    """{model: {dist: row}} for one protocol; keeps only the full-trial row per distance (drops
    n=16 pilots). Rows without protocol fields predate the options and are exact/native."""
    want = {"variant": variant, "mut_rate": mut_rate, "control": control}
    res = {}
    for key, _ in MODELS:
        f = ROOT / "results" / key / "copy.json"
        if not f.exists():
            continue
        by_d = {}
        for r in json.loads(f.read_text())["rows"]:
            if any(r.get(k, DEFAULT[k]) != v for k, v in want.items()):
                continue
            d = r.get("dist")
            if r["status"] != "ok":
                by_d.setdefault(d, r)
            elif r.get("n", 0) >= by_d.get(d, {}).get("n", 0):
                by_d[d] = r
        if by_d:
            res[key] = by_d
    return res


def figure(res):
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#6f6e69", "axes.labelcolor": "#2b2b29",
                         "xtick.color": "#6f6e69", "ytick.color": "#6f6e69",
                         "pdf.fonttype": 42, "font.family": "serif"})
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.7), constrained_layout=True,
                                   gridspec_kw={"width_ratios": [1, 1.25]})
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

    # (b) every model x distance as an annotated grid: readable without decoding overlapping lines.
    rows = [k for k, _ in MODELS if k in res and any(r["status"] == "ok" for r in res[k].values())]
    grid = np.full((len(rows), len(DISTS)), np.nan)
    for i, k in enumerate(rows):
        for j, d in enumerate(DISTS):
            r = res[k].get(d)
            if r is not None and r["status"] == "ok":
                grid[i, j] = r["copy_advantage"]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("w2b", ["#ffffff", PALETTE[0]])
    cmap.set_bad("#f1f0ea")
    b.imshow(np.clip(grid, 0, None), cmap=cmap, vmin=0, vmax=1.0, aspect="auto")
    for i, k in enumerate(rows):
        for j, d in enumerate(DISTS):
            v, r = grid[i, j], res[k].get(d)
            if np.isnan(v):
                txt = "--" if r is None else ("OOM" if "OOM" in r["status"] else "n/a")
                b.text(j, i, txt, ha="center", va="center", fontsize=5.5, color="#9a9992")
            else:
                b.text(j, i, f"{v + 0.0:.2f}".replace("-0.00", "0.00"), ha="center", va="center", fontsize=6,
                       color="white" if v > 0.5 else "#2b2b29",
                       fontweight="bold" if k in MEMORY else "normal")
    b.set_yticks(range(len(rows)), [label[k] for k in rows], fontsize=6.5)
    b.set_xticks(range(len(DISTS)), ["64", "256", "1K", "4K", "16K", "64K", "256K"])
    b.tick_params(length=0)
    for sp in b.spines.values():
        sp.set_visible(False)
    b.axhline(len(MEMORY) - 0.5, color="#2b2b29", lw=0.8)  # memory arms above, everything else below
    b.set_xlabel("Distance $d$ between copies (bp)")
    b.set_title("(b) All models (0.00 = no recall)", loc="left", fontsize=8.5)

    for ax in (a,):
        ax.set_xscale("log", base=2)
        ax.set_xticks(DISTS, ["64", "256", "1K", "4K", "16K", "64K", "256K"])
        ax.axhline(0, color="#6f6e69", lw=0.6, ls="--", zorder=0)
        ax.grid(axis="y", color="#e6e5df", lw=0.5)
        ax.set_xlabel("Distance $d$ between copies (bp)")
    a.set_ylabel("Copy advantage (nats/bp)\n(higher = better recall)")
    a.legend(frameon=False, fontsize=7, loc="lower left", bbox_to_anchor=(0, 0.1))
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


def _fmt(r):
    if r is None:
        return "--"
    if r["status"] != "ok":
        return r"$\times$"
    v = 100 * r["copy_advantage"]
    sig = r["ci_lo"] > 0 or r["ci_hi"] < 0
    return (f"{v:+.1f}" if abs(v) >= 10 else f"{v:+.2f}") + ("$^*$" if sig else "")


def protocol_table():
    """Foreign control and reverse-complement copies, same units as the main table."""
    fo, rc = load(control="foreign"), load(variant="revcomp")
    lines = []
    for k, lab in MODELS:
        if k not in fo and k not in rc:
            continue
        cells = [_fmt(src.get(k, {}).get(d)) for src in (fo, rc) for d in PROTO_DISTS]
        lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
        if k == "kda-local-d02":
            lines.append(r"\midrule")
    (OUT / "protocol_table.tex").write_text("\n".join(lines) + "\n")


def mutation_figure():
    """Copy advantage vs distance per substitution rate, one panel per memory model."""
    arms = [k for k in ("mnemos-dna", "mnemos-dna-c") if load(variant="mutate", mut_rate=0.01).get(k)]
    if not arms:
        return
    fig, axes = plt.subplots(1, len(arms), figsize=(3.5 * len(arms), 2.4), sharey=True,
                             constrained_layout=True)
    axes = list(np.atleast_1d(axes))
    for ax, k in zip(axes, arms):
        for i, p in enumerate(MUT_RATES):
            res = load(variant="mutate", mut_rate=p) if p else load()
            pts = [(d, res[k][d]) for d in PROTO_DISTS if d in res.get(k, {})
                   and res[k][d]["status"] == "ok"]
            if not pts:
                continue
            x, y = [d for d, _ in pts], [r["copy_advantage"] for _, r in pts]
            ax.fill_between(x, [r["ci_lo"] for _, r in pts], [r["ci_hi"] for _, r in pts],
                            color=RAMP[i], alpha=0.15, lw=0)
            ax.plot(x, y, color=RAMP[i], lw=2, marker="o", ms=3.5,
                    label="exact" if p == 0 else f"{p:.0%} subst.")
        ax.set_xscale("log", base=2)
        ax.set_xticks(PROTO_DISTS, ["64", "1K", "16K", "64K"])
        ax.axhline(0, color="#6f6e69", lw=0.6, ls="--", zorder=0)
        ax.grid(axis="y", color="#e6e5df", lw=0.5)
        ax.set_xlabel("Distance $d$ between copies (bp)")
        ax.set_title(dict(MODELS)[k], loc="left", fontsize=8.5)
    axes[0].set_ylabel("Copy advantage (nats/bp)")
    axes[-1].legend(frameon=False, fontsize=6.5, loc="lower left")
    fig.savefig(OUT / "mutation.pdf")
    fig.savefig(OUT / "mutation.png", dpi=200)


if __name__ == "__main__":
    r = load()
    figure(r)
    table(r)
    protocol_table()
    mutation_figure()
    print(f"wrote {OUT}")
