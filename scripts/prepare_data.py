"""Background corpora for the copy probe.

dna:  held-out hg38 — the Basenji/Enformer *test* intervals (same split HyenaDNA and Mnemos-DNA
      hold out), merged into contiguous blocks, uppercased. Saved as uint8 ASCII + block offsets.
text: wikitext-103 test split, concatenated (for the text-LM positive control).

    uv run python scripts/prepare_data.py dna text
"""
import argparse
import gzip
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FASTA_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"
BED_URL = "https://storage.googleapis.com/basenji_barnyard2/sequences_human.bed"


def fetch(url: str, dst: Path):
    if dst.exists() and dst.stat().st_size > 0:
        return
    print(f"downloading {url}")
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dst)


def dna():
    out = DATA / "hg38_test.npz"
    if out.exists():
        print(f"{out} exists"); return
    fa, bed = DATA / "hg38.fa.gz", DATA / "sequences_human.bed"
    fetch(BED_URL, bed)
    fetch(FASTA_URL, fa)
    by_chrom = {}
    for line in bed.read_text().split("\n"):
        f = line.split("\t")
        if len(f) >= 4 and f[3] == "test":
            by_chrom.setdefault(f[0], []).append((int(f[1]), int(f[2])))
    merged = {}
    for c, iv in by_chrom.items():
        iv.sort(); m = [list(iv[0])]
        for s, e in iv[1:]:
            if s <= m[-1][1]: m[-1][1] = max(m[-1][1], e)
            else: m.append([s, e])
        merged[c] = m
    from Bio import SeqIO
    blocks = []
    with gzip.open(fa, "rt") as fh:
        for rec in SeqIO.parse(fh, "fasta"):
            if rec.id in merged:
                s = str(rec.seq).upper()
                blocks += [np.frombuffer(s[a:b].encode(), dtype=np.uint8) for a, b in merged[rec.id]]
                print(f"{rec.id}: {len(merged[rec.id])} blocks")
    seq = np.concatenate(blocks)
    offsets = np.cumsum([0] + [len(b) for b in blocks])
    np.savez(out, seq=seq, offsets=offsets)
    print(f"{out}: {len(seq)/1e6:.1f} Mbp in {len(blocks)} blocks, "
          f"longest {max(map(len, blocks))/1e3:.0f} kbp")


def text():
    out = DATA / "wikitext103_test.txt"
    if out.exists():
        print(f"{out} exists"); return
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    out.write_text("".join(ds["text"]))
    print(f"{out}: {out.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="+", choices=["dna", "text"])
    DATA.mkdir(exist_ok=True)
    for w in ap.parse_args().what:
        {"dna": dna, "text": text}[w]()
