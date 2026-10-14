"""Download ONE model from models.yaml into checkpoints/<name>. Idempotent.

    uv run python scripts/download_model.py hyenadna-small-32k
    uv run python scripts/download_model.py --list
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

from recall_bench import ROOT, model_dir, registry

load_dotenv(ROOT / ".env")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    reg = registry()
    if a.list or not a.name:
        for n, e in reg.items():
            print(f"{n:24s} {e['group']:10s} {e['family']:12s} {e['hf_id']}")
        return
    if a.name not in reg:
        sys.exit(f"unknown model {a.name!r}; see --list")
    e = reg[a.name]
    patterns = [f"{e['subdir']}/*"] if "subdir" in e else None
    path = snapshot_download(e["hf_id"], local_dir=model_dir(a.name), allow_patterns=patterns,
                             token=os.environ.get("HUGGING_FACE_TOKEN") or None)
    print(path)


if __name__ == "__main__":
    main()
