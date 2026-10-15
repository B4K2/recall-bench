"""Zero-shot long-range recall benchmark for genomic (and, as a control, text) LMs.

    import recall_bench as rb
    model = rb.CallableModel(logits_fn=net, tokenize=encode, causal=True, max_ctx=131072)
    rows = rb.evaluate(model, distances=[64, 1024, 16384], trials=32)

    model = rb.load_adapter("hyenadna-small-32k", fetch=True)      # from the registry
    model = rb.from_hf("some/causal-dna-lm", causal=True, max_ctx=8192)
"""
from recall_bench.adapters import CallableModel, download, from_hf, load_adapter
from recall_bench.copy_task import CONTROLS, VARIANTS, DNABackground, TextBackground
from recall_bench.paths import data_dir, home, model_dir, registry
from recall_bench.runner import DISTANCES, evaluate

__version__ = "0.2.1"
__all__ = ["CallableModel", "CONTROLS", "DISTANCES", "DNABackground", "TextBackground",
           "VARIANTS", "data_dir", "download", "evaluate", "from_hf", "home", "load_adapter",
           "model_dir", "registry"]
