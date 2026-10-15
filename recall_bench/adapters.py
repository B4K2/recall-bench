"""One uniform surface over every model family in models.yaml, plus wrappers for your own models.

    a = load_adapter(name, device)
    a.tokenize(str) -> list[int]      # model's own vocab, no special tokens
    a.prefix / a.suffix               # special-token ids wrapped around every window
    a.causal, a.mask_id, a.max_ctx, a.align, a.domain
    a.logits(ids: LongTensor (1, T)) -> (1, T, V) float32

Every family works in its *own* token space; the copy task only ever compares a model with
itself (repeat vs control window), so no cross-model vocabulary mapping is needed.
"""
import importlib

import torch
import yaml

from recall_bench.paths import PKG, model_dir, registry


class Adapter:
    causal = True
    mask_id = None
    prefix: list[int] = []
    suffix: list[int] = []

    def __init__(self, name, entry, device, dtype):
        self.name, self.entry, self.device, self.dtype = name, entry, device, dtype
        self.path = entry.get("path") or model_dir(name)
        self.max_ctx = int(entry["max_ctx"])
        self.align = int(entry.get("align", 1))
        self.domain = entry["domain"]

    def tokenize(self, s: str) -> list[int]:
        raise NotImplementedError

    def _forward(self, ids):
        raise NotImplementedError

    @torch.no_grad()
    def logits(self, ids):
        with torch.autocast("cuda", dtype=self.dtype, enabled=self.device == "cuda"):
            out = self._forward(ids)
        out = getattr(out, "logits", out)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out.float()


class _HFTokenizerMixin:
    def _load_tok(self):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)

    def tokenize(self, s):
        return self.tok(s, add_special_tokens=False)["input_ids"]


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


class Mnemos(Adapter):
    """Mnemos-DNA arms (B4K2xx/Mnemos-DNA). Model code is vendored in recall_bench._mnemos."""

    def __init__(self, *a):
        super().__init__(*a)
        from recall_bench._mnemos.tokenizer import BOS_ID, encode
        d = self.path / self.entry["subdir"]
        # HF configs are overlays on Mnemos-DNA's base.yaml (vendored as configs/mnemos_base.yaml)
        cfg = _merge(yaml.safe_load((PKG / "configs/mnemos_base.yaml").read_text()),
                     yaml.safe_load((d / "config.yaml").read_text()))
        mod = importlib.import_module(f"recall_bench._mnemos.models.{cfg.get('model_module', cfg['model_id'])}")
        self.model = mod.build_model(cfg)
        ck = torch.load(d / "model.pt", map_location="cpu", weights_only=False)
        self.model.load_state_dict(ck.get("model_state_dict", ck))
        self.model = self.model.to(self.device, self.dtype).eval()
        self.model.gradient_checkpointing = False
        self._encode = encode
        self.prefix = [BOS_ID] if cfg.get("prepend_bos") else []

    def tokenize(self, s):
        return self._encode(s)

    def _forward(self, ids):
        return self.model(ids)[0]


class HyenaDNA(_HFTokenizerMixin, Adapter):
    def __init__(self, *a):
        super().__init__(*a)
        from transformers import AutoModelForCausalLM
        self._load_tok()
        self.model = AutoModelForCausalLM.from_pretrained(
            self.path, trust_remote_code=True).to(self.device).eval()

    def _forward(self, ids):
        return self.model(input_ids=ids)


class Caduceus(_HFTokenizerMixin, Adapter):
    """Bidirectional masked LM; loaded through Mnemos-DNA's transformers>=5 patches."""
    causal = False

    def __init__(self, *a):
        super().__init__(*a)
        from recall_bench._mnemos.caduceus import _caduceus_class, _mamba_transformers_shim
        _mamba_transformers_shim()
        self._load_tok()
        self.mask_id = self.tok.mask_token_id
        self.model = _caduceus_class(str(self.path)).from_pretrained(
            self.path, trust_remote_code=True).to(self.device).eval()
        # transformers>=5 no longer ties the non-RCPS (Ph) lm_head to the embeddings through
        # Caduceus' super().tie_weights(), leaving it randomly initialised (it is not stored
        # in the checkpoint). Tie it explicitly; RCPS (PS) uses its own set_weight.
        emb = self.model.get_input_embeddings().weight
        if self.model.config.rcps:
            self.model.lm_head.set_weight(emb)
        else:
            self.model.lm_head.weight = emb

    def _forward(self, ids):
        return self.model(input_ids=ids)


class HFMaskedLM(_HFTokenizerMixin, Adapter):
    causal = False

    def __init__(self, *a):
        super().__init__(*a)
        from transformers import AutoModelForMaskedLM
        self._load_tok()
        self.mask_id = self.tok.mask_token_id
        self.prefix = [self.tok.cls_token_id] if self.tok.cls_token_id is not None else []
        self.suffix = [self.tok.sep_token_id] if self.tok.sep_token_id is not None else []
        self.model = AutoModelForMaskedLM.from_pretrained(
            self.path, trust_remote_code=True).to(self.device).eval()

    def _forward(self, ids):
        return self.model(input_ids=ids)


class HFCausalLM(_HFTokenizerMixin, Adapter):
    def __init__(self, *a):
        super().__init__(*a)
        try:
            import fla  # noqa: F401  registers GLA/DeltaNet with AutoModel
        except ImportError:
            pass
        from transformers import AutoModelForCausalLM
        self._load_tok()
        bos = self.tok.bos_token_id
        self.prefix = [bos] if bos is not None else []
        self.model = AutoModelForCausalLM.from_pretrained(
            self.path, trust_remote_code=True, torch_dtype=self.dtype).to(self.device).eval()

    def _forward(self, ids):
        return self.model(input_ids=ids, use_cache=False)


class Evo2(Adapter):
    """Needs the `evo2` package (Transformer Engine/vortex); optional."""

    def __init__(self, *a):
        super().__init__(*a)
        try:
            from evo2 import Evo2 as _E
        except ImportError as e:
            raise RuntimeError("evo2 not installed: uv pip install evo2 (needs Transformer Engine)") from e
        self.model = _E(self.entry["hf_id"].split("/")[-1])

    def tokenize(self, s):
        return list(self.model.tokenizer.tokenize(s))

    def _forward(self, ids):
        return self.model(ids)[0]


FAMILIES = {"mnemos": Mnemos, "hyenadna": HyenaDNA, "caduceus": Caduceus, "hf_mlm": HFMaskedLM,
            "hf_causal": HFCausalLM, "text_causal": HFCausalLM, "evo2": Evo2}


def download(name: str) -> str:
    """Fetch one registry model into checkpoints/<name> (idempotent)."""
    import os
    from huggingface_hub import snapshot_download
    e = registry()[name]
    patterns = [f"{e['subdir']}/*"] if "subdir" in e else None
    return snapshot_download(e["hf_id"], local_dir=model_dir(name), allow_patterns=patterns,
                             token=os.environ.get("HUGGING_FACE_TOKEN") or None)


def load_adapter(name: str, device: str = "cuda", dtype=torch.bfloat16,
                 fetch: bool = False) -> Adapter:
    """A model from the packaged registry (`recall-bench list`)."""
    reg = registry()
    if name not in reg:
        raise KeyError(f"unknown model {name!r}; see `recall-bench list`")
    if not model_dir(name).exists():
        if not fetch:
            raise FileNotFoundError(f"{model_dir(name)} missing: run `recall-bench download {name}`")
        download(name)
    return FAMILIES[reg[name]["family"]](name, reg[name], device, dtype)


def from_hf(hf_id: str, causal: bool = True, domain: str = "dna", max_ctx: int = 2048,
            align: int = 1, device: str = "cuda", dtype=torch.bfloat16) -> Adapter:
    """Any Hugging Face AutoModelForCausalLM / AutoModelForMaskedLM checkpoint (id or local path)
    with its own tokenizer. Set `align` to k for k-mer tokenizers."""
    entry = {"family": "hf_causal" if causal else "hf_mlm", "domain": domain, "path": hf_id,
             "hf_id": hf_id, "max_ctx": max_ctx, "align": align}
    return (HFCausalLM if causal else HFMaskedLM)(hf_id.replace("/", "--"), entry, device, dtype)


class CallableModel(Adapter):
    """Wrap your own model -- no registry entry, no Hugging Face needed.

        model = CallableModel(
            logits_fn=lambda ids: net(ids),       # LongTensor (1, T) -> (1, T, V)
            tokenize=lambda s: [VOCAB[c] for c in s],
            causal=True, max_ctx=131072)

    For masked LMs pass causal=False and mask_id. prefix/suffix are special-token ids put
    around every window (e.g. [BOS] or [CLS] ... [SEP]). align=k for k-mer tokenizers.
    """

    def __init__(self, logits_fn, tokenize, *, causal=True, max_ctx=2048, domain="dna",
                 align=1, mask_id=None, prefix=(), suffix=(), name="custom", device=None):
        if not causal and mask_id is None:
            raise ValueError("masked LMs need mask_id")
        self.name, self._fn, self._tok = name, logits_fn, tokenize
        self.causal, self.mask_id, self.max_ctx = causal, mask_id, int(max_ctx)
        self.domain, self.align = domain, int(align)
        self.prefix, self.suffix = list(prefix), list(suffix)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def tokenize(self, s):
        return list(self._tok(s))

    @torch.no_grad()
    def logits(self, ids):
        out = self._fn(ids)
        out = getattr(out, "logits", out)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out.float()
