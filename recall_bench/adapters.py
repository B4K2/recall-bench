"""One uniform surface over every model family in models.yaml.

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

from recall_bench import ROOT, model_dir, registry


class Adapter:
    causal = True
    mask_id = None
    prefix: list[int] = []
    suffix: list[int] = []

    def __init__(self, name, entry, device, dtype):
        self.name, self.entry, self.device, self.dtype = name, entry, device, dtype
        self.path = model_dir(name)
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
    """Mnemos-DNA arms (B4K2xx/Mnemos-DNA). Model code comes from the dna-kda package."""

    def __init__(self, *a):
        super().__init__(*a)
        from dataset.tokenizer import BOS_ID, encode
        d = self.path / self.entry["subdir"]
        # HF configs are overlays on Mnemos-DNA's base.yaml (vendored as configs/mnemos_base.yaml)
        cfg = _merge(yaml.safe_load((ROOT / "configs/mnemos_base.yaml").read_text()),
                     yaml.safe_load((d / "config.yaml").read_text()))
        mod = importlib.import_module(f"models.{cfg.get('model_module', cfg['model_id'])}")
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
        from models.external import _caduceus_class, _mamba_transformers_shim
        _mamba_transformers_shim()
        self._load_tok()
        self.mask_id = self.tok.mask_token_id
        self.model = _caduceus_class(str(self.path)).from_pretrained(
            self.path, trust_remote_code=True).to(self.device).eval()

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


def load_adapter(name: str, device: str = "cuda", dtype=torch.bfloat16) -> Adapter:
    entry = registry()[name]
    if not model_dir(name).exists():
        raise SystemExit(f"checkpoints/{name} missing: uv run python scripts/download_model.py {name}")
    return FAMILIES[entry["family"]](name, entry, device, dtype)
