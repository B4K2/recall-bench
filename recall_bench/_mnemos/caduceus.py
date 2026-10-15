"""Caduceus loading helpers for transformers>=5 (from Mnemos-DNA models/external)."""


def _mamba_transformers_shim():
    """mamba-ssm's `utils.generation` imports `GreedySearchDecoderOnlyOutput` /
    `SampleDecoderOnlyOutput`, which transformers renamed to `GenerateDecoderOnlyOutput` in
    v4.41. Caduceus' remote code imports mamba_ssm at module scope, so without this alias the
    checkpoint cannot be loaded at all under transformers>=5. Only generation-output *type
    aliases* are added — no behaviour is patched, and the Caduceus forward path never uses
    them."""
    from transformers import generation
    if not hasattr(generation, "GreedySearchDecoderOnlyOutput"):
        target = generation.GenerateDecoderOnlyOutput
        for name in ("GreedySearchDecoderOnlyOutput", "SampleDecoderOnlyOutput",
                     "GreedySearchEncoderDecoderOutput", "SampleEncoderDecoderOutput"):
            if not hasattr(generation, name):
                setattr(generation, name, target)


def _caduceus_class(hf_id: str, name: str = "CaduceusForMaskedLM"):
    """Resolve (and repair) the checkpoint's remote model class.

    `CaduceusForMaskedLM`, not the bare backbone: the released weights are all prefixed
    `caduceus.`, so loading the backbone class alone silently randomly-initialises every
    parameter (transformers only warns).

    The released code was written against transformers 4.38 and overrides `tie_weights`
    without calling `super()` (and skips it entirely when the config unties the embeddings),
    so under transformers>=5 the loader trips on the `all_tied_weights_keys` mapping that
    `tie_weights` is now expected to populate. The patch gives every instance an empty mapping — correct here, because the plain
    `Caduceus` backbone (no LM head) ties nothing."""
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    cls = get_class_from_dynamic_module(f"modeling_caduceus.{name}", hf_id)
    if not getattr(cls, "_dna_kda_patched", False):
        original_init = cls.__init__

        def __init__(self, *a, **kw):
            original_init(self, *a, **kw)
            if not hasattr(self, "all_tied_weights_keys"):
                self.all_tied_weights_keys = {}

        original_tie = cls.tie_weights

        def tie_weights(self, *a, **kw):
            # transformers>=5 calls tie_weights(missing_keys=..., recompute_mapping=...);
            # the released method takes no arguments and only mirrors the bidirectional
            # Mamba in/out projections, which is argument-independent.
            return original_tie(self)

        cls.__init__ = __init__
        cls.tie_weights = tie_weights
        cls._dna_kda_patched = True
    return cls

