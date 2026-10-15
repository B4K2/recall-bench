from recall_bench._mnemos.models.d01.attention import KDAAttention
from recall_bench._mnemos.models.shared.gpt import GPT
from recall_bench._mnemos.models.shared.hybrid import interleave_layer_types
from recall_bench._mnemos.models.shared.local_attention import WindowedLocalAttention


def build_model(config: dict) -> GPT:
    layer_classes = interleave_layer_types(
        config["num_layers"], KDAAttention, WindowedLocalAttention, config.get("hybrid_ratio", 4)
    )
    return GPT(layer_classes, config)
