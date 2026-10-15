from recall_bench._mnemos.models.d05.gpt import KmerGPT
from recall_bench._mnemos.models.d06.attention import FastWeightKmerAttention


def build_model(config: dict) -> KmerGPT:
    return KmerGPT(config, attn_cls=FastWeightKmerAttention)
