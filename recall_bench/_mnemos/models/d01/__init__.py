from recall_bench._mnemos.models.d01.attention import KDAAttention
from recall_bench._mnemos.models.shared.gpt import GPT


def build_model(config: dict) -> GPT:
    return GPT(KDAAttention, config)
