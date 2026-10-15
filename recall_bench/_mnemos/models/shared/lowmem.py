"""Low-memory execution mode: how the benchmark suite reaches contexts that OOM natively.

Engaged only as a fallback after a native OOM (see benchmarks/harness.py), never by default,
because it changes the cost profile of a measurement and must not be silently mixed into the
§6.3 scaling curve — rows produced under it are labelled separately.

Two distinct mechanisms, because the inference OOM and the training OOM have different causes:

  prefill chunking (this module's `chunk_size`)
      Inference runs under no_grad, so activations are freed as soon as they are consumed and
      there is essentially nothing resident to move to RAM — the peak is the *working set of a
      single layer* (q, k, v, alpha, beta each B*T*d_model). Offloading cannot shrink that;
      only processing fewer tokens at a time can. The KDA recurrence supports this exactly:
      chunk the sequence, thread the recurrent state from one chunk into the next via
      chunk_kda's initial_state/output_final_state. This is the chunkwise-parallel form of the
      *same* recurrence, not a sliding window — a token in the final chunk still sees position
      0 through the carried state, so no information is dropped at chunk boundaries.

      Exactness requires two details that are easy to get wrong:
        - the short causal conv on q/k/v has a receptive field of `kernel_size - 1` tokens, so
          each chunk is fed that many tokens of left context which are then discarded;
        - RoPE arms must be told the absolute position offset of each chunk (see
          models/shared/pos.py), or every chunk is rotated as if it began at position 0.
      tests/test_kda_equivalence.py checks chunked == whole-sequence output.

  activation offload to CPU RAM (`offload_activations`)
      The training-step measurement retains every saved tensor for backward, which is what
      genuinely benefits from living in pinned host RAM. torch.autograd.graph.save_on_cpu
      handles this; it is applied in benchmarks/scaling.py around the train step only, and is
      PCIe-bound, so expect it to be slow.
"""
import contextlib
import threading

_state = threading.local()

DEFAULT_CHUNK_SIZE = 16384


def chunk_size() -> int | None:
    """Sequence chunk for the KDA prefill path, or None when running natively."""
    return getattr(_state, "chunk_size", None)


def offload_activations() -> bool:
    return getattr(_state, "offload_activations", False)


def enabled() -> bool:
    return chunk_size() is not None or offload_activations()


@contextlib.contextmanager
def low_memory_mode(chunk: int | None = DEFAULT_CHUNK_SIZE, offload: bool = True):
    """Enable chunked prefill and/or CPU activation offload for the enclosed block."""
    prev_chunk = getattr(_state, "chunk_size", None)
    prev_offload = getattr(_state, "offload_activations", False)
    _state.chunk_size = chunk
    _state.offload_activations = offload
    try:
        yield
    finally:
        _state.chunk_size = prev_chunk
        _state.offload_activations = prev_offload
