"""Numerical parity gate: MLX bidirectional forward vs PyTorch/HF eager reference.

Mirrors a2d's identity gate (a2d transform/identity.py) for the MLX port: the
same GPT-2 weights loaded into (a) mlx_dllm's bidirectional forward and (b) an
HF eager-attention GPT-2 with all causality neutralized - a2d's alpha=1 decode
configuration - must produce matching logits at every position.

Runs on stock distilgpt2: a2d conversion does not alter base attention weights
(bidirectionality is a decode-time policy), so forward parity on stock weights
makes the same numerical claim.

Tolerances (fp32, measured 2026-07 on M-series, mlx 0.32.1 / torch 2.13):

- a2d's own gate uses 1e-6, but that compares two *torch* graphs built to be
  bit-identical. Across frameworks the bar is the fp32 noise floor: against a
  float64 torch golden, torch-f32 deviates 2.0e-4 and MLX-CPU-f32 2.0e-4;
  their mutual max-abs-diff is 8e-5. CPU gate = 1e-3 (>10x measured margin).
- MLX on Metal accumulates fp32 matmuls at ~1e-3 relative error (a single
  768-dim-reduction matmul already shows 0.1 max-abs vs 1.7e-4 on CPU), giving
  a measured whole-model diff of 0.073 on logits of magnitude ~85. GPU gate =
  0.5: ~7x the measured noise, ~20x below the ~10.0 signal a causality
  regression produces (measured bidirectional-vs-causal pos-0 logit diff).
"""

import mlx.core as mx
import numpy as np
import pytest
import torch

import mlx_dllm

MODEL = "distilbert/distilgpt2"
CPU_TOLERANCE = 1e-3
GPU_TOLERANCE = 0.5


@pytest.fixture(scope="module")
def loaded():
    return mlx_dllm.load(MODEL)


@pytest.fixture(scope="module")
def ref_model():
    """HF GPT-2 in a2d's alpha=1 decode configuration: fully bidirectional.

    In transformers 5.12.x eager attention applies only the model-level
    ``create_causal_mask`` additive mask, so that is the one seam to
    neutralize. Earlier 5.x (verified on 5.0) also kept a per-layer causal
    ``bias`` buffer inside eager attention; the guarded fill covers those
    versions within the ``>=5.5.4,<5.13`` test pin.
    """
    from transformers import GPT2LMHeadModel
    from transformers.models.gpt2 import modeling_gpt2

    model = GPT2LMHeadModel.from_pretrained(
        MODEL, attn_implementation="eager", dtype=torch.float32
    ).eval()
    for m in model.modules():
        if isinstance(m, modeling_gpt2.GPT2Attention) and hasattr(m, "bias"):
            m.bias.fill_(True)
    orig = modeling_gpt2.create_causal_mask
    modeling_gpt2.create_causal_mask = lambda *args, **kwargs: None
    try:
        yield model
    finally:
        modeling_gpt2.create_causal_mask = orig


def _probes(tokenizer, vocab_size):
    text = tokenizer.encode("The quick brown fox jumps over the lazy dog.")
    rand = np.random.default_rng(0).integers(0, vocab_size, size=64).tolist()
    return [text, rand]


def _assert_parity(loaded, ref_model, device, tolerance):
    model, tokenizer, _ = loaded
    prev = mx.default_device()
    mx.set_default_device(device)
    try:
        for ids in _probes(tokenizer, ref_model.config.vocab_size):
            mlx_logits = np.array(
                mlx_dllm.bidirectional_forward(model, mx.array([ids]))
            )
            with torch.no_grad():
                ref_logits = ref_model(torch.tensor([ids])).logits.numpy()
            # All-position logits, not last-position-only.
            assert (
                mlx_logits.shape
                == ref_logits.shape
                == (1, len(ids), ref_model.config.vocab_size)
            )
            diff = float(np.abs(mlx_logits - ref_logits).max())
            assert diff < tolerance, f"max |mlx - torch| = {diff} >= {tolerance}"
    finally:
        mx.set_default_device(prev)


def test_bidirectional_parity_cpu(loaded, ref_model):
    _assert_parity(loaded, ref_model, mx.cpu, CPU_TOLERANCE)


@pytest.mark.skipif(not mx.metal.is_available(), reason="no Metal GPU")
def test_bidirectional_parity_gpu(loaded, ref_model):
    _assert_parity(loaded, ref_model, mx.gpu, GPU_TOLERANCE)


def test_forward_is_noncausal(loaded):
    """The override really changes attention: early-position logits must move
    when causality is dropped (mlx-lm's default forward is causal)."""
    model, tokenizer, _ = loaded
    ids = mx.array([tokenizer.encode("The quick brown fox jumps over the lazy dog.")])
    bidir = np.array(mlx_dllm.bidirectional_forward(model, ids))
    causal = np.array(model(ids))
    assert float(np.abs(bidir[0, 0] - causal[0, 0]).max()) > 1.0
    # and the default path is untouched after the scoped patch exits
    causal_again = np.array(model(ids))
    np.testing.assert_array_equal(causal, causal_again)
