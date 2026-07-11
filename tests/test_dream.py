"""Parity gate + smoke for the adopted Fast-dLLM-mlx Qwen/Dream engine.

Ports test_parity.py's identity gate to the adopted engine: the same tiny
Qwen2 weights loaded into the engine's MLX Dream models - both the reference
``dream_mlx`` model and the accelerated ``fast_dllm_mlx`` model - and into an
HF eager Qwen2 with causality neutralized must produce matching logits at
every position under mlx_dllm's bidirectional forward.

Dream is architecturally Qwen2 (q/k/v bias, GQA, RMSNorm, RoPE, SwiGLU) and
the engine's load() forces its Dream Model class onto any Qwen2-family
checkpoint, so stock tiny-Qwen2 weights exercise exactly the code path a real
Dream checkpoint does. Tenancy: this laptop runs the smallest compatible
checkpoint (hidden 8 / 2 layers, random weights, ~5 MB); full-size Dream-7B
validation is deferred to a mini.

Tolerances (fp32, measured 2026-07-11 on M-series; mlx 0.32.0 / mlx-lm 0.31.3
/ torch 2.13.0 / transformers 5.12.1; reference logit magnitude ~0.04):

- CPU: measured max|mlx - torch| = 2.9e-8. Gate 1e-6 (>30x margin).
- GPU: measured 1.9e-4 - Metal fp32 matmul accumulates ~1e-3 *relative*
  error (same phenomenon test_parity.py measures on GPT-2), and these logits
  are ~2000x smaller than distilgpt2's, hence the much smaller absolute
  noise. Gate 2e-3 (10x margin).
- A causality regression shifts pos-0 logits by 2.9e-2, >14x the GPU gate,
  so both bounds stay discriminative.
"""

import mlx.core as mx
import numpy as np
import pytest
import torch
from mlx.utils import tree_map

import mlx_dllm
from mlx_dllm import dream

MODEL = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
CPU_TOLERANCE = 1e-6
GPU_TOLERANCE = 2e-3
# The checkpoint's vocab_size is 151665, so Dream's real mask id (151666) is
# out of embedding range here; any in-range sentinel exercises the same
# denoise mechanics on this random-weight model.
MASK_TOKEN_ID = 151664


def _fp32(model):
    """Checkpoint ships bf16; cast so parity measures kernels, not rounding.

    (torch's ``from_pretrained(dtype=float32)`` upcasts bf16 exactly, so both
    sides see identical fp32 weights.)
    """
    model.update(tree_map(lambda a: a.astype(mx.float32), model.parameters()))
    model.eval()
    return model


@pytest.fixture(scope="module")
def naive():
    model, tokenizer = dream.load(MODEL, tokenizer_config={"trust_remote_code": False})
    return _fp32(model), tokenizer


@pytest.fixture(scope="module")
def fast():
    model, tokenizer = dream.load_fast(
        MODEL, tokenizer_config={"trust_remote_code": False}
    )
    return _fp32(model), tokenizer


@pytest.fixture(scope="module")
def ref_model():
    """HF eager Qwen2, fully bidirectional.

    transformers 5.12 Qwen2 bakes causality into the model-level
    ``create_causal_mask`` only (eager attention just adds the resulting
    mask; this config has no sliding-window layers), so one seam to
    neutralize.
    """
    from transformers import Qwen2ForCausalLM
    from transformers.models.qwen2 import modeling_qwen2

    model = Qwen2ForCausalLM.from_pretrained(
        MODEL, attn_implementation="eager", dtype=torch.float32
    ).eval()
    orig = modeling_qwen2.create_causal_mask
    modeling_qwen2.create_causal_mask = lambda *args, **kwargs: None
    try:
        yield model
    finally:
        modeling_qwen2.create_causal_mask = orig


def _probes(tokenizer, vocab_size):
    text = tokenizer.encode("The quick brown fox jumps over the lazy dog.")
    rand = np.random.default_rng(0).integers(0, vocab_size, size=64).tolist()
    return [text, rand]


def _assert_parity(loaded, ref_model, device, tolerance):
    model, tokenizer = loaded
    prev = mx.default_device()
    mx.set_default_device(device)
    try:
        for ids in _probes(tokenizer, ref_model.config.vocab_size):
            mlx_logits = np.array(
                mlx_dllm.bidirectional_forward(model, mx.array([ids]))
            )
            with torch.no_grad():
                ref_logits = ref_model(torch.tensor([ids])).logits.numpy()
            assert (
                mlx_logits.shape
                == ref_logits.shape
                == (1, len(ids), ref_model.config.vocab_size)
            )
            diff = float(np.abs(mlx_logits - ref_logits).max())
            assert diff < tolerance, f"max |mlx - torch| = {diff} >= {tolerance}"
    finally:
        mx.set_default_device(prev)


def test_dream_parity_cpu(naive, ref_model):
    _assert_parity(naive, ref_model, mx.cpu, CPU_TOLERANCE)


def test_fast_parity_cpu(fast, ref_model):
    _assert_parity(fast, ref_model, mx.cpu, CPU_TOLERANCE)


@pytest.mark.skipif(not mx.metal.is_available(), reason="no Metal GPU")
def test_dream_parity_gpu(naive, ref_model):
    _assert_parity(naive, ref_model, mx.gpu, GPU_TOLERANCE)


@pytest.mark.skipif(not mx.metal.is_available(), reason="no Metal GPU")
def test_fast_parity_gpu(fast, ref_model):
    _assert_parity(fast, ref_model, mx.gpu, GPU_TOLERANCE)


def test_dream_forward_is_noncausal(naive):
    """The engine's bare model() defaults to mlx-lm's causal mask; our
    bidirectional forward must actually change early-position logits, and the
    default path must be untouched after the scoped patch exits."""
    model, tokenizer = naive
    ids = mx.array([tokenizer.encode("The quick brown fox jumps over the lazy dog.")])
    bidir = np.array(mlx_dllm.bidirectional_forward(model, ids))
    causal = np.array(model(ids))
    assert float(np.abs(bidir[0, 0] - causal[0, 0]).max()) > 1e-2
    causal_again = np.array(model(ids))
    np.testing.assert_array_equal(causal, causal_again)


def test_diffusion_generate_smoke(naive):
    """Adoption milestone: a Qwen-family checkpoint denoises end-to-end
    through mlx_dllm.dream.generate (the engine's full iterative loop)."""
    model, tokenizer = naive
    text = dream.generate(
        model,
        tokenizer,
        "Hello",
        max_new_tokens=8,
        steps=4,
        temperature=0.0,
        mask_token_id=MASK_TOKEN_ID,
    )
    assert isinstance(text, str)


def test_fast_diffusion_generate_smoke(fast):
    """Same milestone through the accelerated Fast-dLLM path (dual KV cache,
    confident-parallel commit)."""
    model, tokenizer = fast
    text = dream.generate_fast(
        model,
        tokenizer,
        "Hello",
        max_new_tokens=8,
        steps=4,
        block_length=8,
        temperature=0.0,
        mask_token_id=MASK_TOKEN_ID,
    )
    assert isinstance(text, str)
