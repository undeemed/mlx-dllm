"""Gemma (v1) bidirectional parity and native diffusion smoke tests.

The checkpoint is a random-weight, two-layer Gemma fixture (~10 MB), not a
full-size Gemma model. Its architecture (model_type="gemma", GemmaForCausalLM,
4 heads / 2 KV heads GQA, tied lm_head) exercises the same mlx-lm Gemma path
without putting heavy-model inference on a developer laptop. Scope is Gemma v1
only; gemma2/gemma3 (sliding-window attention) are out of scope.

Tolerances (fp32, measured 2026-07-13 on aarch64 Linux CPU; mlx 0.32.0 /
mlx-lm 0.31.3 / torch 2.13.0 / transformers 5.0.0; reference logits have
magnitude ~0.03):

- CPU: measured max |MLX - torch| = 2.85e-5 (random-token probe; the sentence
  probe measures 4.5e-6); gate = 1e-4. This is ~1000x looser than qwen2's gate
  and is NOT fp32 noise: it is a deliberate activation-function difference.
  mlx-lm's gemma MLP hardcodes the exact (erf) ``nn.gelu``, but this checkpoint's
  config sets ``hidden_act="gelu_pytorch_tanh"`` (the tanh approximation), which
  HF-eager honors. Forcing the HF reference to exact gelu collapses the diff to
  7.8e-8, confirming gelu is the sole driver. mlx-lm is consumed as a library
  and never forked, so the runtime inherits its exact-gelu choice; for greedy
  argmax denoise this ~3e-5 logit difference is immaterial.
- GPU: unmeasured here (this container is CPU-only, no Metal); gate = 2e-3 by
  analogy to qwen2 (which measured 7.31e-5 on GPU), leaving headroom for the
  ~2.85e-5 gelu offset on top of GPU fp32 matmul error.
- Causal vs bidirectional position-zero logits differ by 2.9e-2, so both gates
  remain discriminative for the attention-policy regression.
"""

import mlx.core as mx
import numpy as np
import pytest
import torch
from mlx.utils import tree_map

import mlx_dllm

MODEL = "trl-internal-testing/tiny-GemmaForCausalLM"
MASK_TOKEN_ID = 4  # Gemma's <mask> reserved token
CPU_TOLERANCE = 1e-4
GPU_TOLERANCE = 2e-3


@pytest.fixture(scope="module")
def loaded():
    model, tokenizer, a2d = mlx_dllm.load(MODEL)
    model.update(tree_map(lambda value: value.astype(mx.float32), model.parameters()))
    model.eval()
    assert type(model).__module__ == "mlx_lm.models.gemma"
    assert a2d is None
    return model, tokenizer


@pytest.fixture(scope="module")
def ref_model():
    """HF eager Gemma with its model-level causal-mask seam disabled."""
    from transformers import GemmaForCausalLM
    from transformers.models.gemma import modeling_gemma

    model = GemmaForCausalLM.from_pretrained(
        MODEL, attn_implementation="eager", dtype=torch.float32
    ).eval()
    original = modeling_gemma.create_causal_mask
    modeling_gemma.create_causal_mask = lambda *args, **kwargs: None
    try:
        yield model
    finally:
        modeling_gemma.create_causal_mask = original


def _assert_parity(loaded, ref_model, device, tolerance):
    model, tokenizer = loaded
    probes = [
        tokenizer.encode("The quick brown fox jumps over the lazy dog."),
        np.random.default_rng(0)
        .integers(0, ref_model.config.vocab_size, size=64)
        .tolist(),
    ]
    previous_device = mx.default_device()
    mx.set_default_device(device)
    try:
        for ids in probes:
            mlx_logits = np.array(
                mlx_dllm.bidirectional_forward(model, mx.array([ids]))
            )
            with torch.no_grad():
                torch_logits = ref_model(torch.tensor([ids])).logits.numpy()
            assert mlx_logits.shape == torch_logits.shape
            difference = float(np.abs(mlx_logits - torch_logits).max())
            assert difference < tolerance, difference
    finally:
        mx.set_default_device(previous_device)


def test_gemma_bidirectional_parity_cpu(loaded, ref_model):
    _assert_parity(loaded, ref_model, mx.cpu, CPU_TOLERANCE)


@pytest.mark.skipif(not mx.metal.is_available(), reason="no Metal GPU")
def test_gemma_bidirectional_parity_gpu(loaded, ref_model):
    _assert_parity(loaded, ref_model, mx.gpu, GPU_TOLERANCE)


def test_gemma_forward_is_noncausal(loaded):
    model, tokenizer = loaded
    ids = mx.array([tokenizer.encode("The quick brown fox jumps over the lazy dog.")])
    bidirectional = np.array(mlx_dllm.bidirectional_forward(model, ids))
    causal = np.array(model(ids))
    assert float(np.abs(bidirectional[0, 0] - causal[0, 0]).max()) > 1e-2
    np.testing.assert_array_equal(causal, np.array(model(ids)))


def test_native_denoise_smoke(loaded):
    model, tokenizer = loaded
    prompt_ids = tokenizer.encode("Hello")
    canvas = mx.array([[*prompt_ids, *([MASK_TOKEN_ID] * 8)]])
    output = mlx_dllm.denoise(model, canvas, mask_token_id=MASK_TOKEN_ID, steps=4)
    assert output.shape == canvas.shape
    assert output[0, : len(prompt_ids)].tolist() == prompt_ids
    assert MASK_TOKEN_ID not in output[0].tolist()

    text = mlx_dllm.generate(
        model,
        tokenizer,
        "Hello",
        max_new_tokens=8,
        steps=4,
        mask_token_id=MASK_TOKEN_ID,
    )
    assert isinstance(text, str)
    assert text
    assert text == tokenizer.decode(output[0].tolist()[len(prompt_ids) :])
