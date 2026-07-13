"""Gemma 3 (text) bidirectional parity and native diffusion smoke tests.

The fixture is a random-weight, six-layer ``Gemma3ForCausalLM`` checkpoint
(~100 KB) vendored under ``tests/fixtures/gemma3-a2d-sample`` - the ``model/``
subtree of the a2d-converted Gemma 3 sample. Unlike the qwen2/gemma1 fixtures
(pulled from the HF hub), this one is an a2d-specific checkpoint with no HF-hosted
equivalent, so it is committed here to keep the parity gate reproducible. Its
dims are tiny (hidden 16, head_dim 8, vocab 65, 6 layers, sliding_window 4,
sliding_window_pattern 3 -> globals at layers 2 & 5, locals at 0/1/3/4) and its
``config.json`` carries the spliced ``a2d`` block (mask_token_id 64). All dims
come from ``config.json``; nothing is hardcoded here.

The sample is ``final_alpha=0.5`` (partially bidirectionalized), so its outputs
are not meaningful text and are not compared against a2d's own inference; these
tests exercise loading, the full (alpha=1.0) bidirectional attention policy, and
the denoise loop's structure only.

Case A - full non-causal UNWINDOWED attention
---------------------------------------------
At a2d's production alpha=1.0 the sliding window is gone: every layer (local and
global) must attend over the whole sequence. mlx-lm's ``gemma3_text`` builds both
the global causal mask and the local sliding-window mask through the same
module-local ``create_attention_mask`` seam (the sliding one just adds a
``window_size=`` kwarg), so the runtime's existing ``_no_causal_mask`` rebind
neutralizes BOTH with no seam extension and no per-family logic. The reference
below therefore disables both of HF's mask builders (``create_causal_mask`` and
``create_sliding_window_causal_mask``) to compare against full attention, and a
dedicated test proves the sliding-window neutralization is load-bearing.

Tolerances (fp32, measured 2026-07-13 on x86_64 Linux CPU; mlx-lm 0.31.3 /
transformers 5.0.0; reference logits have magnitude ~0.2):

- CPU: measured worst max |MLX - torch| = 2.79e-7 across 32 random probes;
  gate = 1e-5 (~36x headroom). This is fp32 accumulation noise, NOT an
  activation gap: mlx-lm's ``gemma3_text`` MLP uses ``gelu_approx`` (the tanh
  gelu), which matches this checkpoint's ``gelu_pytorch_tanh`` - so gemma3 is
  fp32-limited (like qwen2's 2.65e-8) rather than gelu-limited (like gemma1's
  2.85e-5). The ~10x margin over qwen2 comes from 6 layers + qk-norm + four
  RMSNorms per layer accumulating more fp32 error.
- GPU: unmeasured here (CPU-only container, no Metal); gate = 2e-3 by analogy to
  qwen2/gemma1, leaving ample headroom for GPU fp32 matmul error.
- Causal-vs-bidirectional logits differ by ~0.15 and unwindowed-vs-windowed by
  ~0.30, so both gates stay strongly discriminative for an attention-policy
  regression.
"""

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
import torch
from mlx.utils import tree_map

import mlx_dllm

MODEL = str(Path(__file__).parent / "fixtures" / "gemma3-a2d-sample")
MASK_TOKEN_ID = 64  # from the checkpoint's a2d block
CPU_TOLERANCE = 1e-5
GPU_TOLERANCE = 2e-3


@pytest.fixture(scope="module")
def loaded():
    model, tokenizer, a2d = mlx_dllm.load(MODEL)
    model.update(tree_map(lambda value: value.astype(mx.float32), model.parameters()))
    model.eval()
    assert type(model).__module__ == "mlx_lm.models.gemma3_text"
    assert a2d is not None
    assert a2d.mask_token_id == MASK_TOKEN_ID
    return model, tokenizer


@pytest.fixture(scope="module")
def ref_model():
    """HF eager Gemma 3 with BOTH mask seams (causal + sliding) disabled.

    Gemma 3's text model builds a ``{"full_attention": create_causal_mask(...),
    "sliding_attention": create_sliding_window_causal_mask(...)}`` mapping and
    routes each layer to its entry. Returning ``None`` from both makes every
    layer run full, unwindowed attention - the alpha=1.0 policy mlx-dllm applies.
    """
    from transformers import Gemma3ForCausalLM
    from transformers.models.gemma3 import modeling_gemma3

    model = Gemma3ForCausalLM.from_pretrained(
        MODEL, attn_implementation="eager", dtype=torch.float32
    ).eval()
    original_causal = modeling_gemma3.create_causal_mask
    original_sliding = modeling_gemma3.create_sliding_window_causal_mask
    modeling_gemma3.create_causal_mask = lambda *args, **kwargs: None
    modeling_gemma3.create_sliding_window_causal_mask = lambda *args, **kwargs: None
    try:
        yield model
    finally:
        modeling_gemma3.create_causal_mask = original_causal
        modeling_gemma3.create_sliding_window_causal_mask = original_sliding


def _probes(vocab_size):
    """Random-token probes longer than the sliding window (=4).

    The fixture's tokenizer maps arbitrary text to token 0 (its 64-entry vocab
    has no byte fallback), so a sentence probe would be degenerate; random tokens
    exercise every embedding row. Lengths exceed the sliding window so the local
    (windowed) layers genuinely differ from full attention.
    """
    rng = np.random.default_rng(0)
    return [rng.integers(0, vocab_size, size=length).tolist() for length in (8, 12, 16)]


def _assert_parity(loaded, ref_model, device, tolerance):
    model, _ = loaded
    previous_device = mx.default_device()
    mx.set_default_device(device)
    try:
        for ids in _probes(ref_model.config.vocab_size):
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


def test_gemma3_bidirectional_parity_cpu(loaded, ref_model):
    _assert_parity(loaded, ref_model, mx.cpu, CPU_TOLERANCE)


@pytest.mark.skipif(not mx.metal.is_available(), reason="no Metal GPU")
def test_gemma3_bidirectional_parity_gpu(loaded, ref_model):
    _assert_parity(loaded, ref_model, mx.gpu, GPU_TOLERANCE)


def test_gemma3_forward_is_noncausal(loaded):
    """The seam changes attention (bidirectional != causal) and is restored."""
    model, _ = loaded
    ids = mx.array([_probes(model.args.vocab_size)[0]])
    bidirectional = np.array(mlx_dllm.bidirectional_forward(model, ids))
    causal = np.array(model(ids))
    assert float(np.abs(bidirectional[0, 0] - causal[0, 0]).max()) > 1e-2
    # The scoped rebind is undone: a plain forward is causal again, unchanged.
    np.testing.assert_array_equal(causal, np.array(model(ids)))


def test_gemma3_sliding_window_is_neutralized(loaded):
    """Neutralizing the sliding-window mask (not just the causal one) matters.

    Compare full attention (both masks -> None, what the runtime does) against a
    variant that neutralizes only the global causal mask while keeping the local
    sliding-window mask. A clear gap proves the local layers really are windowed
    under stock mlx-lm and that the seam's coverage of the sliding mask is
    load-bearing for Case A - not incidental.
    """
    model, _ = loaded
    ids = mx.array([_probes(model.args.vocab_size)[0]])  # length 8 > window 4
    full = np.array(mlx_dllm.bidirectional_forward(model, ids))

    module = sys.modules[type(model).__module__]
    original = module.create_attention_mask

    def keep_sliding_window(h, cache=None, window_size=None, return_array=False):
        # Global (causal) calls pass window_size=None -> drop them; keep the
        # local sliding-window construction exactly as stock mlx-lm builds it.
        if window_size is None:
            return None
        return original(h, cache, window_size=window_size, return_array=return_array)

    module.create_attention_mask = keep_sliding_window
    try:
        windowed = np.array(model(ids))
    finally:
        module.create_attention_mask = original
    assert float(np.abs(full - windowed).max()) > 1e-2


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
    assert text == tokenizer.decode(output[0].tolist()[len(prompt_ids) :])
