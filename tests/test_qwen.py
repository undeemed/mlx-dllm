"""Qwen2 bidirectional parity and native diffusion smoke tests.

The checkpoint is a random-weight, two-layer Qwen2 fixture (~5 MB), not a
full-size Dream model. Its architecture exercises the same mlx-lm Qwen2 path
without putting heavy-model inference on a developer laptop.

Tolerances (fp32, measured 2026-07-11 on M-series; mlx 0.32.0 / mlx-lm 0.31.3
/ torch 2.13.0 / transformers 5.0.0; reference logits have magnitude ~0.04):

- CPU: measured max |MLX - torch| = 2.65e-8; gate = 1e-6.
- GPU: measured max |MLX - torch| = 7.31e-5; gate = 2e-3.
- Causal vs bidirectional position-zero logits differ by 2.85e-2, so both
  tolerances remain discriminative for the attention-policy regression.
"""

import mlx.core as mx
import numpy as np
import pytest
import torch
from mlx.utils import tree_map

import mlx_dllm

MODEL = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
MASK_TOKEN_ID = 151664
CPU_TOLERANCE = 1e-6
GPU_TOLERANCE = 2e-3


@pytest.fixture(scope="module")
def loaded():
    model, tokenizer, a2d = mlx_dllm.load(MODEL)
    model.update(tree_map(lambda value: value.astype(mx.float32), model.parameters()))
    model.eval()
    assert type(model).__module__ == "mlx_lm.models.qwen2"
    assert a2d is None
    return model, tokenizer


@pytest.fixture(scope="module")
def ref_model():
    """HF eager Qwen2 with its model-level causal-mask seam disabled."""
    from transformers import Qwen2ForCausalLM
    from transformers.models.qwen2 import modeling_qwen2

    model = Qwen2ForCausalLM.from_pretrained(
        MODEL, attn_implementation="eager", dtype=torch.float32
    ).eval()
    original = modeling_qwen2.create_causal_mask
    modeling_qwen2.create_causal_mask = lambda *args, **kwargs: None
    try:
        yield model
    finally:
        modeling_qwen2.create_causal_mask = original


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


def test_qwen_bidirectional_parity_cpu(loaded, ref_model):
    _assert_parity(loaded, ref_model, mx.cpu, CPU_TOLERANCE)


@pytest.mark.skipif(not mx.metal.is_available(), reason="no Metal GPU")
def test_qwen_bidirectional_parity_gpu(loaded, ref_model):
    _assert_parity(loaded, ref_model, mx.gpu, GPU_TOLERANCE)


def test_qwen_forward_is_noncausal(loaded):
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


class _FakeArgs:
    vocab_size = 6


class _FakeModel:
    args = _FakeArgs()


def _fake_forward(calls):
    def forward(model, canvas):
        calls.append(np.array(canvas))
        logits = np.zeros((*canvas.shape, model.args.vocab_size), dtype=np.float32)
        for position in range(canvas.shape[1]):
            logits[0, position, (position % 4) + 1] = position + 1
        return mx.array(logits)

    return forward


def test_denoise_reveals_highest_confidence_and_freezes(monkeypatch):
    from mlx_dllm import diffusion

    calls = []
    monkeypatch.setattr(diffusion, "bidirectional_forward", _fake_forward(calls))
    output = diffusion.denoise(
        _FakeModel(), mx.array([[0, 5, 5, 5, 5]]), mask_token_id=5, steps=2
    )

    assert len(calls) == 2
    assert calls[1].tolist() == [[0, 5, 5, 4, 1]]
    assert output.tolist() == [[0, 2, 3, 4, 1]]


def test_denoise_skips_forward_on_zero_reveal_steps(monkeypatch):
    from mlx_dllm import diffusion

    calls = []
    monkeypatch.setattr(diffusion, "bidirectional_forward", _fake_forward(calls))
    output = diffusion.denoise(
        _FakeModel(), mx.array([[0, 5, 5, 5, 5]]), mask_token_id=5, steps=8
    )

    assert len(calls) == 4
    assert 5 not in output[0].tolist()


def test_denoise_logit_shift_reads_previous_position(monkeypatch):
    from mlx_dllm import diffusion

    def fake_forward(model, canvas):
        logits = np.zeros((*canvas.shape, model.args.vocab_size), dtype=np.float32)
        for position in range(canvas.shape[1]):
            logits[0, position, position + 1] = 3.0
        return mx.array(logits)

    monkeypatch.setattr(diffusion, "bidirectional_forward", fake_forward)
    output = diffusion.denoise(
        _FakeModel(), mx.array([[0, 5, 5]]), mask_token_id=5, steps=1, logit_shift=True
    )
    assert output.tolist() == [[0, 1, 2]]

    with pytest.raises(ValueError, match="unmasked first"):
        diffusion.denoise(
            _FakeModel(), mx.array([[5, 0]]), mask_token_id=5, steps=1, logit_shift=True
        )
