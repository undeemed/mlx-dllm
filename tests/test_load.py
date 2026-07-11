"""Loader contract: a2d checkpoints (HF triple + "a2d" config block) load as-is."""

import json
import shutil

import mlx.core as mx
import pytest

import mlx_dllm
from mlx_dllm.runtime import _download

MODEL = "distilbert/distilgpt2"

A2D_BLOCK = {
    "objective": "mdlm",
    "mask_token_id": 50257,
    "final_alpha": 1.0,
    "sampler": {"canvas_len": 128, "num_steps": 128, "temperature": 1.0},
}


@pytest.fixture()
def a2d_checkpoint(tmp_path):
    """Stock snapshot with a2d's config block spliced in (weights symlinked)."""
    src = _download(MODEL)
    for f in src.iterdir():
        if f.name == "config.json":
            config = json.loads(f.read_text())
            config["a2d"] = A2D_BLOCK
            (tmp_path / f.name).write_text(json.dumps(config))
        elif f.is_file():
            (tmp_path / f.name).symlink_to(f)
    return tmp_path


def test_load_stock_checkpoint_has_no_a2d_block():
    _, _, a2d = mlx_dllm.load(MODEL)
    assert a2d is None


def test_load_a2d_checkpoint(a2d_checkpoint):
    model, tokenizer, a2d = mlx_dllm.load(str(a2d_checkpoint))
    assert a2d == mlx_dllm.A2DConfig(
        objective="mdlm",
        mask_token_id=50257,
        final_alpha=1.0,
        sampler={"canvas_len": 128, "num_steps": 128, "temperature": 1.0},
    )
    # the extra config key is ignored by mlx-lm and the model still runs
    logits = mlx_dllm.bidirectional_forward(model, mx.array([tokenizer.encode("hi there")]))
    assert logits.shape[1] == 2


def test_parse_tolerates_future_keys():
    from mlx_dllm.runtime import _parse_a2d

    a2d = _parse_a2d({"a2d": {**A2D_BLOCK, "block_size": 16}})
    assert a2d.objective == "mdlm"
    assert a2d.mask_token_id == 50257
