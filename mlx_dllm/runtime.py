"""Load GPT-2/Qwen2/Gemma checkpoints and run bidirectional forwards on MLX.

a2d checkpoints are standard HF triples (config.json + tokenizer +
model.safetensors) whose only non-standard bytes are an "a2d" block inside
config.json. Bidirectional attention is NOT in the weights or config - it is a
decode-time policy this runtime supplies (a2d's alpha=1 configuration).
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

# _download/_get_classes are the guts of mlx_lm.utils.load, which offers no
# public seam to hook sanitize; we mirror its 4-line body instead (mlx-lm
# pinned >=0.31,<0.32 in pyproject).
from mlx_lm.utils import _download, _get_classes, load_model, load_tokenizer

from mlx_dllm.families import get_adapter


@dataclass(frozen=True)
class A2DConfig:
    """The "a2d" block a2d splices into config.json."""

    objective: str  # "mdlm"
    mask_token_id: int
    final_alpha: float = 1.0  # provenance; decode always runs alpha=1
    sampler: Optional[dict] = None  # canvas_len / num_steps / temperature


def _parse_a2d(config: dict) -> Optional[A2DConfig]:
    block = config.get("a2d")
    if block is None:
        return None
    # Tolerate future extra keys the same way mlx-lm's ModelArgs.from_dict does.
    return A2DConfig(
        objective=block["objective"],
        mask_token_id=block["mask_token_id"],
        final_alpha=block.get("final_alpha", 1.0),
        sampler=block.get("sampler"),
    )


def _model_classes(config: dict):
    """Resolve mlx-lm's (Model, ModelArgs) classes, applying a family adapter.

    Dispatch is family-agnostic: the per-``model_type`` policy lives in the
    :mod:`mlx_dllm.families` registry, so adding a family is a new adapter
    registration, never an edit here. An adapter's ``sanitize_wrapper`` (when
    present) subclasses the stock model to normalize weight keys; families
    without one (or unregistered ``model_type``s) load through stock mlx-lm.
    """
    model_class, args_class = _get_classes(config)
    adapter = get_adapter(config.get("model_type"))
    if adapter is None or adapter.sanitize_wrapper is None:
        return model_class, args_class

    wrapped = adapter.sanitize_wrapper(model_class)
    # Keep the wrapped class in the stock mlx-lm model module so the
    # _no_causal_mask seam (which rebinds create_attention_mask in
    # sys.modules[type(model).__module__]) resolves the family's own module.
    wrapped.__module__ = model_class.__module__
    return wrapped, args_class


def load(path_or_repo: str):
    """Load an HF-layout GPT-2/Qwen2/Gemma checkpoint through stock mlx-lm.

    Returns ``(model, tokenizer, a2d)`` where ``a2d`` is the parsed
    :class:`A2DConfig` when the checkpoint carries a2d's config block, else
    ``None``. mlx-lm's config parsing ignores the extra "a2d" key, so both
    a2d-converted GPT-2 and stock GPT-2/Qwen2/Gemma checkpoints load as-is.
    Qwen2 and Gemma (v1) resolve directly to mlx-lm's own ``qwen2.Model``
    and ``gemma.Model``; no transformer code is reimplemented here.
    """
    model_path = _download(path_or_repo)
    model, config = load_model(model_path, get_model_classes=_model_classes)
    model.eval()
    tokenizer = load_tokenizer(model_path, eos_token_ids=config.get("eos_token_id"))
    return model, tokenizer, _parse_a2d(config)


@contextmanager
def _no_causal_mask(model: nn.Module):
    """Scoped monkeypatch: the model's mlx-lm module builds no attention mask.

    mlx-lm model ``__call__``s hardcode ``mask = create_attention_mask(...)``
    (-> "causal"); rebinding that name in the model's own module for the
    duration of the call makes SDPA run with ``mask=None``, i.e. fully
    bidirectional. Mirrors a2d's decode-time patch of HF's eager seam
    (a2d transform/attention.py).
    """
    mod = sys.modules[type(model).__module__]
    orig = mod.create_attention_mask
    mod.create_attention_mask = lambda *args, **kwargs: None
    try:
        yield
    finally:
        mod.create_attention_mask = orig


def bidirectional_forward(model: nn.Module, input_ids: mx.array) -> mx.array:
    """Full non-causal forward pass: logits for ALL positions, no KV cache.

    This is a2d's decode configuration (alpha=1, use_cache=False): diffusion
    decoding recomputes the whole canvas each step.
    """
    # ponytail: process-global rebind during the call; fine single-threaded,
    # revisit if forwards ever run concurrently.
    with _no_causal_mask(model):
        return model(input_ids, cache=None)
