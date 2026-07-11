"""Qwen/Dream-family diffusion inference via the adopted Fast-dLLM-mlx engine.

Fast-dLLM-mlx (MacPaw, Apache-2.0) is a pinned git dependency, never vendored;
this module is a thin entry over its two paths:

- ``dream_mlx``: the reference full-canvas iterative denoise loop.
- ``fast_dllm_mlx``: Fast-dLLM accelerated decoding (dual KV cache,
  confident-parallel token commit).

One decode-time correction: ``dream_mlx``'s denoise loop calls its model with
no mask, which mlx-lm defaults to CAUSAL attention for multi-token input.
Dream decodes bidirectionally - upstream's own PyTorch reference
(``dream_mlx/generate_diffusion_torch.py``) and its ``fast_dllm_mlx`` path
both run ``attention_mask="full"``. :func:`generate` therefore wraps the loop
in the same scoped no-causal-mask rebind PR #1 uses for GPT-2 (bidirectional
attention as a decode-time policy this runtime supplies).
"""

from __future__ import annotations

import dream_mlx
import fast_dllm_mlx

from mlx_dllm.runtime import _no_causal_mask


def load(path_or_repo: str, **kwargs):
    """``(model, tokenizer)`` via the reference ``dream_mlx`` engine."""
    return dream_mlx.load(path_or_repo, **kwargs)


def load_fast(path_or_repo: str, **kwargs):
    """``(model, tokenizer)`` via the accelerated ``fast_dllm_mlx`` engine."""
    return fast_dllm_mlx.load(path_or_repo, **kwargs)


def generate(model, tokenizer, prompt, live_preview: bool = False, **kwargs) -> str:
    """Reference full-canvas diffusion generation, forced bidirectional."""
    with _no_causal_mask(model):
        return dream_mlx.diffusion_generate(
            model, tokenizer, prompt, live_preview=live_preview, **kwargs
        )


def generate_fast(model, tokenizer, prompt, **kwargs) -> str:
    """Fast-dLLM accelerated generation (already runs ``mask="full"``)."""
    return fast_dllm_mlx.diffusion_generate(model, tokenizer, prompt, **kwargs)
