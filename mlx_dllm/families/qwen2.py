"""Qwen2 family adapter: loads through stock mlx-lm unchanged (no-op).

Qwen2 checkpoints already match the layout mlx-lm's ``qwen2`` module expects, so
this family supplies no ``sanitize_wrapper``. It is registered explicitly - even
though an unregistered ``model_type`` would also fall through to stock mlx-lm -
so the extension point documents the "nothing special needed" case alongside
gpt2's wrapper, and so the two currently-supported families flow through one
mechanism.
"""

from __future__ import annotations

from mlx_dllm.families import FamilyAdapter, register

register(FamilyAdapter(model_type="qwen2", sanitize_wrapper=None))
