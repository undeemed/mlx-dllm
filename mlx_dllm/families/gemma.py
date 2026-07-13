"""Gemma (v1) family adapter: loads through stock mlx-lm unchanged (no-op).

Stock HF Gemma-1 safetensors already match the layout mlx-lm's ``gemma`` module
expects (standard ``model.*`` keys), and that module handles Gemma's two
family-specific details internally: the ``sqrt(hidden_size)`` embedding scale and
the lm_head tied to ``embed_tokens``. So this family supplies no
``sanitize_wrapper`` - it is registered explicitly, like ``qwen2``, only to
document the "nothing special needed" case through the same mechanism as gpt2's
wrapper.

Scope: Gemma v1 only (``model_type == "gemma"``). Gemma 3 (text) has its own
no-op adapter (``gemma3.py``, ``model_type == "gemma3_text"``). gemma2 remains
out of scope; its ``model_type`` is not "gemma", so it simply falls through to
stock mlx-lm here and is not bidirectionalized by this adapter.

Bidirectional attention needs no weight edit: mlx-lm's ``gemma`` module builds
its mask through the module-local ``create_attention_mask`` seam (same as gpt2
and qwen2), which ``runtime._no_causal_mask`` rebinds at decode time.
"""

from __future__ import annotations

from mlx_dllm.families import FamilyAdapter, register

register(FamilyAdapter(model_type="gemma", sanitize_wrapper=None))
