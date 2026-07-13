"""Gemma 3 (text) family adapter: loads through stock mlx-lm unchanged (no-op).

Scope: Gemma 3 text only (``model_type == "gemma3_text"``), the a2d-conversion
target for Gemma 3 270M. gemma2 and other families are out of scope here.

Weight layout
-------------
a2d-converted Gemma 3 checkpoints already match the layout mlx-lm's
``gemma3_text`` module expects: standard ``model.*`` keys, tied embeddings (no
``lm_head.weight``; mlx-lm's own ``sanitize`` sets ``tie_word_embeddings`` and
loads the head from ``embed_tokens``). So this family supplies no
``sanitize_wrapper`` and registers ``None`` - the same "nothing special needed"
case as ``qwen2`` and ``gemma`` (v1), routed through the same mechanism.

Full non-causal UNWINDOWED attention (a2d's alpha=1 policy)
----------------------------------------------------------
Gemma 3 interleaves *local* (sliding-window) and *global* (full) attention
layers. At a2d's production ``final_alpha=1.0`` the sliding window is gone: every
layer, local and global, must attend over the whole sequence. Reaching that
needs BOTH the causal mask and the sliding-window mask neutralized on every
layer - not just the causal mask that gemma1 removes.

The existing generic seam already does exactly this, so gemma3 needs no seam
extension and no per-family branch. mlx-lm's ``gemma3_text`` builds *both* masks
through the same module-local ``create_attention_mask`` name - the global mask as
``create_attention_mask(h, cache)`` and the sliding mask as the same call with a
``window_size=`` kwarg. ``runtime._no_causal_mask`` rebinds that one name to
``lambda *args, **kwargs: None`` for the duration of the forward pass, so *both*
constructions return ``None`` and every layer runs SDPA with ``mask=None``, i.e.
fully bidirectional and unwindowed. Verified on the tiny fixture: causal-vs-
bidirectional logits differ by ~0.15 and unwindowed-vs-windowed by ~0.30, while
CPU parity vs HF full-attention holds at ~3e-7 (see ``tests/test_gemma3.py``).

This generic coverage depends on mlx-lm keeping both masks on that single seam;
the runtime's ``mlx-lm>=0.31,<0.32`` pin bounds that assumption. gemma3 also uses
``gelu_approx`` (the tanh gelu) in its MLP, which matches published Gemma 3
configs' ``gelu_pytorch_tanh``, so - unlike gemma1 - parity is fp32-limited, not
activation-limited.
"""

from __future__ import annotations

from mlx_dllm.families import FamilyAdapter, register

register(FamilyAdapter(model_type="gemma3_text", sanitize_wrapper=None))
