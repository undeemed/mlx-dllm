# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

## What this is

MLX inference runtime for a2d masked-diffusion (MDLM) GPT-2 checkpoints and a
native Qwen/Dream-family reference decoder.
mlx-lm is consumed as a library, never forked; the only behavioral change is decode-time bidirectional attention (a2d's alpha=1 policy), applied via a scoped rebind of `create_attention_mask` in the model's own mlx-lm module (`mlx_dllm/runtime.py`).
The Qwen path reuses mlx-lm's stock `qwen2` model and adds a fixed-canvas,
confidence-ranked iterative denoise loop. CLI, acceleration, and the Qwen a2d
format bridge remain follow-on PRs.

## Build / test

- `uv venv && uv pip install -e ".[test]"` then `pytest` (needs network once to fetch distilgpt2 and the tiny Qwen2 fixture; ~5s after caching).
- Use `distilbert/distilgpt2` (canonical id) in code and tests - the bare `distilgpt2` alias 404s on the HF xet download path.
- **Linux is CPU-only and needs the right mlx backend.** The `mlx` wheel ships only the Python bindings; its bundled `libmlx.so` is a Metal stub that fails to import on Linux (`libmlx.so: cannot open shared object file`), and mlx-lm depends on `mlx` only under Darwin. `pyproject.toml` therefore pulls `mlx[cpu]` on Linux (the `cpu` extra adds the ABI-matched `mlx-cpu` backend) and plain `mlx` elsewhere (Darwin auto-adds `mlx-metal`). Bare `mlx-cpu` is NOT enough - it has the backend lib but no `mlx.core` bindings. On Linux the two Metal-gated parity tests skip (`mx.metal.is_available()` is False); CPU parity gates still run and must stay green.

## Sharp edges (all bitten once)

- **Dependency pins are load-bearing.** mlx-lm 0.31.x declares `transformers>=5` but crashes on import with transformers >=5.13 (`AutoTokenizer.register` signature change). Runtime pins `transformers>=5.0,<5.13`; tests pin `~=5.0.0` because the parity reference patches transformers-5.0-verified seams. mlx-lm itself is pinned `>=0.31,<0.32` because runtime.py reaches into private internals (`_download`, `_get_classes`) and the gpt2 module's `create_attention_mask` binding.
- **Weight-key normalization is required.** Modern `save_pretrained` output (what a2d writes; also distilgpt2) prefixes keys with `transformer.` and omits tied `lm_head.weight`; mlx-lm's gpt2 module only accepts the unprefixed legacy layout of `openai-community/gpt2`. The gpt2 family adapter (`mlx_dllm/families/gpt2.py`) wraps `sanitize` to strip the prefix and reject a genuinely untied `lm_head.weight` (mlx-lm's gpt2 always ties the head to `wte`, so loading one would silently produce wrong logits). `Model.__module__` must point at `mlx_lm.models.gpt2` or the mask patch resolves the wrong module - `runtime._model_classes` pins this for every wrapped family, so adapters never manage it. See "Family adapters" below.
- **Metal fp32 is not CPU fp32.** MLX GPU matmul accumulates at ~1e-3 relative error (a single 768-dim-reduction matmul shows 0.1 max-abs vs f64; whole distilgpt2 forward 0.073 on logits of ~85). MLX CPU matches torch-f32 at 8e-5. Hence the GPT-2 parity gate (tests/test_parity.py) asserts 1e-3 on CPU and 0.5 on GPU; a causality regression shows up as ~10, so both bounds stay discriminative. Don't "fix" a GPU parity failure by loosening past ~0.5 - investigate instead.
- **transformers 5.x GPT-2 has TWO causality seams**: the per-layer `bias` buffer inside eager attention AND a model-level `create_causal_mask`. A bidirectional torch reference must neutralize both (see `ref_model` fixture).
- **transformers 5.0 Qwen2 has one causality seam for eager attention:** the model-level `create_causal_mask`. mlx-lm Qwen2 uses the same module-local `create_attention_mask` seam as GPT-2, so `_no_causal_mask` works for both without wrapping or reimplementing the transformer.
- **The native Qwen parity fixture is intentionally tiny.** `trl-internal-testing/tiny-Qwen2ForCausalLM-2.5` is a two-layer, hidden-size-8 Qwen2 model (~5 MB). Its weights are bf16; cast both MLX and HF loads to fp32 before measuring framework parity. Its parity gate (tests/test_qwen.py) asserts 1e-6 on CPU and 2e-3 on GPU; measured values live in that file's docstring. Full-size Dream runs belong on a mini.

## Contract notes

- Bidirectional attention is a decode-time policy, NOT in a2d weights/config; a stock causal loader runs a2d checkpoints causally and produces garbage for diffusion. The `a2d` block in config.json (objective / mask_token_id / final_alpha / sampler) is the only marker; `runtime._parse_a2d` tolerates future extra keys.
- Diffusion decode never uses a KV cache (`cache=None`); mlx-lm samplers/tokenizer remain reusable for the future denoise loop.
- The Qwen reference denoiser is greedy, batch-one, fixed-canvas, confidence-ranked, and never remasks. It suppresses the mask token from output logits so every scheduled reveal makes progress. Schedule steps that reveal nothing skip the forward pass entirely. Dual-cache/confident-parallel acceleration and a2d sampler/config integration are intentionally absent.
- **Prediction-position convention is in-place by default** (token for position i read from logits at i) because this runtime targets a2d-converted checkpoints and a2d conversion drops the autoregressive next-token shift.
Published Dream-family checkpoints keep the next-token head (token for position i comes from logits at i-1); `denoise`/`generate` take `logit_shift=True` for those, which requires an unmasked first canvas position.
Do not feed a published Dream checkpoint through the default in-place path - it will silently garble output.
- `generate` returns only the decoded continuation; the prompt text is never included.

## Family adapters (adding a model family)

New model families plug in through the `mlx_dllm/families/` registry, not by editing shared dispatch. `runtime._model_classes` resolves mlx-lm's stock `(Model, ModelArgs)` and then applies whatever adapter `families.get_adapter(model_type)` returns; the bidirectional mask seam (`_no_causal_mask`) and the denoise loop (`diffusion.py`) stay family-agnostic and must never grow per-family conditionals.

To add a family, create one module `mlx_dllm/families/<model_type>.py` that builds a `FamilyAdapter` and calls `register(...)` at import time - nothing else needs editing (the package auto-imports its submodules, and `_model_classes`'s dispatch body does not change). An adapter supplies:

- `model_type` - the `config["model_type"]` string it handles.
- `sanitize_wrapper` - OPTIONAL factory `(model_class) -> subclass` that overrides `sanitize` when the family's on-disk weight layout differs from what its mlx-lm module expects (gpt2 needs one; qwen2 registers `None`). A family that loads cleanly through stock mlx-lm registers `sanitize_wrapper=None`, or need not register at all - an unregistered `model_type` also falls through to stock mlx-lm.

The `Model.__module__` pin the mask seam depends on is applied by `_model_classes` for any wrapped class, so adapters never manage it. See `mlx_dllm/families/__init__.py` (contract + worked example), `families/gpt2.py` (wrapper), and `families/qwen2.py` (no-op). OUT OF SCOPE for the registry: the denoise loop, acceleration, and the a2d-format bridge.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
