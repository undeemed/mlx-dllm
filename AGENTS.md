# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

## What this is

MLX inference runtime for a2d masked-diffusion checkpoints, with two engine paths:

- **GPT-2 / MDLM (a2d's current output)**: ours (`mlx_dllm/runtime.py`). mlx-lm is consumed as a library, never forked; the only behavioral change is decode-time bidirectional attention (a2d's alpha=1 policy), applied via a scoped rebind of `create_attention_mask` in the model's own mlx-lm module.
- **Qwen / Dream family (a2d's future output)**: MacPaw's Fast-dLLM-mlx (Apache-2.0), consumed as a pinned git dependency, never vendored. `mlx_dllm/dream.py` is the thin entry over its `dream_mlx` (reference full-canvas denoise) and `fast_dllm_mlx` (dual-KV-cache, confident-parallel decode) paths.

The a2d-format bridge for the Qwen path (a2d-config parsing + MDLM sampler) is follow-on work (PR #3), as is the GPT-2 denoise loop and CLI.

## Build / test

- `uv venv --python 3.13 && uv pip install -e ".[test]"` then `pytest` (needs network once to fetch distilgpt2 and tiny-Qwen2; a few seconds after caching).
- Use `distilbert/distilgpt2` (canonical id) in code and tests - the bare `distilgpt2` alias 404s on the HF xet download path.
- Tenancy: only tiny models on laptops (tests use `trl-internal-testing/tiny-Qwen2ForCausalLM-2.5`, hidden 8 / 2 layers). Dream-7B and benchmarks run on a mini.

## Sharp edges (all bitten once)

- **Dependency pins are load-bearing.** mlx-lm 0.31.x declares `transformers>=5` but crashes on import with transformers >=5.13 (`AutoTokenizer.register` signature change); fast-dllm-mlx raises the floor to >=5.5.4, so both runtime and tests pin `>=5.5.4,<5.13`. mlx-lm itself is pinned `>=0.31,<0.32` because runtime.py reaches into private internals (`_download`, `_get_classes`) and the gpt2 module's `create_attention_mask` binding. `requires-python>=3.13` is inherited from fast-dllm-mlx's metadata - pip refuses to resolve it on older interpreters.
- **Upstream Fast-dLLM-mlx does not pip-install.** MacPaw/Fast-dLLM-mlx has no `[build-system]` table, and the setuptools legacy fallback errors on its multiple flat-layout top-level packages. The dependency therefore pins a commit on the `pip-installable` branch of the `undeemed/Fast-dLLM-mlx` fork: one packaging-only commit (build-system + explicit package list + benchmark-only deps moved to an extra) on top of the recorded upstream commit. To bump: rebase that branch onto new upstream, re-pin.
- **`dream_mlx`'s naive denoise loop is causal by default.** Its model forwards `mask=None` into mlx-lm's `create_attention_mask`, which returns "causal" for multi-token input - contradicting Dream semantics, upstream's own torch reference (`attention_mask="full"`), and upstream's `fast_dllm_mlx` path (`mask="full"`). `mlx_dllm.dream.generate` supplies bidirectionality via the same scoped rebind runtime.py uses; don't call `dream_mlx.diffusion_generate` directly.
- **Weight-key normalization is required.** Modern `save_pretrained` output (what a2d writes; also distilgpt2) prefixes keys with `transformer.` and omits tied `lm_head.weight`; mlx-lm's gpt2 module only accepts the unprefixed legacy layout of `openai-community/gpt2`. `runtime._model_classes` wraps `sanitize` to strip the prefix and reject a genuinely untied `lm_head.weight` (mlx-lm's gpt2 always ties the head to `wte`, so loading one would silently produce wrong logits). Keep `Model.__module__` pointing at `mlx_lm.models.gpt2` or the mask patch resolves the wrong module.
- **Metal fp32 is not CPU fp32.** MLX GPU matmul accumulates at ~1e-3 *relative* error (a single 768-dim-reduction matmul shows 0.1 max-abs vs f64; whole distilgpt2 forward 0.073 on logits of ~85; tiny-Qwen2 forward 1.9e-4 on logits of ~0.04). MLX CPU matches torch-f32 at 8e-5. Hence the parity gates assert measured-noise-margin bounds per model (tests/test_parity.py: 1e-3 CPU / 0.5 GPU; tests/test_dream.py: 1e-6 CPU / 2e-3 GPU) with causality-regression signal >10x above each GPU bound. Don't "fix" a GPU parity failure by loosening past the documented signal margin - investigate instead.
- **transformers causality seams move between 5.x minors.** GPT-2 on 5.0 had TWO seams (per-layer `bias` buffer inside eager attention + model-level `create_causal_mask`); 5.12 removed the buffer, leaving only `create_causal_mask` (the GPT-2 `ref_model` fixture guards the fill with `hasattr`). Qwen2 on 5.12 has only `create_causal_mask`. When bumping transformers, expect the ref fixtures to need seam re-verification - a missed seam fails the parity gate loudly (diff lands at causal-signal magnitude, orders above tolerance).
- **tiny-Qwen2's vocab (151665) is one short of Dream's mask id (151666).** Passing the real Dream `mask_token_id` to the tiny test checkpoint indexes out of embedding range; tests pass an explicit in-range sentinel instead.

## Contract notes

- Bidirectional attention is a decode-time policy, NOT in a2d weights/config; a stock causal loader runs a2d checkpoints causally and produces garbage for diffusion. The `a2d` block in config.json (objective / mask_token_id / final_alpha / sampler) is the only marker; `runtime._parse_a2d` tolerates future extra keys.
- Diffusion decode on the GPT-2 path never uses a KV cache (`cache=None`); mlx-lm samplers/tokenizer remain reusable for the future denoise loop. The fast Dream path DOES cache, via fast_dllm_mlx's dual KV cache.
- Qwen/Dream entry points (`mlx_dllm/dream.py`): `dream.load` / `dream.generate` (reference loop, forced bidirectional) and `dream.load_fast` / `dream.generate_fast` (Fast-dLLM accelerated). The parity gate covers both engine model classes; `mlx_dllm.bidirectional_forward` works unchanged on them because they import `create_attention_mask` into their own modules just like mlx-lm models.
- Fast-dLLM-mlx stays a dependency: no engine source is copied into this repo, so Apache-2.0 attribution needs nothing beyond the dependency reference (their LICENSE/NOTICE travel with their package).
