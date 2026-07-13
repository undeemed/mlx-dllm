# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

## What this is

MLX inference runtime for a2d masked-diffusion (MDLM) GPT-2 checkpoints and a
native Qwen/Dream-family reference decoder.
mlx-lm is consumed as a library, never forked; the only behavioral change is decode-time bidirectional attention (a2d's alpha=1 policy), applied via a scoped rebind of `create_attention_mask` in the model's own mlx-lm module (`mlx_dllm/runtime.py`).
The Qwen path reuses mlx-lm's stock `qwen2` model and adds a fixed-canvas,
confidence-ranked iterative denoise loop; Gemma (v1) and Gemma 3 (text) load the
same way through their no-op family adapters. `runtime.load` accepts either a
plain HF checkpoint or an a2d run-dir root (see "a2d run-dir loading" under
Contract notes). CLI and acceleration remain follow-on PRs.

## Build / test

- `uv venv && uv pip install -e ".[test]"` then `pytest` (needs network once to fetch distilgpt2 and the tiny Qwen2/Gemma fixtures; ~9s after caching). The Gemma 3 fixture is vendored under `tests/fixtures/gemma3-a2d-sample` (no download).
- Use `distilbert/distilgpt2` (canonical id) in code and tests - the bare `distilgpt2` alias 404s on the HF xet download path.
- **Torch is CPU-only on Linux via a uv source override.** The default PyPI torch wheel is a CUDA build (hundreds of MB of unusable GPU libs on a CUDA-less/Metal-less Linux box); `pyproject.toml`'s `[[tool.uv.index]] pytorch-cpu` + `[tool.uv.sources] torch` marker (`sys_platform == 'linux'`) makes fresh uv resolution pick `torch==<pin>+cpu` on Linux while macOS keeps the default MPS-capable wheel. It changes the source, never the version pin. For a manual install use `uv pip install torch --index-url https://download.pytorch.org/whl/cpu`.
- **Linux is CPU-only and needs the right mlx backend.** The `mlx` wheel ships only the Python bindings; its bundled `libmlx.so` is a Metal stub that fails to import on Linux (`libmlx.so: cannot open shared object file`), and mlx-lm depends on `mlx` only under Darwin. `pyproject.toml` therefore pulls `mlx[cpu]` on Linux (the `cpu` extra adds the ABI-matched `mlx-cpu` backend) and plain `mlx` elsewhere (Darwin auto-adds `mlx-metal`). Bare `mlx-cpu` is NOT enough - it has the backend lib but no `mlx.core` bindings. On Linux the three Metal-gated parity tests skip (`mx.metal.is_available()` is False); CPU parity gates still run and must stay green.

## Sharp edges (all bitten once)

- **Dependency pins are load-bearing.** mlx-lm 0.31.x declares `transformers>=5` but crashes on import with transformers >=5.13 (`AutoTokenizer.register` signature change). Runtime pins `transformers>=5.0,<5.13`; tests pin `~=5.0.0` because the parity reference patches transformers-5.0-verified seams. mlx-lm itself is pinned `>=0.31,<0.32` because runtime.py reaches into private internals (`_download`, `_get_classes`) and the gpt2 module's `create_attention_mask` binding.
- **Weight-key normalization is required.** Modern `save_pretrained` output (what a2d writes; also distilgpt2) prefixes keys with `transformer.` and omits tied `lm_head.weight`; mlx-lm's gpt2 module only accepts the unprefixed legacy layout of `openai-community/gpt2`. The gpt2 family adapter (`mlx_dllm/families/gpt2.py`) wraps `sanitize` to strip the prefix and reject a genuinely untied `lm_head.weight` (mlx-lm's gpt2 always ties the head to `wte`, so loading one would silently produce wrong logits). `Model.__module__` must point at `mlx_lm.models.gpt2` or the mask patch resolves the wrong module - `runtime._model_classes` pins this for every wrapped family, so adapters never manage it. See "Family adapters" below.
- **Metal fp32 is not CPU fp32.** MLX GPU matmul accumulates at ~1e-3 relative error (a single 768-dim-reduction matmul shows 0.1 max-abs vs f64; whole distilgpt2 forward 0.073 on logits of ~85). MLX CPU matches torch-f32 at 8e-5. Hence the GPT-2 parity gate (tests/test_parity.py) asserts 1e-3 on CPU and 0.5 on GPU; a causality regression shows up as ~10, so both bounds stay discriminative. Don't "fix" a GPU parity failure by loosening past ~0.5 - investigate instead.
- **transformers 5.x GPT-2 has TWO causality seams**: the per-layer `bias` buffer inside eager attention AND a model-level `create_causal_mask`. A bidirectional torch reference must neutralize both (see `ref_model` fixture).
- **transformers 5.0 Qwen2 and Gemma each have one causality seam for eager attention:** the model-level `create_causal_mask`. mlx-lm Qwen2 and Gemma use the same module-local `create_attention_mask` seam as GPT-2, so `_no_causal_mask` works for all three without wrapping or reimplementing the transformer.
- **transformers 5.0 Gemma 3 has TWO mask seams:** the model-level `create_causal_mask` (global/`full_attention` layers) AND `create_sliding_window_causal_mask` (local/`sliding_attention` layers), assembled into a per-layer mapping. A full-attention torch reference must neutralize both (see `tests/test_gemma3.py`'s `ref_model`).
- **The native Qwen parity fixture is intentionally tiny.** `trl-internal-testing/tiny-Qwen2ForCausalLM-2.5` is a two-layer, hidden-size-8 Qwen2 model (~5 MB). Its weights are bf16; cast both MLX and HF loads to fp32 before measuring framework parity. Its parity gate (tests/test_qwen.py) asserts 1e-6 on CPU and 2e-3 on GPU; measured values live in that file's docstring. Full-size Dream runs belong on a mini.
- **Gemma parity is gelu-limited, not fp32-limited.** mlx-lm's `gemma` module hardcodes the exact (erf) `nn.gelu`, but published Gemma configs (and the tiny fixture `trl-internal-testing/tiny-GemmaForCausalLM`) set `hidden_act="gelu_pytorch_tanh"`, which HF-eager honors. So bidirectional CPU parity vs HF is ~2.85e-5 (dominated by that activation gap; forcing HF to exact gelu collapses it to ~8e-8), NOT qwen2's ~1e-6. The gemma gate (tests/test_gemma.py) is therefore 1e-4 CPU / 2e-3 GPU, still 290x below the 2.9e-2 causal-regression signal. This is stock mlx-lm behavior (never forked); immaterial for greedy argmax denoise. This entry is Gemma **v1** only (`model_type="gemma"`); Gemma **3** is fp32-limited instead (next entry), and gemma2 is out of scope - its mlx-lm module builds its single mask with `create_attention_mask(..., return_array=True)`, unverified against this seam.
- **Gemma 3 (`model_type="gemma3_text"`) is a no-op adapter, and its parity is fp32-limited, NOT gelu-limited.** Contrary to the earlier gemma2/gemma3 note, mlx-lm's `gemma3_text` builds BOTH masks through the plain module-local `create_attention_mask` - the global one as `create_attention_mask(h, cache)` and the local/sliding one as the same call with `window_size=` (no `return_array=True`). So `_no_causal_mask`'s `lambda: None` neutralizes both, giving full non-causal UNWINDOWED attention on every layer (Case A) with no seam extension and no per-family code; `families/gemma3.py` registers `sanitize_wrapper=None`. Unlike gemma1, its MLP uses `gelu_approx` (the tanh gelu) which matches published `gelu_pytorch_tanh`, so CPU parity vs HF full-attention is fp32 noise (~2.79e-7, gate 1e-5 / 2e-3 GPU), ~10x qwen2's floor from 6 layers + qk-norm + four RMSNorms/layer. Discriminative: causal-vs-bidir ~0.15, unwindowed-vs-windowed ~0.30. The fixture is vendored (`tests/fixtures/gemma3-a2d-sample`) since no HF-hosted a2d-gemma3 tiny model exists. gemma2/gemma3-vision remain out of scope.

## Contract notes

- Bidirectional attention is a decode-time policy, NOT in a2d weights/config; a stock causal loader runs a2d checkpoints causally and produces garbage for diffusion. The `a2d` block in config.json (objective / mask_token_id / final_alpha / sampler) is the only marker; `runtime._parse_a2d` tolerates future extra keys.
- Diffusion decode never uses a KV cache (`cache=None`); mlx-lm samplers/tokenizer remain reusable for the future denoise loop.
- The Qwen reference denoiser is greedy, batch-one, fixed-canvas, confidence-ranked, and never remasks. It suppresses the mask token from output logits so every scheduled reveal makes progress. Schedule steps that reveal nothing skip the forward pass entirely. Dual-cache/confident-parallel acceleration and a2d sampler/config integration are intentionally absent.
- **Prediction-position convention is in-place by default** (token for position i read from logits at i) because this runtime targets a2d-converted checkpoints and a2d conversion drops the autoregressive next-token shift.
Published Dream-family checkpoints keep the next-token head (token for position i comes from logits at i-1); `denoise`/`generate` take `logit_shift=True` for those, which requires an unmasked first canvas position.
Do not feed a published Dream checkpoint through the default in-place path - it will silently garble output.
- `generate` returns only the decoded continuation; the prompt text is never included.
- **a2d run-dir loading is model-agnostic** (`runtime._resolve_run_dir` + `load`). A run-dir root (detected by `manifest.json` and/or a `model/` subdir) resolves to `model/`, else the latest `checkpoints/checkpoint-<N>/` by numeric N (not lexical: `checkpoint-10` > `checkpoint-2`). The `a2d` config block is authoritative for `mask_token_id`/sampler; `manifest.json` is provenance only (attached as `a2d.manifest`) - its `model_spec.mask_token_id` is the pre-conversion value, often `null`, and is never consulted. A raw `checkpoint-<N>/` carries no a2d block, so that fallback returns `a2d=None`. Plain HF dirs/repo ids are unaffected (backward-compatible). `generate`/`denoise` accept an optional `a2d=` whose block supplies defaults (`mask_token_id`, `canvas_len`->`max_new_tokens`, `num_steps`->`steps`); explicit args override and `temperature` is ignored (greedy decoder). Tests build a tiny run-dir in `tmp_path` from `tests/fixtures/gemma3-a2d-sample` - see `tests/test_a2d_loader.py`. OUT OF SCOPE still: CLI, acceleration.

## Family adapters (adding a model family)

New model families plug in through the `mlx_dllm/families/` registry, not by editing shared dispatch. `runtime._model_classes` resolves mlx-lm's stock `(Model, ModelArgs)` and then applies whatever adapter `families.get_adapter(model_type)` returns; the bidirectional mask seam (`_no_causal_mask`) and the denoise loop (`diffusion.py`) stay family-agnostic and must never grow per-family conditionals.

To add a family, create one module `mlx_dllm/families/<model_type>.py` that builds a `FamilyAdapter` and calls `register(...)` at import time - nothing else needs editing (the package auto-imports its submodules, and `_model_classes`'s dispatch body does not change). An adapter supplies:

- `model_type` - the `config["model_type"]` string it handles.
- `sanitize_wrapper` - OPTIONAL factory `(model_class) -> subclass` that overrides `sanitize` when the family's on-disk weight layout differs from what its mlx-lm module expects (gpt2 needs one; qwen2, gemma, and gemma3 register `None`). A family that loads cleanly through stock mlx-lm registers `sanitize_wrapper=None`, or need not register at all - an unregistered `model_type` also falls through to stock mlx-lm.

Family modules are auto-imported while `mlx_dllm` is still initializing: no top-level imports from `mlx_dllm`/`mlx_dllm.runtime` (partially-initialized ImportError), and an import-time exception in any family module breaks `import mlx_dllm` for everyone - full constraints in `mlx_dllm/families/__init__.py`.

The `Model.__module__` pin the mask seam depends on is applied by `_model_classes` for any wrapped class, so adapters never manage it. See `mlx_dllm/families/__init__.py` (contract + worked example), `families/gpt2.py` (wrapper), and `families/qwen2.py`/`families/gemma.py`/`families/gemma3.py` (no-ops; `gemma3.py`'s docstring explains why sliding-window attention still needs no special seam). OUT OF SCOPE for the registry: the denoise loop, acceleration, and the a2d-format bridge.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
