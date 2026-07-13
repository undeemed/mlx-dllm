# mlx-dllm

MLX (Apple Silicon) inference runtime for masked-diffusion language-model checkpoints produced by [a2d](https://github.com/undeemed/a2d), plus a native reference decoder for Qwen/Dream-family diffusion models.

a2d converts autoregressive checkpoints (GPT-2 today) into masked-diffusion models (MDLM).
The saved artifact is a standard Hugging Face triple (`config.json` + tokenizer + `model.safetensors`) plus an `"a2d"` block in `config.json`.
a2d also emits a **run-directory** around that checkpoint (a `model/` subdir, a `manifest.json`, and `checkpoints/checkpoint-<N>/` resume material); the runtime loads either shape.
Bidirectional attention is a decode-time policy, not baked into the weights - this runtime supplies it.

## What exists today

- `mlx_dllm.load(path_or_repo)` - loads GPT-2, Qwen2, Gemma (v1), and Gemma 3 (text) family HF checkpoints via `mlx_lm` (unmodified, as a library) and returns `(model, tokenizer, a2d)` with the parsed `a2d` config block when present.
`path_or_repo` may be a plain HF checkpoint (local dir or hub repo id) **or an a2d run-dir root** (see below).
Per-family loading policy lives in the `mlx_dllm/families/` adapter registry: a new model family is one new module with a `register(...)` call, no edits to shared dispatch (contract and worked example in `mlx_dllm/families/__init__.py`).
Gemma 3 interleaves local (sliding-window) and global (full) attention layers; at a2d's `alpha=1` decode policy the window is dropped and every layer runs full non-causal unwindowed attention, which the runtime reaches by neutralizing both the causal and sliding-window masks with no per-family code (see `mlx_dllm/families/gemma3.py`).
- **Run-dir loading** - point `load()` at an a2d run-dir root and it resolves the consumable checkpoint (prefers `model/`, else the latest `checkpoints/checkpoint-<N>/` by numeric N), attaches `manifest.json` to the returned config as `a2d.manifest` (provenance only), and reads `mask_token_id` and the sampler from the `a2d` config block - the manifest's own `model_spec.mask_token_id` (the pre-conversion value, often `null`) is never used. Run-dir detection is by `manifest.json` and/or a `model/` subdir, unless a `config.json` sits at the root - a real run-dir root never has one, so that always marks a plain HF checkpoint; plain HF checkpoints are unaffected. It is fully model-agnostic (no per-family logic).
- `mlx_dllm.bidirectional_forward(model, input_ids)` - a full non-causal forward pass (no KV cache) returning logits for **all** positions, matching a2d's `alpha=1` decode configuration.
- `mlx_dllm.denoise(model, canvas, mask_token_id=..., steps=...)` - the native Qwen/Dream correctness path: full bidirectional recomputation, greedy per-position predictions, and linearly scheduled confidence-ranked reveals with no KV cache or remasking.
Predictions are read in-place (token for position `i` from logits at `i`), the a2d convention since a2d conversion drops the autoregressive next-token shift; pass `logit_shift=True` for published Dream checkpoints that keep the next-token head (token for position `i` from logits at `i - 1`).
- `mlx_dllm.generate(...)` - creates a fixed masked continuation canvas, denoises it with that reference path, and returns only the decoded continuation (prompt text excluded).
- **a2d defaults** - `generate` and `denoise` take an optional `a2d=A2DConfig` (the one `load` returns): `mask_token_id`, `canvas_len` (`generate`'s `max_new_tokens`), and `num_steps` default from the run-dir's own `a2d` block, so a run-dir round-trips as `model, tok, a2d = load(run_dir); generate(model, tok, prompt, a2d=a2d)`. Explicit arguments always override; the sampler's `temperature` is ignored because the reference decoder is greedy.
- Numerical parity gates prove the GPT-2, Qwen2, Gemma (v1), and Gemma 3 MLX forwards match PyTorch/HF eager bidirectional references.
The Qwen and Gemma tests use tiny fixtures; the Gemma 3 gate also disables HF's sliding-window mask so both frameworks run full unwindowed attention. Full-size Dream/Gemma 3 validation is deliberately deferred to separate (Apple-silicon) hardware.

Acceleration (dual cache / confident-parallel decoding) and a CLI are follow-on work.

## Prior art

The Qwen/Dream diffusion techniques were informed by [Fast-dLLM-mlx](https://github.com/MacPaw/Fast-dLLM-mlx) by MacPaw (Apache-2.0) and the original [NVLabs/Fast-dLLM](https://github.com/NVlabs/Fast-dLLM). They are reimplemented independently here: no Fast-dLLM-mlx code is copied, vendored, forked, imported, or added as a dependency. This implementation stays lean by reusing mlx-lm's own Qwen2 transformer and adding only the diffusion decoding layer.

## Install

```sh
pip install -e .           # runtime: mlx + mlx-lm
pip install -e ".[test]"   # + torch/transformers for the parity gate
pytest
```

On macOS, `mlx` pulls the Metal backend automatically.
On Linux the install resolves to `mlx[cpu]` (the ABI-matched CPU backend; the plain `mlx` wheel's `libmlx.so` is a Metal-only stub there), so the runtime is CPU-only and the Metal-gated parity tests skip.
