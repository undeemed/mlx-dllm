# mlx-dllm

MLX (Apple Silicon) inference runtime for masked-diffusion language-model checkpoints produced by [a2d](https://github.com/undeemed/a2d), plus a native reference decoder for Qwen/Dream-family diffusion models.

a2d converts autoregressive checkpoints (GPT-2 today) into masked-diffusion models (MDLM).
The saved artifact is a standard Hugging Face triple (`config.json` + tokenizer + `model.safetensors`) plus an `"a2d"` block in `config.json`.
Bidirectional attention is a decode-time policy, not baked into the weights - this runtime supplies it.

## What exists today

- `mlx_dllm.load(path_or_repo)` - loads GPT-2 and Qwen2-family HF checkpoints via `mlx_lm` (unmodified, as a library) and returns the parsed `a2d` config block when present.
- `mlx_dllm.bidirectional_forward(model, input_ids)` - a full non-causal forward pass (no KV cache) returning logits for **all** positions, matching a2d's `alpha=1` decode configuration.
- `mlx_dllm.denoise(model, canvas, mask_token_id=..., steps=...)` - the native Qwen/Dream correctness path: full bidirectional recomputation, greedy per-position predictions, and linearly scheduled confidence-ranked reveals with no KV cache or remasking.
Predictions are read in-place (token for position `i` from logits at `i`), the a2d convention since a2d conversion drops the autoregressive next-token shift; pass `logit_shift=True` for published Dream checkpoints that keep the next-token head (token for position `i` from logits at `i - 1`).
- `mlx_dllm.generate(...)` - creates a fixed masked continuation canvas, denoises it with that reference path, and returns only the decoded continuation (prompt text excluded).
- Numerical parity gates prove the GPT-2 and Qwen2 MLX forwards match PyTorch/HF eager bidirectional references. The Qwen test uses a two-layer ~5 MB fixture; full-size Dream validation is deliberately deferred to separate hardware.

Acceleration (dual cache / confident-parallel decoding), an a2d-format bridge for Qwen, and a CLI are follow-on work.

## Prior art

The Qwen/Dream diffusion techniques were informed by [Fast-dLLM-mlx](https://github.com/MacPaw/Fast-dLLM-mlx) by MacPaw (Apache-2.0) and the original [NVLabs/Fast-dLLM](https://github.com/NVlabs/Fast-dLLM). They are reimplemented independently here: no Fast-dLLM-mlx code is copied, vendored, forked, imported, or added as a dependency. This implementation stays lean by reusing mlx-lm's own Qwen2 transformer and adding only the diffusion decoding layer.

## Install

```sh
pip install -e .           # runtime: mlx + mlx-lm
pip install -e ".[test]"   # + torch/transformers for the parity gate
pytest
```
