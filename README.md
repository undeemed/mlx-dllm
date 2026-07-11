# mlx-dllm

MLX (Apple Silicon) inference runtime for masked-diffusion language-model checkpoints produced by [a2d](https://github.com/undeemed/a2d).

a2d converts autoregressive checkpoints (GPT-2 today) into masked-diffusion models (MDLM).
The saved artifact is a standard Hugging Face triple (`config.json` + tokenizer + `model.safetensors`) plus an `"a2d"` block in `config.json`.
Bidirectional attention is a decode-time policy, not baked into the weights - this runtime supplies it.

## What exists today

**GPT-2 / MDLM path (a2d's current output):**

- `mlx_dllm.load(path_or_repo)` - loads any GPT-2 HF-layout checkpoint via `mlx_lm` (unmodified, as a library) and returns the parsed `a2d` config block when present.
- `mlx_dllm.bidirectional_forward(model, input_ids)` - a full non-causal forward pass (no KV cache) returning logits for **all** positions, matching a2d's `alpha=1` decode configuration.
- A numerical parity gate (`tests/test_parity.py`) proving the MLX bidirectional forward matches a PyTorch/HF eager bidirectional reference.

**Qwen / Dream path (a2d's future output), powered by [Fast-dLLM-mlx](https://github.com/MacPaw/Fast-dLLM-mlx) (MacPaw, Apache-2.0) as a pinned git dependency:**

- `mlx_dllm.dream.load(path_or_repo)` + `mlx_dllm.dream.generate(model, tokenizer, prompt, ...)` - the engine's full iterative denoise loop, forced bidirectional (the engine's naive path otherwise inherits mlx-lm's causal default).
- `mlx_dllm.dream.load_fast` + `mlx_dllm.dream.generate_fast` - the Fast-dLLM accelerated path (dual KV cache, confident-parallel token commit).
- The same parity gate ported to both of the engine's Dream model classes (`tests/test_dream.py`), MLX vs a causality-neutralized PyTorch Qwen2 reference.

The a2d-format bridge for the Qwen path (a2d config parsing + MDLM sampler), the GPT-2 denoise loop, and a CLI are follow-on work.

## Install

Requires Python >= 3.13 (inherited from the Fast-dLLM-mlx dependency).

```sh
pip install -e .           # runtime: mlx + mlx-lm + fast-dllm-mlx
pip install -e ".[test]"   # + torch/transformers for the parity gates
pytest
```
