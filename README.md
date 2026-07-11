# mlx-dllm

MLX (Apple Silicon) inference runtime for masked-diffusion language-model checkpoints produced by [a2d](https://github.com/undeemed/a2d).

a2d converts autoregressive checkpoints (GPT-2 today) into masked-diffusion models (MDLM).
The saved artifact is a standard Hugging Face triple (`config.json` + tokenizer + `model.safetensors`) plus an `"a2d"` block in `config.json`.
Bidirectional attention is a decode-time policy, not baked into the weights - this runtime supplies it.

## What exists today

- `mlx_dllm.load(path_or_repo)` - loads any GPT-2 HF-layout checkpoint via `mlx_lm` (unmodified, as a library) and returns the parsed `a2d` config block when present.
- `mlx_dllm.bidirectional_forward(model, input_ids)` - a full non-causal forward pass (no KV cache) returning logits for **all** positions, matching a2d's `alpha=1` decode configuration.
- A numerical parity gate (`tests/test_parity.py`) proving the MLX bidirectional forward matches a PyTorch/HF eager bidirectional reference.

The iterative denoise loop and CLI are follow-on work.

## Install

```sh
pip install -e .           # runtime: mlx + mlx-lm
pip install -e ".[test]"   # + torch/transformers for the parity gate
pytest
```
