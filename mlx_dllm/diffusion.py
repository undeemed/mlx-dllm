"""Native reference decoding for Qwen/Dream-family diffusion models."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mlx_dllm.runtime import bidirectional_forward


def denoise(
    model: nn.Module,
    canvas: mx.array,
    *,
    mask_token_id: int,
    steps: int,
) -> mx.array:
    """Greedily reveal a masked, fixed-length canvas over ``steps`` passes.

    This is the deliberately simple correctness path: every step recomputes
    every position with bidirectional attention and no KV cache. Revealed
    tokens remain fixed; the highest-confidence masked positions are revealed
    according to a linear cumulative schedule.

    The reference path currently accepts one sequence at a time. Batched
    scheduling can be added when there is a concrete caller for it.
    """
    if canvas.ndim != 2 or canvas.shape[0] != 1:
        raise ValueError("canvas must have shape (1, sequence_length)")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if not 0 <= mask_token_id < model.args.vocab_size:
        raise ValueError("mask_token_id must be inside the model vocabulary")

    masked_count = int(mx.sum(canvas == mask_token_id).item())
    if masked_count == 0:
        return canvas

    revealed = 0
    positions = mx.arange(canvas.shape[1])
    vocabulary = mx.arange(model.args.vocab_size)

    for step in range(steps):
        logits = bidirectional_forward(model, canvas)
        # A mask is an input sentinel, never a valid revealed output token.
        logits = mx.where(vocabulary == mask_token_id, -mx.inf, logits)
        predictions = mx.argmax(logits, axis=-1)
        confidence = mx.max(mx.softmax(logits, axis=-1), axis=-1)

        reveal_total = round((step + 1) * masked_count / steps)
        reveal_count = reveal_total - revealed
        if reveal_count == 0:
            continue

        still_masked = canvas == mask_token_id
        scores = mx.where(still_masked, confidence, -mx.inf)
        selected_indices = mx.argsort(-scores[0])[:reveal_count]
        selected = mx.any(positions[:, None] == selected_indices[None, :], axis=1)[
            None, :
        ]
        canvas = mx.where(selected & still_masked, predictions, canvas)
        mx.eval(canvas)
        revealed = reveal_total

    return canvas


def generate(
    model: nn.Module,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    mask_token_id: int,
    steps: int | None = None,
) -> str:
    """Append a masked canvas to ``prompt``, denoise it, and decode the text."""
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    prompt_ids = tokenizer.encode(prompt)
    canvas = mx.array([[*prompt_ids, *([mask_token_id] * max_new_tokens)]])
    output = denoise(
        model,
        canvas,
        mask_token_id=mask_token_id,
        steps=max_new_tokens if steps is None else steps,
    )
    return tokenizer.decode(output[0].tolist())
