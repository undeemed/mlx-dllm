"""Native reference decoding for Qwen/Dream-family diffusion models."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mlx_dllm.runtime import A2DConfig, bidirectional_forward


def _a2d_default(explicit, a2d_value):
    """Return ``explicit`` when the caller supplied it, else the a2d default."""
    return explicit if explicit is not None else a2d_value


def denoise(
    model: nn.Module,
    canvas: mx.array,
    *,
    mask_token_id: int | None = None,
    steps: int | None = None,
    logit_shift: bool = False,
    a2d: A2DConfig | None = None,
) -> mx.array:
    """Greedily reveal a masked, fixed-length canvas over ``steps`` passes.

    This is the deliberately simple correctness path: every step recomputes
    every position with bidirectional attention and no KV cache. Revealed
    tokens remain fixed; the highest-confidence masked positions are revealed
    according to a linear cumulative schedule.

    ``mask_token_id`` and ``steps`` default from ``a2d`` (an :class:`A2DConfig`,
    typically the one returned by :func:`mlx_dllm.load` for a run-dir): the mask
    id from ``a2d.mask_token_id`` and ``steps`` from ``a2d.sampler["num_steps"]``.
    Explicit arguments always take precedence. ``a2d.sampler["temperature"]`` is
    intentionally ignored - this reference decoder is greedy (argmax).

    By default the prediction for position ``i`` is read from the logits at
    position ``i`` (in-place). This is the a2d convention: a2d conversion
    drops the autoregressive next-token shift, so a2d-converted checkpoints
    predict each masked position at its own logit row. Published Dream-family
    checkpoints keep the next-token head instead, reading the token for
    position ``i`` from the logits at ``i - 1``; pass ``logit_shift=True``
    for those, which requires an unmasked first canvas position.

    The reference path currently accepts one sequence at a time. Batched
    scheduling can be added when there is a concrete caller for it.
    """
    sampler = a2d.sampler if a2d is not None else None
    mask_token_id = _a2d_default(
        mask_token_id, a2d.mask_token_id if a2d is not None else None
    )
    steps = _a2d_default(steps, sampler.get("num_steps") if sampler else None)
    if mask_token_id is None:
        raise ValueError("mask_token_id is required (pass it or an a2d config)")
    if steps is None:
        raise ValueError("steps is required (pass it or an a2d config with a sampler)")
    if canvas.ndim != 2 or canvas.shape[0] != 1:
        raise ValueError("canvas must have shape (1, sequence_length)")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if not 0 <= mask_token_id < model.args.vocab_size:
        raise ValueError("mask_token_id must be inside the model vocabulary")
    if logit_shift and bool((canvas[0, 0] == mask_token_id).item()):
        raise ValueError("logit_shift requires an unmasked first canvas position")

    masked_count = int(mx.sum(canvas == mask_token_id).item())
    if masked_count == 0:
        return canvas

    revealed = 0
    positions = mx.arange(canvas.shape[1])
    vocabulary = mx.arange(model.args.vocab_size)

    for step in range(steps):
        reveal_total = round((step + 1) * masked_count / steps)
        reveal_count = reveal_total - revealed
        if reveal_count == 0:
            continue

        logits = bidirectional_forward(model, canvas)
        if logit_shift:
            logits = mx.concatenate([logits[:, :1], logits[:, :-1]], axis=1)
        # A mask is an input sentinel, never a valid revealed output token.
        logits = mx.where(vocabulary == mask_token_id, -mx.inf, logits)
        predictions = mx.argmax(logits, axis=-1)
        confidence = mx.max(mx.softmax(logits, axis=-1), axis=-1)

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
    max_new_tokens: int | None = None,
    mask_token_id: int | None = None,
    steps: int | None = None,
    logit_shift: bool = False,
    a2d: A2DConfig | None = None,
) -> str:
    """Append a masked canvas to ``prompt``, denoise it, and decode only the
    continuation (the prompt text is not included in the return value).

    ``max_new_tokens``, ``mask_token_id``, and ``steps`` default from ``a2d``
    (an :class:`A2DConfig`, typically the one returned by :func:`mlx_dllm.load`
    for a run-dir): the canvas length from ``a2d.sampler["canvas_len"]``, the
    mask id from ``a2d.mask_token_id``, and the step count from
    ``a2d.sampler["num_steps"]``. Explicit arguments always take precedence.
    ``a2d.sampler["temperature"]`` is intentionally ignored - the reference
    decoder is greedy. With no ``a2d`` and no explicit ``steps``, ``steps``
    defaults to ``max_new_tokens`` (one reveal per position, as before).
    """
    sampler = a2d.sampler if a2d is not None else None
    max_new_tokens = _a2d_default(
        max_new_tokens, sampler.get("canvas_len") if sampler else None
    )
    mask_token_id = _a2d_default(
        mask_token_id, a2d.mask_token_id if a2d is not None else None
    )
    steps = _a2d_default(steps, sampler.get("num_steps") if sampler else None)
    if max_new_tokens is None:
        raise ValueError(
            "max_new_tokens is required (pass it or an a2d config with a sampler)"
        )
    if mask_token_id is None:
        raise ValueError("mask_token_id is required (pass it or an a2d config)")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    prompt_ids = tokenizer.encode(prompt)
    canvas = mx.array([[*prompt_ids, *([mask_token_id] * max_new_tokens)]])
    output = denoise(
        model,
        canvas,
        mask_token_id=mask_token_id,
        steps=max_new_tokens if steps is None else steps,
        logit_shift=logit_shift,
    )
    return tokenizer.decode(output[0].tolist()[len(prompt_ids) :])
