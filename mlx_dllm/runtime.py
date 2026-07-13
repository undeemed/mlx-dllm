"""Load GPT-2/Qwen2/Gemma/Gemma-3 checkpoints and run bidirectional forwards on MLX.

a2d checkpoints are standard HF triples (config.json + tokenizer +
model.safetensors) whose only non-standard bytes are an "a2d" block inside
config.json. Bidirectional attention is NOT in the weights or config - it is a
decode-time policy this runtime supplies (a2d's alpha=1 configuration).
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

# _download/_get_classes are the guts of mlx_lm.utils.load, which offers no
# public seam to hook sanitize; we mirror its 4-line body instead (mlx-lm
# pinned >=0.31,<0.32 in pyproject).
from mlx_lm.utils import _download, _get_classes, load_model, load_tokenizer

from mlx_dllm.families import get_adapter


@dataclass(frozen=True)
class A2DConfig:
    """The "a2d" block a2d splices into config.json.

    ``manifest`` is the run-dir's :file:`manifest.json` (provenance metadata),
    attached by :func:`load` when a run-dir root is loaded and ``None`` for a
    plain HF checkpoint. It is metadata only: the ``a2d`` config block - never
    the manifest - is authoritative for ``mask_token_id`` and the sampler.
    """

    objective: str  # "mdlm"
    mask_token_id: int
    final_alpha: float = 1.0  # provenance; decode always runs alpha=1
    sampler: Optional[dict] = None  # canvas_len / num_steps / temperature
    manifest: Optional[dict] = None  # run-dir manifest.json provenance, if any


def _parse_a2d(config: dict) -> Optional[A2DConfig]:
    block = config.get("a2d")
    if block is None:
        return None
    # Tolerate future extra keys the same way mlx-lm's ModelArgs.from_dict does.
    return A2DConfig(
        objective=block["objective"],
        mask_token_id=block["mask_token_id"],
        final_alpha=block.get("final_alpha", 1.0),
        sampler=block.get("sampler"),
    )


def _model_classes(config: dict):
    """Resolve mlx-lm's (Model, ModelArgs) classes, applying a family adapter.

    Dispatch is family-agnostic: the per-``model_type`` policy lives in the
    :mod:`mlx_dllm.families` registry, so adding a family is a new adapter
    registration, never an edit here. An adapter's ``sanitize_wrapper`` (when
    present) subclasses the stock model to normalize weight keys; families
    without one (or unregistered ``model_type``s) load through stock mlx-lm.
    """
    model_class, args_class = _get_classes(config)
    adapter = get_adapter(config.get("model_type"))
    if adapter is None or adapter.sanitize_wrapper is None:
        return model_class, args_class

    wrapped = adapter.sanitize_wrapper(model_class)
    # Keep the wrapped class in the stock mlx-lm model module so the
    # _no_causal_mask seam (which rebinds create_attention_mask in
    # sys.modules[type(model).__module__]) resolves the family's own module.
    wrapped.__module__ = model_class.__module__
    return wrapped, args_class


def _parse_manifest(manifest_path: Path) -> Optional[dict]:
    """Read a run-dir ``manifest.json`` as an opaque provenance dict.

    Returns the raw JSON object untouched (tolerant of unknown/extra keys and of
    ``model_spec.mask_token_id == null``); the manifest is metadata only and its
    mask id is the PRE-conversion detected value, so it is never consulted for
    the runtime mask token. Returns ``None`` if the file is absent, unreadable,
    or not a JSON object.
    """
    try:
        obj = json.loads(manifest_path.read_text())
    except (OSError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def _latest_checkpoint(root: Path) -> Optional[Path]:
    """Return the ``checkpoints/checkpoint-<N>`` dir with the greatest numeric N.

    Selection is by the integer suffix, not lexical order, so ``checkpoint-10``
    beats ``checkpoint-9``. Returns ``None`` when there is no such directory.
    """
    checkpoints = root / "checkpoints"
    if not checkpoints.is_dir():
        return None
    prefix = "checkpoint-"
    numbered = [
        (int(d.name[len(prefix) :]), d)
        for d in checkpoints.iterdir()
        if d.is_dir()
        and d.name.startswith(prefix)
        and d.name[len(prefix) :].isdecimal()
    ]
    if not numbered:
        return None
    return max(numbered, key=lambda item: item[0])[1]


def _resolve_run_dir(path_or_repo: str) -> tuple[Optional[Path], Optional[dict]]:
    """Resolve an a2d run-dir root to its consumable checkpoint + manifest.

    A local directory is treated as a run-dir when it holds a ``manifest.json``
    and/or a ``model/`` subdirectory - unless a ``config.json`` sits at its
    root. A genuine a2d run-dir root never has one (its configs live inside
    ``model/`` and the checkpoint dirs), so a root ``config.json`` marks a plain
    HF checkpoint that merely carries a stray manifest or ``model/`` entry. The
    consumable checkpoint is ``model/`` when present, else the latest
    ``checkpoints/checkpoint-<N>/`` (numeric max). Anything else - a plain HF
    checkpoint directory, or a remote/absent repo id - returns ``(None, None)``
    so :func:`load` falls through to stock loading.
    """
    root = Path(path_or_repo)
    if not root.is_dir():
        return None, None  # remote repo id or a bare file: not a local run-dir
    if (root / "config.json").is_file():
        return None, None  # loadable checkpoint at the root: plain HF dir
    manifest_path = root / "manifest.json"
    model_dir = root / "model"
    if not manifest_path.is_file() and not model_dir.is_dir():
        return None, None  # plain HF checkpoint directory (today's behavior)

    manifest = _parse_manifest(manifest_path) if manifest_path.is_file() else None
    if model_dir.is_dir():
        return model_dir, manifest
    checkpoint = _latest_checkpoint(root)
    if checkpoint is None:
        raise FileNotFoundError(
            f"a2d run-dir {root} has neither a model/ subdirectory nor any "
            "checkpoints/checkpoint-<N>/ to load"
        )
    return checkpoint, manifest


def load(path_or_repo: str):
    """Load an HF-layout GPT-2/Qwen2/Gemma/Gemma-3 checkpoint through stock mlx-lm.

    ``path_or_repo`` may be a plain HF checkpoint (local directory or hub repo
    id) or an a2d **run-dir root** (identified by ``manifest.json`` and/or a
    ``model/`` subdirectory). For a run-dir the consumable checkpoint is
    ``model/`` when present, else the latest ``checkpoints/checkpoint-<N>/``.

    Returns ``(model, tokenizer, a2d)`` where ``a2d`` is the parsed
    :class:`A2DConfig` when the loaded checkpoint carries a2d's config block,
    else ``None``. mlx-lm's config parsing ignores the extra "a2d" key, so both
    a2d-converted and stock GPT-2/Qwen2/Gemma/Gemma-3 checkpoints load as-is.
    Qwen2, Gemma (v1), and Gemma 3 (text) resolve directly to mlx-lm's own
    ``qwen2.Model``, ``gemma.Model``, and ``gemma3_text.Model``; no
    transformer code is reimplemented here.

    When a run-dir manifest is present it is attached to the returned
    ``A2DConfig`` as ``a2d.manifest`` (provenance only). The ``a2d`` config
    block, not the manifest, is authoritative for ``mask_token_id`` and the
    sampler. In the rare fallback case where the resolved checkpoint carries no
    a2d block (a raw ``checkpoint-<N>/``), ``a2d`` is ``None``.
    """
    checkpoint, manifest = _resolve_run_dir(path_or_repo)
    if checkpoint is None:
        checkpoint = _download(path_or_repo)
    model, config = load_model(checkpoint, get_model_classes=_model_classes)
    model.eval()
    tokenizer = load_tokenizer(checkpoint, eos_token_ids=config.get("eos_token_id"))
    a2d = _parse_a2d(config)
    if a2d is not None and manifest is not None:
        a2d = replace(a2d, manifest=manifest)
    return model, tokenizer, a2d


@contextmanager
def _no_causal_mask(model: nn.Module):
    """Scoped monkeypatch: the model's mlx-lm module builds no attention mask.

    mlx-lm model ``__call__``s hardcode ``mask = create_attention_mask(...)``
    (-> "causal"); rebinding that name in the model's own module for the
    duration of the call makes SDPA run with ``mask=None``, i.e. fully
    bidirectional. Mirrors a2d's decode-time patch of HF's eager seam
    (a2d transform/attention.py).
    """
    mod = sys.modules[type(model).__module__]
    orig = mod.create_attention_mask
    mod.create_attention_mask = lambda *args, **kwargs: None
    try:
        yield
    finally:
        mod.create_attention_mask = orig


def bidirectional_forward(model: nn.Module, input_ids: mx.array) -> mx.array:
    """Full non-causal forward pass: logits for ALL positions, no KV cache.

    This is a2d's decode configuration (alpha=1, use_cache=False): diffusion
    decoding recomputes the whole canvas each step.
    """
    # ponytail: process-global rebind during the call; fine single-threaded,
    # revisit if forwards ever run concurrently.
    with _no_causal_mask(model):
        return model(input_ids, cache=None)
