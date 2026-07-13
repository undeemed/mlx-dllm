"""Per-model-family loading adapters for the bidirectional MLX runtime.

New model families plug into :func:`mlx_dllm.runtime.load` through this registry
instead of editing shared dispatch. Each family declares only what deviates
from stock mlx-lm; the generic bits (the bidirectional ``create_attention_mask``
seam in ``runtime._no_causal_mask`` and the ``diffusion`` denoise loop) stay
family-agnostic and never learn about individual families.

Adding a family
---------------
Drop one module in this package (``mlx_dllm/families/<model_type>.py``) that
builds a :class:`FamilyAdapter` and calls :func:`register`. Nothing else needs
editing - the modules here are auto-imported, so a new file is picked up
automatically, and ``runtime._model_classes`` dispatches purely through
:func:`get_adapter`::

    # mlx_dllm/families/llama.py
    from mlx_dllm.families import FamilyAdapter, register

    def _wrap(model_class):
        class Model(model_class):
            def sanitize(self, weights):
                ...                       # normalize weight keys for this family
                return super().sanitize(weights)
        return Model

    register(FamilyAdapter(model_type="llama", sanitize_wrapper=_wrap))

A family that loads cleanly through stock mlx-lm registers with
``sanitize_wrapper=None`` (see ``qwen2`` or ``gemma``) - or need not register at all; an
unregistered ``model_type`` falls through to stock mlx-lm unchanged.

Import constraints: family modules are auto-imported at package import time,
while ``mlx_dllm`` (and ``mlx_dllm.runtime``) are still initializing. Do NOT
import from ``mlx_dllm`` or ``mlx_dllm.runtime`` at module top level - that
raises a partially-initialized-module ImportError. Import only ``mlx.core``/
``mlx.nn``, the stdlib, and ``mlx_dllm.families`` itself; defer any runtime
import into a function body if one is ever needed. An uncaught exception in
any family module makes ``import mlx_dllm`` fail for everyone, so keep these
modules import-light and side-effect-free apart from the ``register`` call.

What an adapter supplies
------------------------
* ``model_type`` - the ``config["model_type"]`` string this adapter handles.
* ``sanitize_wrapper`` - an OPTIONAL factory ``(model_class) -> subclass`` that
  overrides ``sanitize`` when the family's on-disk weight layout differs from
  what its mlx-lm module expects. ``None`` means "use stock mlx-lm as-is".

The model-module identity that ``_no_causal_mask`` relies on is handled for the
family: whenever a ``sanitize_wrapper`` produces a subclass, ``_model_classes``
pins the subclass's ``__module__`` back to the stock mlx-lm model module, so
``sys.modules[type(model).__module__].create_attention_mask`` keeps resolving
the family's own mlx-lm module. Families never manage ``__module__`` themselves.
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from typing import Callable, Dict, Optional


@dataclass(frozen=True)
class FamilyAdapter:
    """Loading policy for one model family (keyed by HF ``model_type``).

    See the module docstring for the full extension-point contract.

    Attributes:
        model_type: The ``config["model_type"]`` value this adapter handles.
        sanitize_wrapper: Optional factory that takes the stock mlx-lm model
            class and returns a subclass overriding ``sanitize`` to normalize
            weight keys. ``None`` (the common case) loads the family through
            stock mlx-lm unchanged.
    """

    model_type: str
    sanitize_wrapper: Optional[Callable[[type], type]] = None


_REGISTRY: Dict[str, FamilyAdapter] = {}


def register(adapter: FamilyAdapter) -> None:
    """Register ``adapter`` under its ``model_type`` (last registration wins)."""
    _REGISTRY[adapter.model_type] = adapter


def get_adapter(model_type: Optional[str]) -> Optional[FamilyAdapter]:
    """Return the adapter for ``model_type``, or ``None`` if none is registered.

    ``None`` means the family loads through stock mlx-lm classes unchanged - the
    default for any ``model_type`` without a dedicated adapter.
    """
    return _REGISTRY.get(model_type) if model_type is not None else None


def _autodiscover() -> None:
    """Import every family submodule so its ``register`` call runs on load."""
    for info in pkgutil.iter_modules(__path__):
        if not info.name.startswith("_"):
            importlib.import_module(f"{__name__}.{info.name}")


_autodiscover()
