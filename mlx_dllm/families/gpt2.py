"""GPT-2 family adapter: normalize modern-HF weight names to mlx-lm's layout."""

from __future__ import annotations

import mlx.core as mx

from mlx_dllm.families import FamilyAdapter, register


def _wrap_gpt2(model_class: type) -> type:
    """Return a GPT-2 subclass whose ``sanitize`` normalizes weight keys."""

    class Model(model_class):
        """GPT-2 with modern-HF weight names normalized to mlx-lm's layout.

        ``save_pretrained`` (what a2d writes; also distilgpt2) prefixes every
        key with "transformer." and omits the tied "lm_head.weight"; mlx-lm's
        gpt2 module expects the unprefixed legacy gpt2 layout.
        """

        def sanitize(self, weights):
            lm_head = weights.pop("lm_head.weight", None)
            weights = {k.removeprefix("transformer."): v for k, v in weights.items()}
            if lm_head is not None:
                wte = weights.get("wte.weight")
                if wte is None or not mx.array_equal(lm_head, wte):
                    raise ValueError(
                        "checkpoint has an untied lm_head.weight; mlx-lm's gpt2 "
                        "ties the output head to wte, so loading it would "
                        "silently produce wrong logits"
                    )
            return super().sanitize(weights)

    return Model


register(FamilyAdapter(model_type="gpt2", sanitize_wrapper=_wrap_gpt2))
