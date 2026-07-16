"""a2d run-dir loading: resolve the consumable checkpoint, honor the manifest,
and let a run-dir's own a2d block drive ``generate``/``denoise`` defaults.

The fixtures are assembled in ``tmp_path`` from the vendored, tiny random-weight
Gemma 3 checkpoint under ``tests/fixtures/gemma3-a2d-sample`` (hidden 16, vocab
65, a2d-block ``mask_token_id`` 64, sampler ``canvas_len``/``num_steps`` 8). That
checkpoint is ``final_alpha=0.5`` (structure only, not quality), so these tests
assert structural behavior - loads, resolves, runs, reveals every mask, preserves
the prompt - never output quality or numeric parity. Nothing here is
Gemma-specific: the loader is model-agnostic and never inspects the family.
"""

import json
from pathlib import Path

import mlx.core as mx
import pytest

import mlx_dllm
from mlx_dllm.runtime import _parse_manifest, _resolve_run_dir

FIXTURE = Path(__file__).parent / "fixtures" / "gemma3-a2d-sample"
MASK_TOKEN_ID = 64  # from the checkpoint's a2d block
CANVAS_LEN = 8  # a2d sampler canvas_len / num_steps in the fixture config


def _make_manifest(mask_token_id=None):
    """A run-dir manifest.json shaped like a2d's real output.

    ``model_spec.mask_token_id`` is the PRE-conversion detected value (often
    ``null`` in real runs) and must never be used as the runtime mask id. An
    unknown nested key is included so parsing is proven tolerant of extra keys.
    """
    return {
        "schema_version": "0.1.0",
        "a2d_version": "0.1.0",
        "job_id": "test-job-0001",
        "status": "completed",
        "model_spec": {
            "model_type": "gemma3_text",
            "mask_token_id": mask_token_id,
            "capabilities": ["paradigm.ar-transformer", "attn.swa"],
        },
        "conversion_config": {"objective": "mdlm", "anneal_steps": 2},
        "future_unknown_field": {"nested": [1, 2, 3]},
    }


def _populate(dest, src=FIXTURE, *, drop_a2d=False):
    """Materialize an HF checkpoint at ``dest`` from ``src``.

    ``config.json`` is rewritten (optionally dropping the ``a2d`` block, to mimic
    a raw trainer ``checkpoint-<N>/``); every other file is symlinked so the tiny
    fixture weights are never duplicated on disk.
    """
    dest.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if not f.is_file():
            continue
        if f.name == "config.json":
            config = json.loads(f.read_text())
            if drop_a2d:
                config.pop("a2d", None)
            (dest / f.name).write_text(json.dumps(config))
        else:
            (dest / f.name).symlink_to(f.resolve())


def _run_dir_with_model(tmp_path, *, manifest_mask=None, add_checkpoints=()):
    root = tmp_path / "run"
    _populate(root / "model")
    (root / "manifest.json").write_text(json.dumps(_make_manifest(manifest_mask)))
    for number in add_checkpoints:
        (root / "checkpoints" / f"checkpoint-{number}").mkdir(parents=True)
    return root


def _run_dir_checkpoints_only(tmp_path, numbers, *, loadable=False):
    root = tmp_path / "run"
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(_make_manifest()))
    for number in numbers:
        dest = root / "checkpoints" / f"checkpoint-{number}"
        if loadable:
            _populate(dest, drop_a2d=True)
        else:
            dest.mkdir(parents=True)
    return root


# --- resolution + manifest parsing (no model load) --------------------------


def test_resolve_prefers_model_subdir(tmp_path):
    root = _run_dir_with_model(tmp_path, add_checkpoints=(2,))
    checkpoint, manifest = _resolve_run_dir(str(root))
    assert checkpoint == root / "model"
    assert manifest == _make_manifest()


def test_resolve_falls_back_to_latest_checkpoint(tmp_path):
    # Numeric max, not lexical: checkpoint-10 must beat checkpoint-2/-9.
    root = _run_dir_checkpoints_only(tmp_path, (1, 2, 9, 10))
    checkpoint, manifest = _resolve_run_dir(str(root))
    assert checkpoint == root / "checkpoints" / "checkpoint-10"
    assert manifest == _make_manifest()


def test_resolve_plain_hf_dir_is_not_run_dir(tmp_path):
    # No manifest.json and no model/ subdir -> falls through to stock loading.
    plain = tmp_path / "plain"
    _populate(plain)
    assert _resolve_run_dir(str(plain)) == (None, None)


def test_resolve_plain_hf_dir_with_stray_manifest_falls_through(tmp_path):
    # A root config.json marks a plain HF checkpoint even when a stray
    # manifest.json sits next to it (a real run-dir root never has one).
    plain = tmp_path / "plain"
    _populate(plain)
    (plain / "manifest.json").write_text(json.dumps(_make_manifest()))
    assert _resolve_run_dir(str(plain)) == (None, None)


def test_resolve_repo_id_is_not_run_dir():
    # A non-local path (hub repo id) is never a run-dir.
    assert _resolve_run_dir("distilbert/distilgpt2") == (None, None)


def test_resolve_skips_non_decimal_checkpoint_names(tmp_path):
    # Suffixes int() rejects (superscripts, words) are ignored, not a crash.
    root = _run_dir_checkpoints_only(tmp_path, (1,))
    (root / "checkpoints" / "checkpoint-²").mkdir()
    (root / "checkpoints" / "checkpoint-final").mkdir()
    checkpoint, _ = _resolve_run_dir(str(root))
    assert checkpoint == root / "checkpoints" / "checkpoint-1"


def test_resolve_run_dir_without_model_or_checkpoints_raises(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps(_make_manifest()))
    with pytest.raises(FileNotFoundError):
        _resolve_run_dir(str(root))


def test_manifest_tolerant_of_null_mask_and_extra_keys(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_make_manifest(mask_token_id=None)))
    manifest = _parse_manifest(path)
    assert manifest["model_spec"]["mask_token_id"] is None
    assert manifest["future_unknown_field"] == {"nested": [1, 2, 3]}


def test_parse_missing_manifest_returns_none(tmp_path):
    assert _parse_manifest(tmp_path / "does-not-exist.json") is None


def test_parse_non_dict_manifest_returns_none(tmp_path):
    # Valid JSON that is not an object violates the provenance-dict contract.
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(["not", "a", "dict"]))
    assert _parse_manifest(path) is None


# --- loading through the run-dir root ---------------------------------------


def test_load_run_dir_root_resolves_model(tmp_path):
    root = _run_dir_with_model(tmp_path, manifest_mask=None)
    model, tokenizer, a2d = mlx_dllm.load(str(root))
    assert type(model).__module__ == "mlx_lm.models.gemma3_text"
    assert a2d is not None
    assert a2d.mask_token_id == MASK_TOKEN_ID  # from the config a2d block
    assert a2d.sampler == {"canvas_len": 8, "num_steps": 8, "temperature": 1.0}
    # Manifest surfaced as provenance, tolerant of its null mask id.
    assert a2d.manifest == _make_manifest(None)
    assert a2d.manifest["model_spec"]["mask_token_id"] is None
    # The loaded model actually runs.
    logits = mlx_dllm.bidirectional_forward(model, mx.array([tokenizer.encode("hi")]))
    assert logits.shape[0] == 1


def test_a2d_block_mask_id_wins_over_manifest(tmp_path):
    # Manifest carries a DIFFERENT (non-null) mask id; the config a2d block wins.
    root = _run_dir_with_model(tmp_path, manifest_mask=7)
    _, _, a2d = mlx_dllm.load(str(root))
    assert a2d.manifest["model_spec"]["mask_token_id"] == 7
    assert a2d.mask_token_id == MASK_TOKEN_ID


def test_load_fallback_to_checkpoint(tmp_path):
    # No model/ subdir: load resolves the latest checkpoint-<N>. That raw trainer
    # checkpoint has no a2d block, so a2d is None (honest degraded result).
    root = _run_dir_checkpoints_only(tmp_path, (1, 2), loadable=True)
    checkpoint, _ = _resolve_run_dir(str(root))
    assert checkpoint == root / "checkpoints" / "checkpoint-2"
    model, tokenizer, a2d = mlx_dllm.load(str(root))
    assert a2d is None
    logits = mlx_dllm.bidirectional_forward(model, mx.array([tokenizer.encode("hi")]))
    assert logits.shape[0] == 1


def test_load_plain_hf_dir_backward_compat():
    # The vendored dir (no manifest, no model/ subdir) still loads exactly as
    # before: a2d parsed from the config block, no manifest attached.
    _, _, a2d = mlx_dllm.load(str(FIXTURE))
    assert a2d is not None
    assert a2d.mask_token_id == MASK_TOKEN_ID
    assert a2d.manifest is None


def test_load_plain_hf_dir_with_stray_manifest(tmp_path):
    # A stray manifest.json must not reroute a loadable checkpoint through the
    # run-dir path (which would raise): stock loading, manifest never attached.
    plain = tmp_path / "plain"
    _populate(plain)
    (plain / "manifest.json").write_text(json.dumps(_make_manifest()))
    _, _, a2d = mlx_dllm.load(str(plain))
    assert a2d is not None
    assert a2d.mask_token_id == MASK_TOKEN_ID
    assert a2d.manifest is None


# --- generate/denoise driven by the run-dir's own a2d defaults --------------


@pytest.fixture(scope="module")
def loaded_run_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("run-dir")
    _populate(root / "model")
    (root / "manifest.json").write_text(json.dumps(_make_manifest()))
    return mlx_dllm.load(str(root))


def test_denoise_uses_a2d_defaults(loaded_run_dir):
    model, tokenizer, a2d = loaded_run_dir
    prompt_ids = tokenizer.encode("Hello")
    canvas = mx.array([[*prompt_ids, *([MASK_TOKEN_ID] * CANVAS_LEN)]])
    # No explicit mask_token_id / steps: both come from the run-dir's a2d block.
    output = mlx_dllm.denoise(model, canvas, a2d=a2d)
    assert output.shape == canvas.shape
    assert output[0, : len(prompt_ids)].tolist() == prompt_ids
    assert MASK_TOKEN_ID not in output[0].tolist()


def test_generate_uses_a2d_defaults(loaded_run_dir):
    model, tokenizer, a2d = loaded_run_dir
    # max_new_tokens, mask_token_id, and steps all default from a2d.
    text = mlx_dllm.generate(model, tokenizer, "Hello", a2d=a2d)
    assert isinstance(text, str)
    # Same result as spelling the run-dir defaults out explicitly.
    explicit = mlx_dllm.generate(
        model,
        tokenizer,
        "Hello",
        max_new_tokens=CANVAS_LEN,
        mask_token_id=MASK_TOKEN_ID,
        steps=CANVAS_LEN,
    )
    assert text == explicit


def test_explicit_mask_token_id_overrides_a2d(loaded_run_dir):
    model, tokenizer, a2d = loaded_run_dir
    prompt_ids = tokenizer.encode("Hello")
    canvas = mx.array([[*prompt_ids, *([MASK_TOKEN_ID] * CANVAS_LEN)]])
    # Explicit mask id 63 (valid, absent from the canvas) overrides a2d's 64,
    # so denoise finds nothing to reveal and returns the canvas untouched.
    output = mlx_dllm.denoise(model, canvas, mask_token_id=63, a2d=a2d)
    assert output.tolist() == canvas.tolist()
    assert MASK_TOKEN_ID in output[0].tolist()


def test_explicit_max_new_tokens_overrides_a2d(loaded_run_dir):
    model, tokenizer, a2d = loaded_run_dir
    # Explicit 0 overrides a2d's canvas_len (8) and reaches the positivity guard.
    with pytest.raises(ValueError, match="positive"):
        mlx_dllm.generate(model, tokenizer, "Hello", max_new_tokens=0, a2d=a2d)


def test_generate_requires_mask_token_id(loaded_run_dir):
    model, tokenizer, _ = loaded_run_dir
    with pytest.raises(ValueError, match="mask_token_id"):
        mlx_dllm.generate(model, tokenizer, "Hello", max_new_tokens=4)


def test_denoise_requires_mask_token_id(loaded_run_dir):
    model, tokenizer, _ = loaded_run_dir
    canvas = mx.array([tokenizer.encode("Hello")])
    with pytest.raises(ValueError, match="mask_token_id"):
        mlx_dllm.denoise(model, canvas, steps=2)
