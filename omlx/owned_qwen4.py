# SPDX-License-Identifier: Apache-2.0
"""28: loaded-instance checks for owned work on Qwen4-Exp (Flash-Next).

The recorded runtime signature says what was requested; these checks read
what was built. Every PLE table must be SSD-backed from this model's
directory, with gather-ahead off and never submitted, and MoE expert offload
must cover every expert layer at the configured capacity. The settings and
the module-level PLE globals are never consulted: a later Qwen4-Exp load
overwrites the globals, and only the installed instances describe this
engine. Runs on the engine's MLX thread.
"""

from pathlib import Path


def _under(path, root):
    """A file inside ``root`` (nested shard folders included), never root itself."""
    return path != root and path.is_relative_to(root)


def qwen4_instance_refusal(
    vlm_model,
    *,
    model_dir,
    fraction,
    wrapped,
    disk_table_type,
    iter_switch_glus,
    is_mtp_path,
    offload_caches,
    minimum_experts,
):
    """The first mismatch between the loaded model and owned work's needs, or None."""
    root = Path(model_dir).expanduser().resolve()
    language = getattr(vlm_model, "language_model", None)
    layers = getattr(getattr(language, "model", None), "layers", None) or ()
    tables = [
        layer.ple.ple_embedding.ngram_embedding
        for layer in layers
        if getattr(layer, "ple", None) is not None
    ]
    if not tables:
        return "ple_unobserved"
    for table in tables:
        if type(table) is not disk_table_type:
            return "ple_resident"
        if table.gather_ahead is not False or table.prefetch_submissions != 0:
            return "ple_gather_ahead"
        readers = list(getattr(table, "_readers", {}).values())
        if not readers or not all(
            _under(Path(reader.path).resolve(), root) for reader in readers
        ):
            return "ple_foreign_path"
    if any(not is_mtp_path(path) for *_, path in iter_switch_glus(vlm_model)):
        return "moe_unwrapped"
    caches = offload_caches(vlm_model)
    if not caches or len(caches) != wrapped:
        return "moe_coverage"
    minimum = minimum_experts(root)
    for cache in caches:
        n = cache.n_experts
        # full_capacity, not capacity: a prefill shrinks the slots to the floor.
        if cache.full_capacity != min(n, max(minimum, round(n * fraction))):
            return "moe_capacity"
    return None


def qwen4_instance_refusal_for_engine(engine):
    """The check on a VLM engine, with the real table type and offload helpers."""
    vlm_model = getattr(engine, "_vlm_model", None)
    settings = getattr(engine, "_model_settings", None)
    if vlm_model is None or settings is None:
        return "engine_unavailable"
    from mlx_vlm.models.qwen4_exp.language import DiskBackedShardedEmbedding

    from .patches.moe_expert_offload import (
        _is_mtp_path,
        _iter_switch_glus,
        _minimum_experts,
        moe_offload_caches,
    )

    return qwen4_instance_refusal(
        vlm_model,
        model_dir=engine._model_name,
        # The fraction the engine wrapped with (engine/vlm.py).
        fraction=float(
            getattr(settings, "moe_expert_offload_resident_fraction", 0.25)
        ),
        wrapped=int(getattr(engine, "_moe_offload_wrapped", 0) or 0),
        disk_table_type=DiskBackedShardedEmbedding,
        iter_switch_glus=_iter_switch_glus,
        is_mtp_path=_is_mtp_path,
        offload_caches=moe_offload_caches,
        minimum_experts=_minimum_experts,
    )
