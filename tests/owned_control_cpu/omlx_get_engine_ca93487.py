# SPDX-License-Identifier: Apache-2.0
# Verbatim EnginePool.get_engine from oMLX ca93487 (before slice 27.3 moved
# its locked part into _get_engine_locked). The differential baseline for
# test_provider_owned_preparation.py; never edited by hand.
from __future__ import annotations


class OldEnginePoolGetEngine:
    async def get_engine(
        self,
        model_id: str,
        force_lm: bool = False,
        _lease: bool = False,
        runtime_settings: object | None = None,
    ) -> (
        BaseEngine
        | EmbeddingEngine
        | RerankerEngine
        | STTEngine
        | STSEngine
        | TTSEngine
    ):
        """
        Get or load engine for the specified model.

        This method implements pre-load memory checking:
        1. Check if model is already loaded -> return immediately
        2. Check if model is too large for memory limit -> raise error
        3. Evict LRU models until there's enough space
        4. Load the model
        5. Return the engine

        Args:
            model_id: The model ID to get engine for
            force_lm: Force loading as LM (BatchedEngine) even for VLM models.
                Useful for text-only tasks like accuracy benchmarks.
            runtime_settings: Optional transient settings used for this engine
                load. When its engine-construction signature differs from the
                currently loaded engine, the old engine is unloaded and the new
                variant is loaded without mutating persisted model settings.

        Returns:
            The loaded engine (BaseEngine for LLM, EmbeddingEngine for embeddings)

        Raises:
            ModelNotFoundError: If model is not discovered
            ModelTooLargeError: If model exceeds memory limit
            InsufficientMemoryError: If can't free enough memory (all pinned)
            ModelLoadingError: If model is already being loaded
        """
        entry = self._entries.get(model_id)
        if force_lm and entry is not None and not self._has_mlx_lm_path(entry):
            # The VLM engine is the only text engine for these families.
            force_lm = False
        reload_failed = await self._reloadable_failed_engine(model_id)
        await self._check_splash_launcher(model_id, runtime_settings)
        ready = self._acquire_loaded_engine(
            model_id, force_lm, _lease, runtime_settings, reload_failed
        )
        if ready is not None:
            return ready
        async with self._lock:
            entry = self._entries.get(model_id)
            if not entry:
                raise ModelNotFoundError(model_id, list(self._entries.keys()))
            if entry.pending_unload_reason:
                raise ModelBusyError(model_id, "start work while unload is pending")
            expected_signature = self._engine_runtime_signature(
                model_id,
                runtime_settings,
            )
            ngram_admission_ceiling = None
            if (entry.config_model_type or "").replace("-", "_").lower() in {
                "qwen4_exp",
                "deepseek_v41",
            }:
                candidate = self._current_ceiling()
                if candidate <= 0:
                    candidate = self._fallback_admission_ceiling()
                if candidate > 0:
                    ngram_admission_ceiling = candidate
            unloaded_for_admission = False

            if (
                reload_failed
                and entry.engine is not None
                and self._runtime_failed_engine(entry.engine)
            ):
                logger.warning(
                    "Distributed runtime for %s is no longer serviceable (%s); "
                    "unloading it so this request reloads the deployment",
                    model_id,
                    getattr(entry.engine, "runtime_failed_reason", ""),
                )
                await self._unload_engine(model_id)
                unloaded_for_admission = True

            # Already loaded - just update access time
            if entry.engine is not None:
                if (
                    expected_signature is not None
                    and entry.runtime_settings_signature is not None
                    and entry.runtime_settings_signature != expected_signature
                ) or (
                    runtime_settings is not None
                    and entry.runtime_settings_signature is None
                ):
                    self._raise_if_reload_busy(
                        entry,
                        "reload runtime settings variant",
                    )
                    logger.info(
                        "Runtime settings variant changed for %s; "
                        "unloading before reload.",
                        model_id,
                    )
                    await self._unload_engine(model_id)
                    unloaded_for_admission = True
                # If force_lm requested but current engine is VLM, unload and reload
                if (
                    entry.engine is not None
                    and force_lm
                    and isinstance(entry.engine, VLMBatchedEngine)
                ):
                    self._raise_if_reload_busy(entry, "reload as LM")
                    logger.info(
                        f"Unloading VLM engine for {model_id} "
                        f"(force_lm=True, reloading as LM)"
                    )
                    await self._unload_engine(model_id)
                    unloaded_for_admission = True
                elif entry.engine is not None:
                    self._validate_llm_engine_ready(model_id, entry.engine)
                    if entry.runtime_settings_signature is None:
                        entry.runtime_settings_signature = expected_signature
                    entry.last_access = time.time()
                    if _lease:
                        entry.in_use += 1
                    return entry.engine

            self._raise_if_model_path_missing_locked(model_id, entry)
            self._raise_if_load_failed(model_id, entry)

            # Pre-load admission against the memory ceiling from the
            # process memory enforcer (min of static and dynamic). Try
            # evicting LRU non-pinned models first; if the model still
            # cannot fit after evicting everything available, raise.
            #
            # Eviction starts at the enforcer's *soft* watermark, not the
            # ceiling (#2319): the soft..ceiling band is exactly the
            # hard-pressure zone, and a second model admitted into it kept
            # both models resident through the load (swapping for minutes)
            # only to have the first request's prefill guard evict the old
            # one anyway. Evicting down to the same soft target *before*
            # the new weights allocate fixes the ordering; refusing a load
            # still requires exceeding the ceiling.
            #
            # ceiling == 0 means the guard is disabled or the enforcer is
            # not wired up. Eviction on model swap must not die with the
            # guard (#2290): fall back to the best-effort admission
            # ceiling (static, guard-independent) and keep evicting, but
            # never refuse the load under it — with the guard off the
            # user opted out of hard limits.
            # A distributed coordinator admits only rank zero's planned shard,
            # not the complete model. A local VLM-shaped checkpoint served by
            # the text engine (force_lm or a model_type_override that flipped
            # engine_type to "batched") loads only its language weights, so
            # admit that path by the text-only estimate instead of the
            # vision-inclusive file size (#2385).
            deployment = self._distributed_deployment_for_entry(entry)
            admission_size = self._entry_resident_size(entry)
            if (
                deployment is None
                and entry.text_only_size
                and (force_lm or entry.engine_type == "batched")
            ):
                admission_size = entry.text_only_size
            admission_settings = runtime_settings
            if admission_settings is None and self._settings_manager is not None:
                get_settings = getattr(self._settings_manager, "get_settings", None)
                if callable(get_settings):
                    admission_settings = get_settings(model_id)
            load_settings = self._effective_qwen4_model_settings(
                entry,
                admission_settings,
                ceiling=ngram_admission_ceiling,
            )
            load_settings = self._effective_deepseek_v41_model_settings(
                entry, load_settings, ceiling=ngram_admission_ceiling
            )
            ngram_admission_override = load_settings is not admission_settings
            runtime_load_settings = (
                load_settings if ngram_admission_override else runtime_settings
            )
            admission_size = self._entry_runtime_resident_size(
                entry,
                load_settings,
                base_size=admission_size,
            )
            admission_kind = "local shard" if deployment is not None else "model"

            ceiling = self._current_ceiling()
            best_effort = False
            if ceiling <= 0:
                ceiling = self._fallback_admission_ceiling()
                best_effort = ceiling > 0
            if ceiling > 0:
                soft_target = self._admission_soft_target()
                evict_target = min(soft_target, ceiling) if soft_target > 0 else ceiling
                evicted_any = unloaded_for_admission
                ceiling_rechecks = 0
                while True:
                    # Consult the tracked accumulator alongside live memory:
                    # after a model settles or idles, mx.get_active_memory() and
                    # the process footprint can read well below the model's true
                    # resident size, while _current_model_memory still reflects
                    # the committed total. Using only live memory lets a second
                    # large model load without evicting the first, over-
                    # committing past the ceiling (#1623).
                    current = max(
                        mx.get_active_memory(),
                        _settled_phys_footprint(),
                        self._current_model_memory,
                    )
                    projected = current + admission_size
                    if projected <= evict_target:
                        break
                    victim = self._find_lru_victim()
                    if victim is not None:
                        logger.info(
                            f"Evicting '{victim}' to fit '{model_id}' "
                            f"under the admission soft target "
                            f"({format_size(projected)} > "
                            f"{format_size(evict_target)})"
                        )
                        await self._unload_engine(victim)
                        evicted_any = True
                        continue
                    if projected <= ceiling:
                        # Above the soft target with nothing left to
                        # evict, but still under the ceiling: admit. The
                        # soft target only decides when eviction starts
                        # (#2319); refusal keeps the ceiling-only
                        # contract.
                        if evict_target < ceiling:
                            logger.info(
                                f"Admitting '{model_id}' above the "
                                f"admission soft target with no idle "
                                f"model left to evict "
                                f"({format_size(projected)} > "
                                f"{format_size(evict_target)}, ceiling "
                                f"{format_size(ceiling)})"
                            )
                        break
                    failure_current = current
                    failure_projected = projected
                    failure_label = "current"

                    if evicted_any:
                        # Nothing else to evict after unloading at least one
                        # model in this get_engine() call. Before failing,
                        # re-test against the *tracked committed* baseline.
                        # The phys_footprint term folded into `current` is the
                        # macOS kernel ledger, which can still count
                        # reclaimable residue from models we just evicted.
                        # Pinned/in-use models that could not be evicted remain
                        # counted in _current_model_memory, preserving the
                        # #1623 undercount guard. Without a local eviction,
                        # keep trusting phys_footprint because it may be
                        # unrelated process pressure rather than model residue.
                        committed = max(
                            mx.get_active_memory(), self._current_model_memory
                        )
                        committed_projected = committed + admission_size
                        if committed_projected <= ceiling:
                            logger.info(
                                f"Admitting '{model_id}': committed baseline "
                                f"{format_size(committed_projected)} fits ceiling "
                                f"{format_size(ceiling)} "
                                f"(live footprint {format_size(projected)} included "
                                "reclaimable residue from evicted models)"
                            )
                            break
                        failure_current = committed
                        failure_projected = committed_projected
                        failure_label = "committed"

                    if best_effort:
                        # Memory guard is off: evicting was all we could
                        # do. Admit over the static ceiling instead of
                        # refusing, matching the unguarded no-hard-limit
                        # contract.
                        logger.warning(
                            f"Loading '{model_id}' past the static memory "
                            f"ceiling with the memory guard disabled "
                            f"(projected {format_size(failure_projected)} > "
                            f"ceiling {format_size(ceiling)}, "
                            f"{failure_label} baseline) and nothing left to "
                            f"evict; the system may swap heavily."
                        )
                        break

                    if (
                        ceiling_rechecks < _ADMISSION_CEILING_RECHECKS
                        and self._dynamic_ceiling_binds()
                    ):
                        # Pages of a model unloaded just before this load
                        # reach the free list up to ~0.2s after the process
                        # footprint drops, so the dynamic ceiling can read
                        # low for that long. Re-read it before refusing.
                        ceiling_rechecks += 1
                        await asyncio.sleep(_ADMISSION_CEILING_RECHECK_S)
                        ceiling = max(ceiling, self._current_ceiling())
                        soft_target = max(soft_target, self._admission_soft_target())
                        evict_target = (
                            min(soft_target, ceiling) if soft_target > 0 else ceiling
                        )
                        continue

                    # Still over budget under the applicable baseline. Use
                    # ModelTooLargeError when the model alone exceeds the
                    # ceiling (no chance of fitting), InsufficientMemoryError
                    # when current usage leaves no room.
                    if admission_size > ceiling:
                        binding, advice = self._ceiling_binding_and_advice(
                            ceiling=ceiling,
                            current=failure_current,
                            tail="use a smaller model",
                        )
                        raise ModelTooLargeError(
                            model_id,
                            admission_size,
                            ceiling,
                            binding=binding,
                            advice=advice,
                        )
                    binding, advice = self._ceiling_binding_and_advice(
                        ceiling=ceiling,
                        current=failure_current,
                        tail="unload another model",
                    )
                    label = f"{binding} memory ceiling" if binding else "memory ceiling"
                    raise InsufficientMemoryError(
                        required=admission_size,
                        current=failure_current,
                        message=(
                            f"Cannot load {model_id}: projected memory "
                            f"{format_size(failure_projected)} would exceed "
                            f"the {label} {format_size(ceiling)} "
                            f"({failure_label}: {format_size(failure_current)}, "
                            f"{admission_kind}: {format_size(admission_size)}). "
                            f"{advice or DEFAULT_CEILING_ADVICE}."
                        ),
                    )

            # Now load the model
            await self._load_engine(
                model_id,
                force_lm=force_lm,
                runtime_settings=runtime_load_settings,
            )

            loaded = self._entries[model_id]
            if ngram_admission_override and expected_signature is not None:
                # Automatic mmap is local to this admission attempt. Keep the
                # user's requested variant as the reuse key so the next request
                # does not reload the model merely because pressure recovered.
                loaded.runtime_settings_signature = expected_signature
            self._validate_llm_engine_ready(model_id, loaded.engine)
            if _lease:
                loaded.in_use += 1
            return loaded.engine
