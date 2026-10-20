# SPDX-License-Identifier: Apache-2.0
"""Owned preparation: exact-model engine acquisition for control executions.

Engine-free. The pool's ``acquire_owned`` applies these refusals and returns an
``OwnedLease``; the preparation stage owns that lease and releases it exactly
once. Until owned admission exists, the stage prepares, releases and reports
that nothing reached the scheduler.
"""

from dataclasses import dataclass
import asyncio
import time

from .api.prism_control_coordinator import (
    DRAIN_WINDOW_SECONDS,
    EngineStageUnavailable,
    _settle,
)
from .host_submit import submission_failures
from .owned_admission import (
    OwnedAdmission,
    RunningCleaned,
    StreamSynchronized,
    WaitingCleaned,
)

# Settings that let a load serve something other than the configured engine,
# or that owned cleanup cannot stop safely (25b.4).
OWNED_SETTINGS_REFUSALS = (
    ("external_engine", "external_engine"),
    ("dflash_enabled", "dflash"),
    ("mtp_enabled", "mtp"),
    ("vlm_mtp_enabled", "vlm_mtp"),
    ("specprefill_enabled", "specprefill"),
    ("qwen35_ane_prefill_enabled", "ane_prefill"),
)
# Families whose admission rewrites their settings under memory pressure.
PRESSURE_OVERRIDE_FAMILIES = frozenset({"qwen4_exp", "deepseek_v41"})


def _resident_fraction(value):
    try:
        fraction = float(value)
    except (TypeError, ValueError):
        return False
    return 0.0 < fraction <= 1.0


# 28 (Flash-Next): the exact runtime signature owned work accepts for a
# pressure family, as canonical ``repr`` values. A missing, extra or duplicate
# key, or any other value, is a variant nobody reviewed. DeepSeek V4.1 has no
# entry: its loader can drop CED prefill under a signature that keeps it.
OWNED_PRESSURE_SIGNATURES = {
    "qwen4_exp": {
        "trust_remote_code": {"False"},
        "external_engine": {"None"},
        "index_cache_freq": {"None"},
        "embedding_audio_enabled": {"False"},
        "mtp_enabled": {"False"},
        "qwen4_ple_ssd_offload": {"True"},
        "qwen4_ple_gather_ahead": {"False"},
        "qwen4_gdn_decode_wide_proj": {"False"},
        "deepseek_v41_engram_ssd_offload": {"False"},
        "deepseek_v41_ced_prefill_enabled": {"False"},
        "turboquant_kv_enabled": {"False"},
        "qwen35_oq_a8_enabled": {"False"},
        "qwen35_ane_prefill_enabled": {"False"},
        "moe_expert_offload_enabled": {"True"},
        # Checked against the loaded caches (owned_qwen4).
        "moe_expert_offload_resident_fraction": _resident_fraction,
        "specprefill_enabled": {"False"},
        "dflash_enabled": {"False"},
        "vlm_mtp_enabled": {"False"},
    },
}


def owned_family(entry):
    return (getattr(entry, "config_model_type", None) or "").replace("-", "_").lower()


def owned_signature_refusal(family, signature):
    """``variant_unreviewed`` unless ``signature`` is exactly the family's allowlist."""
    allowed = OWNED_PRESSURE_SIGNATURES.get(family)
    if allowed is None:
        return "pressure_override"
    if signature is None:
        return "settings_unavailable"
    keys = [key for key, _ in signature]
    if len(keys) != len(set(keys)) or set(keys) != set(allowed):
        return "variant_unreviewed"
    for key, value in signature:
        rule = allowed[key]
        if not (rule(value) if callable(rule) else value in rule):
            return "variant_unreviewed"
    return None


def owned_loaded_variant_refusal(entry):
    """After the load: a pressure family's RECORDED signature, checked the same way."""
    family = owned_family(entry)
    if family not in PRESSURE_OVERRIDE_FAMILIES:
        return None
    return owned_signature_refusal(family, getattr(entry, "runtime_settings_signature", None))


class OwnedPreparationRefused(Exception):
    """Nothing was dispatched; ``reclaim`` is this failed load's reclaim task."""

    def __init__(self, reason, reclaim=None):
        super().__init__(reason)
        self.reason = reason
        self.reclaim = reclaim


class ReclaimUnfinished(Exception):
    """A failed load's device reclaim did not finish within the drain window."""


@dataclass(frozen=True)
class OwnedLease:
    model_id: str
    engine: object


def owned_entry_refusal(
    entry, settings, *, distributed, diffusion, signature=None, submit_failures=0
):
    """The pre-load refusal reason for an exact pool entry, or None.

    ``signature`` is a zero-argument callable returning the runtime signature
    of the persisted settings; it is called only for a pressure family.
    ``submit_failures`` is ``host_submit.submission_failures()``.
    """
    if entry is None:
        return "model_unknown"
    if entry.engine_type not in ("batched", "vlm"):
        return "engine_type"
    if getattr(entry, "vlm_fallback_rewritten", False):
        return "fallback_engine"
    if distributed:
        return "distributed"
    if diffusion:
        return "diffusion"
    if submit_failures:
        # A host read may run with no future to join (host_submit).
        return "host_read_ambiguous"
    family = owned_family(entry)
    if family in PRESSURE_OVERRIDE_FAMILIES and family not in OWNED_PRESSURE_SIGNATURES:
        return "pressure_override"
    if settings is None:
        return "settings_unavailable"
    for key, reason in OWNED_SETTINGS_REFUSALS:
        if getattr(settings, key, None):
            return reason
    if family in OWNED_PRESSURE_SIGNATURES:
        # After the retained refusals, so each keeps its own reason.
        return owned_signature_refusal(family, signature() if signature else None)
    return None


async def _join_reclaim(execution, reclaim, cancellations):
    """Wait for a failed load's reclaim within the shared drain deadline."""
    execution.drain_deadline = time.monotonic() + DRAIN_WINDOW_SECONDS
    if reclaim is None:
        return
    while not reclaim.done():
        remaining = execution.drain_deadline - time.monotonic()
        if remaining <= 0:
            raise ReclaimUnfinished()
        try:
            # Never cancels the reclaim; only waits for it.
            await asyncio.wait((reclaim,), timeout=remaining)
        except asyncio.CancelledError:
            cancellations.append(None)
    if reclaim.cancelled() or reclaim.exception() is not None:
        raise ReclaimUnfinished()


def make_preparation_stage(get_pool):
    """The 27.3 engine stage: prepare, release, report nothing dispatched.

    The pool is looked up per execution: the server creates it after the
    coordinator.
    """

    async def stage(execution):
        pool = get_pool()
        if pool is None:
            raise EngineStageUnavailable()
        cancellations = execution.cancellations
        try:
            task = asyncio.ensure_future(
                pool.acquire_owned(execution.model_id, fence=execution.verify)
            )
            try:
                # The settle loop always receives the result: a load is never
                # abandoned and its lease always reaches this one owner.
                execution.lease = await _settle(task, cancellations)
            except OwnedPreparationRefused as refusal:
                await _join_reclaim(execution, refusal.reclaim, cancellations)
                raise EngineStageUnavailable() from None
        finally:
            # Exactly one release, cleared first so it can never repeat, and
            # awaited through the settle loop so a cancellation never skips the
            # terminal after it.
            lease, execution.lease = execution.lease, None
            if lease is not None:
                await _settle(
                    asyncio.ensure_future(pool.release_engine(lease.model_id)),
                    cancellations,
                )
        # Owned admission (27.4) replaces this: nothing reached the scheduler.
        raise EngineStageUnavailable()

    return stage


# ---------------------------------------------------------------------------
# 27.4b: owned admission and streaming. 27.5: owned termination publishes a
# terminal only from facts observed on the handle; production wires this stage.

READ_SLICE_SECONDS = 0.25
ABSENCE_POLL_SECONDS = 0.02
_MEDIA_PART_TYPES = frozenset({"input_image", "input_file", "image_url", "file", "input_audio"})


class ExecutionUnresolved(Exception):
    """No terminal can be published: a stop proof is missing or late."""


def _is_error(output):
    return bool(getattr(output, "error", None)) or getattr(
        output, "finish_reason", None
    ) in ("error", "abort")


def owned_request_refusal(request, *, quantizing=False):
    """The reason owned work refuses a registered request, or None.

    Text-only work without tools or a response format; everything else is
    refused before any engine work.
    """
    if quantizing:
        return "quantizing"
    if getattr(request, "tools", None):
        return "tools"
    text = getattr(request, "text", None)
    if text is not None and getattr(text, "format", None) is not None:
        return "text_format"
    stack = [request.model_dump(mode="json").get("input")]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            if value.get("type") in _MEDIA_PART_TYPES:
                return "media_input"
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
    return None


def _unload_requested(pool, model_id):
    entry = pool.get_entry(model_id)
    return (
        entry is None
        or bool(getattr(entry, "pending_unload_reason", None))
        or bool(getattr(entry, "abort_requested", False))
    )


def _drain_deadline(execution):
    if execution.drain_deadline is None:
        execution.drain_deadline = time.monotonic() + DRAIN_WINDOW_SECONDS
    return execution.drain_deadline


def _start_cleanup(execution, core, handle):
    """Start the owned cleanup once; it publishes nothing in 27.4.

    The cleanup gets its own drain window from the moment it starts, never
    the admission's, so a cancel long after admission can still retire the
    request [F2]. Its first pin fixes the handle's stream deadline.
    """
    if execution.cleanup_task is not None:
        return
    execution.cleanup_task = asyncio.ensure_future(
        core._owned_cleanup_waiting(handle, time.monotonic() + DRAIN_WINDOW_SECONDS)
    )


async def _join(execution, core, handle, deadline):
    before = asyncio.current_task().cancelling()
    joined = await handle.join(core, deadline)
    if (
        joined is None
        and asyncio.current_task().cancelling() > before
        and time.monotonic() < deadline
    ):
        # A cancellation inside the join's wait: join once more [C3].
        execution.cancellations.append(None)
        joined = await handle.join(core, deadline)
    return joined


async def _read(execution, core, collector):
    """Emit text deltas until the output finishes or the reader must stop.

    Records and returns the final output, or returns None without one. The
    admission deadline bounds admission only: a generation ends at its
    output cap, or when Prism cancels it at its own job deadline.
    """
    stop_until = None
    while True:
        if getattr(core, "_closed", False) or getattr(core, "_closing", False):
            return None
        # An owned cleanup writes no collector output. Once its task has
        # ended, whatever the result, the cancel was already acknowledged and
        # there is nothing left to stream, even if a job that outlived the
        # task's wait still retires the request later.
        cleanup = execution.cleanup_task
        if cleanup is not None and cleanup.done():
            return None
        if execution.stop_requested:
            # Shutdown gets its own window from when the reader first sees it.
            if stop_until is None:
                stop_until = time.monotonic() + DRAIN_WINDOW_SECONDS
            elif time.monotonic() >= stop_until:
                return None
        output = collector.get_nowait()
        if output is None:
            try:
                await asyncio.wait_for(collector.ready.wait(), READ_SLICE_SECONDS)
            except TimeoutError:
                pass
            continue
        error = _is_error(output)
        if output.new_text and not error:
            execution.emit(
                "delta",
                {
                    "text": output.new_text,
                    "prompt_tokens": output.prompt_tokens,
                    "completion_tokens": output.completion_tokens,
                },
            )
        if output.finished:
            if not error:
                execution.emit(
                    "finished",
                    {
                        "finish_reason": output.finish_reason,
                        "prompt_tokens": output.prompt_tokens,
                        "completion_tokens": output.completion_tokens,
                    },
                )
            # No await since get_nowait(): a fence sees either no final
            # output and starts the cleanup, or this record and does not.
            execution.final_output = output
            return output


async def _admit(execution, pool, prepare_chat, submit_failures):
    model_id = execution.model_id
    engine = execution.lease.engine
    cancellations = execution.cancellations
    if _unload_requested(pool, model_id):
        raise EngineStageUnavailable()
    try:
        prepared = await prepare_chat(engine, execution.request, model_id)
        messages, chat_kwargs = prepared.messages, dict(prepared.chat_kwargs)
        if getattr(execution.request, "prompt_cache_retention", None) == "none":
            # Never stored at completion, as the admin benchmarks do: no cache
            # entry persists. A request longer than one cache block can still
            # write temporary boundary snapshots while it runs; the scheduler
            # removes them when it ends. A classifier request is unique and
            # never reused.
            chat_kwargs["skip_cache_store"] = True
        await engine.preflight_chat(messages, request_id=None, **chat_kwargs)
    except asyncio.CancelledError:
        # Nothing was dispatched; counted as the coordinator counts verify's.
        cancellations.append(None)
        raise EngineStageUnavailable() from None
    except Exception:
        # Nothing was dispatched: a template, context or prefill refusal.
        raise EngineStageUnavailable() from None
    if _unload_requested(pool, model_id):
        raise EngineStageUnavailable()
    core = pool._resolve_engine_core_from_engine(engine)
    request_id = execution.claim.request_id
    try:
        handle = OwnedAdmission(
            core,
            request_id,
            execution.deadline,
            # 28: a pressure family (Flash-Next) never re-runs owned work.
            retry=owned_family(pool.get_entry(model_id)) not in PRESSURE_OVERRIDE_FAMILIES,
            # 28: the executor dispatch admits nothing once this has moved.
            submit_failures=submit_failures,
        )
    except (ValueError, AttributeError):
        raise EngineStageUnavailable() from None

    def fence():
        handle.cancel()
        # A cancel after the output finished leaves termination to prove it.
        if execution.admitted and execution.final_output is None:
            _start_cleanup(execution, core, handle)

    # Publish before the final verify [slice 26 F2-1].
    execution.publish_fence(fence)
    try:
        await execution.verify()
    except asyncio.CancelledError:
        cancellations.append(None)
        raise EngineStageUnavailable() from None
    except Exception:
        raise EngineStageUnavailable() from None
    if _unload_requested(pool, model_id):
        raise EngineStageUnavailable()

    before = asyncio.current_task().cancelling()
    try:
        await engine.admit_owned_chat(
            messages, admission=handle, request_id=request_id, **chat_kwargs
        )
    except BaseException:
        # The outcome comes from the joined facts, never from the exception.
        if asyncio.current_task().cancelling() > before:
            cancellations.append(None)

    deadline = _drain_deadline(execution)
    joined = await _join(execution, core, handle, deadline)
    if handle.future is None or (joined is not None and not joined.insertion_attempted):
        await core.probe_owned_absence(handle, deadline)
        core.release_owned_failure(handle)
        raise EngineStageUnavailable()
    if joined is None:
        raise ExecutionUnresolved("admission_unjoinable")
    if joined.admission_failed:
        result = await core.probe_owned_absence(handle, deadline)
        if result == "absent":
            core.release_owned_failure(handle)
            fact = await core._owned_stream_barrier(handle, deadline)
            if type(fact) is StreamSynchronized:
                return "not_started", ("admission_refused", "stream_synchronized")
            raise ExecutionUnresolved("barrier_refused")
        if result not in ("present", "finished_before_cleanup"):
            raise ExecutionUnresolved("probe_unknown")
    # Admitted, on time or late. No await between the flag and the re-check.
    collector = core._output_collectors.get(request_id)
    execution.admitted = True
    if execution.cancel_requested:
        _start_cleanup(execution, core, handle)
    if collector is None:
        raise ExecutionUnresolved("collector_missing")
    final = await _read(execution, core, collector)
    # One window from here bounds the absence wait, the barrier and finish.
    execution.drain_deadline = time.monotonic() + DRAIN_WINDOW_SECONDS
    return await _terminate(execution, core, handle, final, submit_failures)


async def _absent(execution, core, handle, deadline):
    """Probe until the request is gone from the scheduler, within the window."""
    cancellations = execution.cancellations
    while True:
        before = asyncio.current_task().cancelling()
        result = await core.probe_owned_absence(handle, deadline)
        if asyncio.current_task().cancelling() > before:
            cancellations.append(None)
            if time.monotonic() < deadline:
                continue
        if result == "absent":
            return
        # A probe cut off by the window's end answers unknown: label it so.
        if time.monotonic() >= deadline:
            raise ExecutionUnresolved("absence_unproved")
        if result not in ("present", "finished_before_cleanup"):
            raise ExecutionUnresolved("probe_unknown")
        try:
            await asyncio.sleep(ABSENCE_POLL_SECONDS)
        except asyncio.CancelledError:
            cancellations.append(None)


async def _terminate(execution, core, handle, final, submit_failures):
    """27.5: the terminal, only from facts observed on this handle.

    28: every admitted execution leaves through here. If a host-read
    submission failed since preparation, a read may still run with no future
    to join, so no outcome is published.
    """
    outcome = await _terminal_facts(execution, core, handle, final)
    if submission_failures() != submit_failures:
        raise ExecutionUnresolved("host_read_ambiguous")
    return outcome


async def _terminal_facts(execution, core, handle, final):
    if getattr(core, "_closed", False) or getattr(core, "_closing", False):
        raise ExecutionUnresolved("core_closing")
    deadline = execution.drain_deadline
    request_id = execution.claim.request_id
    task = execution.cleanup_task
    if task is not None:
        # A cancel or the deadline: only a retired request is cancelled.
        fact = await _settle(task, execution.cancellations)
        if type(fact) not in (WaitingCleaned, RunningCleaned):
            raise ExecutionUnresolved("cleanup_without_fact")
        # An owned cleanup never marks the request finished, so the reaper
        # never drops this collector; nothing reads it any more.
        core._cleanup_request(request_id)
        if type(await core._owned_stream_barrier(handle, deadline)) is not StreamSynchronized:
            raise ExecutionUnresolved("barrier_refused")
        cleaned = "waiting_cleaned" if type(fact) is WaitingCleaned else "running_cleaned"
        return "cancelled", (cleaned, "stream_synchronized")
    if final is None:
        raise ExecutionUnresolved("no_final_output")
    await _absent(execution, core, handle, deadline)
    # The barrier's one job is submitted only now, after absence is proved.
    if type(await core._owned_stream_barrier(handle, deadline)) is not StreamSynchronized:
        raise ExecutionUnresolved("barrier_refused")
    # The request was marked finished; the orphan reaper drops its collector.
    if _is_error(final):
        return "failed", ("error_output", "absent_proved", "stream_synchronized")
    return "completed", ("finished_output", "absent_proved", "stream_synchronized")


def make_admission_stage(get_pool, prepare_chat, refusal):
    """The engine stage: preparation, owned admission, streaming and termination.

    ``prepare_chat(engine, request, model_id)`` is the server's shared
    Responses preparation with MCP exposure off; ``refusal(request)`` returns
    the reason owned work refuses the request, or None.
    """

    async def stage(execution):
        pool = get_pool()
        if (
            pool is None
            or execution.request is None
            or refusal(execution.request) is not None
        ):
            raise EngineStageUnavailable()
        cancellations = execution.cancellations
        # Read before acquisition, which refuses when it is already non-zero.
        submit_failures = submission_failures()
        try:
            task = asyncio.ensure_future(
                pool.acquire_owned(execution.model_id, fence=execution.verify)
            )
            try:
                execution.lease = await _settle(task, cancellations)
            except OwnedPreparationRefused as refused:
                await _join_reclaim(execution, refused.reclaim, cancellations)
                raise EngineStageUnavailable() from None
            return await _admit(execution, pool, prepare_chat, submit_failures)
        finally:
            lease, execution.lease = execution.lease, None
            if lease is not None:
                await _settle(
                    asyncio.ensure_future(pool.release_engine(lease.model_id)),
                    cancellations,
                )

    return stage
