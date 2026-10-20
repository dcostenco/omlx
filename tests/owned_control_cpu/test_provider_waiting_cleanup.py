# SPDX-License-Identifier: Apache-2.0
"""Queued-only cleanup proof: actual scheduler/core methods and CPU executor."""

import ast
import asyncio
from collections import deque
from dataclasses import dataclass
import logging
from pathlib import Path
import time
from types import MethodType, SimpleNamespace
from typing import Any, Optional
import unittest
from unittest.mock import Mock

import test_provider_stream_barrier as stream_fixture
from omlx.owned_admission import OwnedAdmission, RunningCleaned, WaitingCleaned
from omlx.request import Request, RequestStatus, RequestOutput

ROOT = stream_fixture.admission_fixture.ROOT
SOURCE = ROOT / "omlx/scheduler.py"
tree = ast.parse(SOURCE.read_text())
cls = next(
    n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler"
)
NAMES = {
    "add_request",
    "_cleanup_owned_waiting",
    "_retain_consume_owned_waiting",
    "_invalidate_owned_waiting",
    "_owned_waiting_fenced",
    "_select_unfenced_waiting",
    "_clear_memory_admission_blocker",
    "_clear_store_cache_admission_blocker",
    "_prepare_prefix_cache_for_request",
    "_schedule_waiting",
    # 25b.4: the single classified entry the core now calls.
    "_cleanup_owned_request",
    "_owned_admission_complete",
    "_owned_running_refusal",
    # 27.3: the model-level part, shared with owned preparation.
    "_owned_model_refusal",
    "_owned_running_state",
    "_cleanup_owned_running",
}
methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in NAMES]
assert len(methods) == len(NAMES)
constants = [
    n
    for n in cls.body
    if isinstance(n, ast.Assign)
    and any(
        isinstance(t, ast.Name)
        and t.id in {"_MAX_OWNED_WAITING_PROOFS", "_OWNED_PREFIX_FENCED"}
        for t in n.targets
    )
]
proof_class = next(
    n
    for n in tree.body
    if isinstance(n, ast.ClassDef) and n.name == "_WaitingCleanupProof"
)
namespace = dict(
    Any=Any,
    Optional=Optional,
    dataclass=dataclass,
    Request=Request,
    RequestOutput=RequestOutput,
    RequestStatus=RequestStatus,
    OwnedAdmission=OwnedAdmission,
    WaitingCleaned=WaitingCleaned,
    RunningCleaned=RunningCleaned,
    deque=deque,
    time=time,
    logger=logging.getLogger("waiting-proof"),
    # Running-path helpers; queued tests never reach them with real effects.
    _PrefillState=type("_PrefillState", (), {}),
    _mtp_priming=SimpleNamespace(_eligible_host=lambda model: None),
    _model_mtp_decode_enabled=lambda model: False,
    _block_drafter_for=lambda model: None,
    _unregister_uid_row=lambda model, uid: None,
    get_prefill_tracker=lambda: SimpleNamespace(remove=lambda rid: None),
)
module = ast.fix_missing_locations(
    ast.Module(
        body=[
            proof_class,
            ast.ClassDef(
                name="TestedScheduler",
                bases=[],
                keywords=[],
                body=[*constants, *methods],
                decorator_list=[],
            ),
        ],
        type_ignores=[],
    )
)
exec(compile(module, str(SOURCE), "exec"), namespace)
Scheduler = namespace["TestedScheduler"]
namespace["Scheduler"] = Scheduler

core_tree = ast.parse((ROOT / "omlx/engine_core.py").read_text())
core_class = next(
    n for n in core_tree.body if isinstance(n, ast.ClassDef) and n.name == "EngineCore"
)
core_methods = [
    n
    for n in core_class.body
    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    and n.name in {"_cleanup_owned_waiting_on_executor", "_owned_cleanup_waiting"}
]
core_ns = dict(stream_fixture.namespace, WaitingCleaned=WaitingCleaned, RunningCleaned=RunningCleaned)
exec(
    compile(
        ast.Module(body=core_methods, type_ignores=[]),
        str(ROOT / "omlx/engine_core.py"),
        "exec",
    ),
    core_ns,
)


class WaitingCleanupTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = stream_fixture.StreamBarrierTests()
        await self.fixture.asyncSetUp()
        self.core = self.fixture.core
        s = self.scheduler = Scheduler()
        self.core.scheduler = s
        s.requests = {}
        s.waiting = deque()
        s.running = {}
        s.prefilling = deque()
        s._prefill_states = {}
        s.request_id_to_uid = {}
        s.uid_to_request_id = {}
        s._inflight_store_futures = {}
        s._boundary_cache_snapshots = {}
        s._request_detokenizers = {}
        s._output_parser_sessions = {}
        s._prefix_cache_prepared = set()
        s._pending_async_removes = deque()
        s._pending_abort_ids = set()
        s._pending_prefill_eviction_request = None
        s.paged_cache_manager = None
        s._owned_waiting_cleanup_proofs = {}
        s._cache_freshness_waits = {}
        s._throttle_notified_requests = set()
        s.finished_req_ids = set()
        s._stream = self.core._mlx_stream
        s._specprefill_draft_model = None
        s._memory_admission_blocked_request_id = None
        s._memory_admission_blocked_since = 0
        s._store_cache_admission_blocked_request_id = None
        s._store_cache_admission_blocked_since = 0
        s._boundary_snapshot_store = SimpleNamespace(cleanup_request=Mock())
        s.config = SimpleNamespace(max_num_seqs=1)
        s.tokenizer = SimpleNamespace(encode=lambda prompt: [1, 2])
        s._resolve_generation_prompt_start = Mock()
        s.moe_offload_stats = None
        s.block_aware_cache = None
        s.preflight_or_raise = Mock()
        s._release_paged_cache_for_request = Mock()
        s._admission_paused = False
        s._store_cache_gate = None
        s._prefill_memory_guard = False
        s._memory_limit_bytes = 0
        s._num_admitted_requests = Mock(return_value=0)
        s._effective_max_num_seqs = Mock(return_value=1)
        s._prefill_gate_open = Mock(return_value=False)
        s._should_defer_for_cache_freshness = Mock(return_value=False)
        s._ensure_batch_generator = Mock()
        s.batch_generator = None
        s._step_failed = False
        s._prefill_eviction_awaiting = set()
        s._specprefill_active_request_id = None
        s.model = SimpleNamespace()
        for name in ["_cleanup_owned_waiting_on_executor", "_owned_cleanup_waiting"]:
            setattr(self.core, name, MethodType(core_ns[name], self.core))

    async def asyncTearDown(self):
        await self.fixture.asyncTearDown()

    async def admitted(self, rid="synthetic-cleanup"):
        owned = OwnedAdmission(self.core, rid, time.monotonic() + 3)
        await self.core.add_request("synthetic", request_id=rid, _owned_admission=owned)
        return owned

    async def cleanup(self, owned):
        return await self.core._owned_cleanup_waiting(owned, time.monotonic() + 3)

    async def test_success_retires_exact_request_keeps_fact_and_frees_capacity(self):
        owned = await self.admitted()
        rid = owned.binding.request_id
        s = self.scheduler
        other = await self.admitted("synthetic-other")
        other_req = other.request
        s._cache_freshness_waits.update(
            {rid: object(), other.binding.request_id: object()}
        )
        s._throttle_notified_requests.update([rid, other.binding.request_id])
        s._memory_admission_blocked_request_id = rid
        s._store_cache_admission_blocked_request_id = other.binding.request_id
        fact = await self.cleanup(owned)
        self.assertIs(fact.handle, owned)
        self.assertIs(fact.request, owned.request)
        self.assertEqual(fact.generation_id, self.core._prism_control_generation_id)
        self.assertNotIn(rid, s.requests)
        self.assertFalse(any(r is owned.request for r in s.waiting))
        self.assertNotIn(rid, s._cache_freshness_waits)
        self.assertNotIn(rid, s._throttle_notified_requests)
        self.assertIsNone(s._memory_admission_blocked_request_id)
        self.assertEqual(
            s._store_cache_admission_blocked_request_id, other.binding.request_id
        )
        self.assertIs(s.requests[other.binding.request_id], other_req)
        self.assertIn(other.binding.request_id, s._cache_freshness_waits)
        self.assertIn(other.binding.request_id, s._throttle_notified_requests)
        self.assertFalse(s._owned_waiting_cleanup_proofs)
        self.assertIs(owned._cleanup_fact, fact)
        self.assertEqual(owned.request.status, RequestStatus.FINISHED_ABORTED)
        s._boundary_snapshot_store.cleanup_request.assert_called_once_with(
            rid, strict=True
        )
        self.core._cleanup_request.assert_not_called()
        self.assertFalse(hasattr(fact, "backend_stopped"))

    async def test_snapshot_error_retains_request_failed_record_and_fence(self):
        owned = await self.admitted()
        self.scheduler._boundary_snapshot_store.cleanup_request.side_effect = (
            TimeoutError("synthetic_writer_busy")
        )
        self.assertIsNone(await self.cleanup(owned))
        self.assertIs(self.scheduler.requests[owned.binding.request_id], owned.request)
        self.assertEqual(
            self.scheduler._owned_waiting_cleanup_proofs[
                owned.binding.request_id
            ].state,
            "failed",
        )
        self.assertTrue(self.scheduler._owned_waiting_fenced(owned.request))
        self.assertIsNone(await self.cleanup(owned))
        self.scheduler._boundary_snapshot_store.cleanup_request.assert_called_once()

    async def test_capacity_refusal_is_retained_without_mutating_queue(self):
        owned = await self.admitted()
        self.scheduler._MAX_OWNED_WAITING_PROOFS = 0
        self.assertIsNone(await self.cleanup(owned))
        self.assertEqual(owned._cleanup_outcome, "capacity_refused")
        self.assertTrue(self.scheduler._owned_waiting_fenced(owned.request))
        self.assertIs(self.scheduler.waiting[0], owned.request)
        self.assertFalse(self.scheduler._owned_waiting_cleanup_proofs)
        self.scheduler._boundary_snapshot_store.cleanup_request.assert_not_called()

    async def test_same_id_replacement_and_wrong_nonce_cannot_authorize_cleanup(self):
        for replace in [True, False]:
            with self.subTest(replace=replace):
                owned = await self.admitted("synthetic-" + str(replace))
                original = self.core._prism_control_generation_id
                if replace:
                    replacement = Request(
                        owned.binding.request_id, "other", owned.request.sampling_params
                    )
                    self.scheduler.requests[owned.binding.request_id] = replacement
                else:
                    self.core._prism_control_generation_id = (
                        "synthetic-wrong-generation"
                    )
                try:
                    self.assertIsNone(await self.cleanup(owned))
                finally:
                    self.core._prism_control_generation_id = original
                self.assertIn(owned.binding.request_id, self.scheduler.requests)
        self.scheduler._boundary_snapshot_store.cleanup_request.assert_not_called()

    async def test_active_cached_vlm_deferred_and_requeue_states_are_unsupported(self):
        changes = [
            ("status", RequestStatus.RUNNING),
            ("prefill_started_at", 1.0),
            ("prompt_cache", []),
            ("vlm_inputs_embeds", object()),
            ("prefill_resumed_tokens", 1),
        ]
        for index, (field, value) in enumerate(changes):
            with self.subTest(field=field):
                owned = await self.admitted("synthetic-active-" + str(index))
                setattr(owned.request, field, value)
                self.assertIsNone(await self.cleanup(owned))
                self.assertEqual(owned._cleanup_outcome, "unsupported")
                self.assertIs(
                    self.scheduler.requests[owned.binding.request_id], owned.request
                )
        self.scheduler._boundary_snapshot_store.cleanup_request.assert_not_called()

    async def test_fenced_head_rotation_preserves_pristine_and_ordinary_order(self):
        owned = await self.admitted()
        other = await self.admitted("synthetic-next")
        owned.cancel()
        self.scheduler._schedule_waiting()
        self.assertIs(self.scheduler.waiting[0], other.request)
        self.assertTrue(owned._waiting_pristine)
        self.assertIsNotNone(await self.cleanup(owned))
        other.cancel()
        before = list(self.scheduler.waiting)
        self.assertFalse(self.scheduler._select_unfenced_waiting())
        self.assertEqual(list(self.scheduler.waiting), before)

    async def test_actual_scheduling_pop_irreversibly_invalidates_history(self):
        owned = await self.admitted()
        self.scheduler._prefill_gate_open.return_value = True
        self.scheduler._schedule_waiting()
        self.assertFalse(owned._waiting_pristine)
        self.assertIs(self.scheduler.waiting[0], owned.request)
        self.assertIsNone(await self.cleanup(owned))
        self.assertEqual(owned._cleanup_outcome, "unsupported")

    async def test_direct_fenced_prefix_entry_returns_sentinel_without_resources(self):
        owned = await self.admitted()
        owned.cancel()
        result = self.scheduler._prepare_prefix_cache_for_request(owned.request)
        self.assertIs(result, self.scheduler._OWNED_PREFIX_FENCED)
        self.assertFalse(owned._waiting_pristine)
        self.scheduler._ensure_batch_generator.assert_not_called()
        self.assertIsNone(await self.cleanup(owned))

    async def test_queue_removal_error_never_publishes_success(self):
        owned = await self.admitted()

        class RefuseRemoval(deque):
            def remove(self, value):
                raise ValueError("synthetic_remove_failed")

        self.scheduler.waiting = RefuseRemoval(self.scheduler.waiting)
        self.assertIsNone(await self.cleanup(owned))
        self.assertEqual(owned._cleanup_outcome, "failed")
        self.assertIsNone(owned._cleanup_fact)
        self.assertIs(self.scheduler.requests[owned.binding.request_id], owned.request)
        self.assertTrue(self.scheduler._owned_waiting_fenced(owned.request))

    async def test_consume_mismatch_is_failed_before_retaining_fact(self):
        owned = await self.admitted()
        owned.cancel()
        owned.pin_stream_deadline(self.core, time.monotonic() + 3)
        fact = await asyncio.wrap_future(
            self.fixture.fixture.executor.submit(
                self.scheduler._cleanup_owned_waiting, owned
            )
        )
        record = self.scheduler._owned_waiting_cleanup_proofs[owned.binding.request_id]
        record.state = "pending"
        result = self.scheduler._retain_consume_owned_waiting(owned, fact)
        self.assertIsNone(result)
        self.assertIsNone(owned._cleanup_fact)
        self.assertEqual(record.state, "failed")
        self.assertEqual(len(self.scheduler._owned_waiting_cleanup_proofs), 1)

    async def test_cancelled_cleanup_retains_real_future_and_blocks_premature_stream_barrier(
        self,
    ):
        owned = await self.admitted()
        entered, release = self.fixture.fixture.event(), self.fixture.fixture.event()

        def held(rid, **kwargs):
            entered.set()
            release.wait(2)

        self.scheduler._boundary_snapshot_store.cleanup_request.side_effect = held
        task = asyncio.create_task(self.cleanup(owned))
        self.fixture.fixture.tasks.append(task)
        await self.fixture.fixture.until(entered.is_set)
        original = owned._cleanup_future
        task.cancel()
        self.assertIsNone(await task)
        self.assertFalse(original.cancelled())
        self.assertFalse(original.done())
        self.assertIsNone(
            await self.core._owned_stream_barrier(owned, time.monotonic() + 3)
        )
        self.fixture.sync.assert_not_called()
        release.set()
        fact = await self.cleanup(owned)
        self.assertIsNotNone(fact)
        self.assertIs(owned._cleanup_future, original)
        self.scheduler._boundary_snapshot_store.cleanup_request.assert_called_once()
        self.assertIsNotNone(
            await self.core._owned_stream_barrier(owned, time.monotonic() + 3)
        )

    async def test_prior_diagnostic_barrier_refuses_cleanup_without_retiring_request(
        self,
    ):
        owned = await self.admitted()
        self.assertIsNotNone(
            await self.core._owned_stream_barrier(owned, time.monotonic() + 3)
        )
        self.assertIsNone(await self.cleanup(owned))
        self.assertEqual(owned._cleanup_outcome, "prior_stream_barrier")
        self.assertIs(self.scheduler.requests[owned.binding.request_id], owned.request)

    async def test_ordinary_requests_keep_original_scheduler_path(self):
        await self.core.add_request("ordinary", request_id="synthetic-ordinary")
        request = self.scheduler.requests["synthetic-ordinary"]
        self.assertFalse(self.scheduler._owned_waiting_fenced(request))
        self.scheduler._prefill_gate_open.return_value = True
        self.scheduler._schedule_waiting()
        self.assertIs(self.scheduler.waiting[0], request)
        self.assertFalse(hasattr(request, "_prism_owned_admission"))


if __name__ == "__main__":
    unittest.main()
