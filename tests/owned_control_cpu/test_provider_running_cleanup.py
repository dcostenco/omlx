# SPDX-License-Identifier: Apache-2.0
"""Slice 25b.4 CPU proof: one owned-cleanup entry for decoding and chunked prefill.

Actual Scheduler/EngineCore methods run on a real single-worker executor. The
paged-cache manager is the real class (loaded without the cache package, so no
MLX import); the BatchGenerator is a fake with mlx-lm's three-stage uid
semantics. KV slicing, Metal stream hazards and sampler alignment are not
observable here.
"""

import __future__
import ast
import asyncio
from collections import deque
from contextlib import contextmanager
import importlib.util
import logging
import sys
import time
import types
from types import MethodType, SimpleNamespace
from typing import Any, Optional
import unittest
from unittest.mock import Mock, patch

import test_provider_waiting_cleanup as waiting_fixture
from omlx.owned_admission import OwnedAdmission, RunningCleaned, StreamSynchronized, WaitingCleaned, _admitted_text_only
from omlx.request import Request, RequestStatus

ROOT = waiting_fixture.ROOT
SOURCE = ROOT / "omlx/scheduler.py"
tree = ast.parse(SOURCE.read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")

# The real PagedCacheManager, imported under a private package name so the
# omlx.cache package initializer (which imports MLX) never runs.
_CACHE_DIR = ROOT / "omlx/cache"
_pkg = types.ModuleType("omlx_cache_private")
_pkg.__path__ = [str(_CACHE_DIR)]
sys.modules.setdefault("omlx_cache_private", _pkg)
_spec = importlib.util.spec_from_file_location("omlx_cache_private.paged_cache", _CACHE_DIR / "paged_cache.py")
paged_cache = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = paged_cache
_spec.loader.exec_module(paged_cache)
assert "mlx" not in sys.modules

NAMES = set(waiting_fixture.NAMES) | {
    "_remove_uid_from_active_batch",
    "step",
    "prefill_eviction_finished",
    # Review cycle 1: the real requeue, generator drop, legacy abort,
    # failure reset and eviction hand-off run here, not simulations.
    "_mark_owned_started",
    "_reschedule_running_requests",
    "_reset_request_for_reprefill",
    "_do_abort_request",
    "fail_all_requests",
    "_publish_prefill_eviction",
    "_requeue_prefill_retry",
    # Slice 28: the retry sites consult the owned no-retry mark.
    "_owned_no_retry",
    "_queue_owned_no_retry_output",
    "_end_owned_without_retry",
}
methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in NAMES]
assert len(methods) == len(NAMES), sorted(NAMES - {m.name for m in methods})
# Real methods that only newer upstream trees have (0.7.1.dev2+:
# _schedule_waiting builds its stop matcher through this wrapper).
OPTIONAL_NAMES = {"_stop_sequences_for_insert"}
methods += [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in OPTIONAL_NAMES]
constants = [
    n for n in cls.body
    if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id in {"_MAX_OWNED_WAITING_PROOFS", "_OWNED_PREFIX_FENCED"}
        for t in n.targets
    )
]
helper_classes = [
    n for n in tree.body
    if isinstance(n, ast.ClassDef) and n.name in {"_WaitingCleanupProof", "_PrefillState"}
]
assert len(helper_classes) == 2
# Module-level functions the extracted methods call: the real ones, followed
# transitively (upstream 0.7.1.dev2+ added _requeue_row_state and
# _request_count to _schedule_waiting). Names the namespace below stubs win.
_module_functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def _called_module_functions(roots, stubbed):
    wanted, queue = {}, list(roots)
    while queue:
        for name in {n.id for n in ast.walk(queue.pop()) if isinstance(n, ast.Name)}:
            if name in _module_functions and name not in stubbed and name not in wanted:
                wanted[name] = _module_functions[name]
                queue.append(wanted[name])
    return list(wanted.values())


class FakeMx:
    """Device stub: synchronize is recorded; stream() is a no-op context."""

    def __init__(self):
        self.synchronize = Mock()

    @contextmanager
    def stream(self, _stream):
        yield


class FakeBatchGenerator:
    """mlx-lm BatchGenerator uid stages: 0 unprocessed, 1 prompt, 2 generation."""

    def __init__(self):
        self.stages = {0: [], 1: [], 2: []}
        self.remove_error = None
        self.partial_error = None
        self.ignore_remove = False

    def _find_uids(self, uids):
        wanted, found = set(uids), {}
        for stage in (2, 1, 0):
            for index, uid in enumerate(self.stages[stage]):
                if uid in wanted:
                    found[uid] = (stage, index)
        return found

    def remove(self, uids, return_prompt_caches=False):
        if self.partial_error is not None:
            # Filter the generation stage, then fail before the other stages.
            self.stages[2] = [u for u in self.stages[2] if u not in set(uids)]
            raise self.partial_error
        if self.remove_error is not None:
            raise self.remove_error
        if self.ignore_remove:
            return {}
        for stage in self.stages:
            self.stages[stage] = [u for u in self.stages[stage] if u not in set(uids)]
        return {}


fake_mx = FakeMx()
tracker = SimpleNamespace(removed=[], remove=lambda rid: tracker.removed.append(rid))
namespace = dict(
    waiting_fixture.namespace,
    Any=Any,
    Optional=Optional,
    mx=fake_mx,
    RunningCleaned=RunningCleaned,
    deque=deque,
    get_prefill_tracker=lambda: tracker,
    _unregister_uid_row=lambda model, uid: None,
    _unregister_uid_rows_for_model=lambda model: None,
    _safe_sync_stream=lambda stream: None,
    Scheduler=SimpleNamespace(_clear_cache=lambda scheduler: None),
)
helper_functions = _called_module_functions([*helper_classes, *methods], namespace)
exec(
    compile(
        ast.fix_missing_locations(ast.Module(body=[
            *helper_classes,
            *helper_functions,
            ast.ClassDef(name="RunningScheduler", bases=[], keywords=[], body=[*constants, *methods], decorator_list=[]),
        ], type_ignores=[])),
        str(SOURCE), "exec",
        # Python 3.13 evaluates annotations at def time (3.14 defers them), and
        # the stub namespace lacks names the real module imports, such as
        # PrefillMemoryExceededError; defer them here on every version.
        flags=__future__.annotations.compiler_flag, dont_inherit=True,
    ),
    namespace,
)
RunningScheduler = namespace["RunningScheduler"]
PrefillState = namespace["_PrefillState"]


class FakeBlockAwareCache:
    def __init__(self):
        self.cleared = []

    def clear_request_entry(self, request_id):
        self.cleared.append(request_id)


class RunningCleanupTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.base = waiting_fixture.WaitingCleanupTests()
        await self.base.asyncSetUp()
        self.core = self.base.core
        s = RunningScheduler()
        s.__dict__.update(self.base.scheduler.__dict__)
        self.core.scheduler = self.s = s
        # A fresh device stub per test; the extracted methods read mx.synchronize at call time.
        fake_mx.synchronize = Mock()
        tracker.removed.clear()
        s.batch_generator = FakeBatchGenerator()
        s.model = SimpleNamespace()
        s.block_aware_cache = FakeBlockAwareCache()
        s._detect_boundary_snapshot_need = Mock(return_value=False)
        s.paged_cache_manager = self.pcm = paged_cache.PagedCacheManager(block_size=4, max_blocks=32, model_name="synthetic")
        s._cleanup_detokenizer = Mock()
        s._cleanup_output_parser_session = Mock()
        s._clear_request_admission_bookkeeping = Mock()
        s._schedule_deferred_metal_clear = Mock()
        s._boundary_snapshot_store = SimpleNamespace(cleanup_request=Mock(), cleanup_all=Mock())
        s._specprefill_active_request_id = None
        s._step_failed = False
        s._prefill_eviction_awaiting = set()
        s._pending_prefill_eviction_request = None
        s._cleanup_specprefill = Mock()
        s._drop_boundary_snapshots_for_request = Mock()
        s._drain_pending_async_removes = Mock(return_value=0)
        s._generation_overflow_recovery_ids = set()
        # Scheduler.__init__'s default since upstream 0.7.1.dev2: aborts whose
        # cache store is kept; _do_abort_request reads it.
        s._abort_keep_cache_ids = set()
        s._publish_admin_snapshot = Mock()
        # A cached two-block prefix nobody holds: ref_count 0, hashes registered.
        tokens, parent, seed = list(range(8)), None, self.pcm.create_block_table("seed")
        for i in range(2):
            block = self.pcm.allocate_block()
            self.pcm.add_block_to_table(seed, block, 4)
            self.pcm.register_block_hash(block, tokens[i * 4:(i + 1) * 4], parent)
            parent = block.block_hash
        self.prefix = [(b, self.pcm.allocated_blocks[b].block_hash) for b in seed.block_ids]
        self.pcm.release_for_eviction(list(seed.block_ids))
        self.pcm.request_tables.pop("seed", None)

    async def asyncTearDown(self):
        await self.base.asyncTearDown()

    def refs(self):
        return [self.pcm.allocated_blocks[b].ref_count for b, _ in self.prefix]

    def fetch(self, request):
        """What BlockAwarePrefixCache.fetch_cache does to the manager on a hit."""
        table = self.pcm.create_block_table(request.request_id)
        for block_id, block_hash in self.prefix:
            block = self.pcm.acquire_cached_block(block_id, block_hash)
            table.block_ids.append(block.block_id)
            table.num_tokens += block.token_count
        request.block_table = table

    async def admitted(self, rid, **fields):
        owned = OwnedAdmission(self.core, rid, time.monotonic() + 3)
        await self.core.add_request("synthetic", request_id=rid, _owned_admission=owned, **fields)
        return owned

    def to_decoding(self, owned, uid, stage=2):
        s, request, rid = self.s, owned.request, owned.binding.request_id
        s.waiting.remove(request)
        owned._waiting_pristine = False
        request.status = RequestStatus.RUNNING
        request.batch_uid = uid
        s.running[rid] = request
        s.request_id_to_uid[rid] = uid
        s.uid_to_request_id[uid] = rid
        s.batch_generator.stages[stage].append(uid)
        # What _schedule_waiting does when it commits the request to compute
        # (the real call is exercised in the scheduling test below).
        s._mark_owned_started(request)
        self.fetch(request)

    def to_chunked(self, owned):
        s, request, rid = self.s, owned.request, owned.binding.request_id
        s.waiting.remove(request)
        owned._waiting_pristine = False
        s._mark_owned_started(request)
        state = PrefillState(request=request, cache=[object()], tokens_remaining=None, last_token=[0],
            tokens_processed=4, base_size=0, emitted_boundaries={}, boundary_enabled=False,
            block_size=4, total_length=12)
        s._prefill_states[rid] = state
        s.prefilling.append(request)
        self.fetch(request)
        return state

    async def cleanup(self, owned):
        return await self.core._owned_cleanup_waiting(owned, time.monotonic() + 3)

    def assert_retired(self, owned, uid=None):
        s, rid = self.s, owned.binding.request_id
        self.assertNotIn(rid, s.requests)
        self.assertNotIn(rid, s.running)
        self.assertNotIn(rid, s.request_id_to_uid)
        if uid is not None:
            self.assertNotIn(uid, s.uid_to_request_id)
        self.assertNotIn(rid, s._prefill_states)
        self.assertFalse(any(r is owned.request for r in s.prefilling))
        self.assertIsNone(self.pcm.get_block_table(rid))
        self.assertIsNone(owned.request.block_table)
        self.assertEqual(owned.request.status, RequestStatus.FINISHED_ABORTED)

    async def test_decoding_success_preserves_unrelated_row_and_shared_block_refs(self):
        owned, other = await self.admitted("synthetic-r"), await self.admitted("synthetic-u")
        self.to_decoding(other, uid=7)
        before = self.refs()
        self.to_decoding(owned, uid=3)
        self.assertEqual(self.refs(), [r + 1 for r in before])
        self.s._boundary_cache_snapshots["synthetic-r"] = object()
        self.s._boundary_cache_snapshots["synthetic-u"] = unrelated_snapshot = object()
        fact = await self.cleanup(owned)
        self.assertIs(type(fact), RunningCleaned)
        # Every retirement step ran for this request only.
        self.assertNotIn("synthetic-r", self.s._boundary_cache_snapshots)
        self.assertIs(self.s._boundary_cache_snapshots["synthetic-u"], unrelated_snapshot)
        self.s._cleanup_detokenizer.assert_called_once_with("synthetic-r")
        self.s._cleanup_output_parser_session.assert_called_once_with("synthetic-r")
        self.s._clear_request_admission_bookkeeping.assert_any_call("synthetic-r")
        self.s._schedule_deferred_metal_clear.assert_called_once_with()
        self.assertIn("synthetic-r", tracker.removed)
        self.assertIn("synthetic-r", self.s.finished_req_ids)
        self.assertIsNotNone(self.s.batch_generator)
        self.s._boundary_snapshot_store.cleanup_request.assert_called_once_with("synthetic-r", strict=True)
        self.assertEqual(owned._cleanup_failures, ())
        self.assertEqual(self.s.request_id_to_uid["synthetic-u"], 7)
        self.assertEqual((fact.state, fact.request_id), ("decoding", "synthetic-r"))
        self.assertIs(fact.handle, owned)
        self.assert_retired(owned, uid=3)
        self.assertEqual(self.refs(), before)
        self.assertEqual(self.s.batch_generator._find_uids([7]), {7: (2, 0)})
        self.assertIs(self.s.running["synthetic-u"], other.request)
        self.assertIsNotNone(self.pcm.get_block_table("synthetic-u"))
        self.assertEqual(self.s.block_aware_cache.cleared, ["synthetic-r"])
        self.assertEqual(owned._classified_state, "decoding")
        self.assertFalse(self.s._owned_waiting_cleanup_proofs)
        self.assertIs(owned._cleanup_fact, fact)
        fake_mx.synchronize.assert_called_once_with(self.s._stream)
        barrier = await self.core._owned_stream_barrier(owned, time.monotonic() + 3)
        self.assertIs(type(barrier), StreamSynchronized)

    async def test_chunked_prefill_success_preserves_order_and_releases_own_refs(self):
        first, owned, last = [await self.admitted(n) for n in ("synthetic-a", "synthetic-r", "synthetic-z")]
        before = self.refs()
        self.to_chunked(first)
        mid_refs = self.refs()
        state = self.to_chunked(owned)
        self.to_chunked(last)
        fact = await self.cleanup(owned)
        self.assertEqual((type(fact), fact.state), (RunningCleaned, "chunked_prefill"))
        self.assert_retired(owned)
        self.assertEqual([r.request_id for r in self.s.prefilling], ["synthetic-a", "synthetic-z"])
        self.assertIsNone(state.cache)
        self.assertEqual(self.refs(), [r + 1 for r in mid_refs])  # a and z still hold theirs
        self.assertNotEqual(self.refs(), before)
        self.assertIn("synthetic-r", tracker.removed)

    async def test_uid_at_any_stage_is_supported_and_absent_uid_is_refused(self):
        for stage in (0, 1):
            await self.asyncTearDown(); await self.asyncSetUp()
            owned = await self.admitted("synthetic-stage")
            self.to_decoding(owned, uid=5, stage=stage)
            self.assertEqual((await self.cleanup(owned)).state, "decoding")
        await self.asyncTearDown(); await self.asyncSetUp()
        owned = await self.admitted("synthetic-gone")
        self.to_decoding(owned, uid=5)
        self.s.batch_generator.stages[2].clear()
        self.assertIsNone(await self.cleanup(owned))
        self.assertEqual((owned._classified_state, owned._cleanup_outcome), ("unsupported_state", "unsupported"))
        self.assertIn("synthetic-gone", self.s.running)
        self.assertFalse(self.s._owned_waiting_cleanup_proofs)

    async def test_refusals_change_nothing_use_no_slot_and_never_fence_a_requeue(self):
        def none(s, rid):
            return None

        model_features = {
            "mtp_priming": {"_mtp_priming": SimpleNamespace(_eligible_host=lambda model: model)},
            "mtp_decode": {"_model_mtp_decode_enabled": lambda model: True},
            "block_drafter": {"_block_drafter_for": lambda model: object()},
        }
        cases = {
            "pending_abort": lambda s, rid: s._pending_abort_ids.add(rid),
            "scheduler_dirty": lambda s, rid: setattr(s, "_step_failed", True),
            "eviction_pending": lambda s, rid: s._prefill_eviction_awaiting.add(rid),
            "specprefill": lambda s, rid: setattr(s, "_specprefill_active_request_id", rid),
            "vlm_mtp": lambda s, rid: setattr(s, "_vlm_mtp_drafter", object()),
            "not_text_only": lambda s, rid: setattr(s.requests[rid], "rope_deltas", 1.5),
            "mtp_priming": none,
            "mtp_decode": none,
            "block_drafter": none,
            "specprefill_draft_model": lambda s, rid: setattr(s, "_specprefill_draft_model", object()),
            # The scheduler no longer runs on the stream the admission bound.
            "admission_incomplete": lambda s, rid: setattr(s, "_stream", object()),
        }
        for reason, apply in cases.items():
            with self.subTest(reason=reason), patch.dict(namespace, model_features.get(reason, {})):
                await self.asyncTearDown(); await self.asyncSetUp()
                owned = await self.admitted("synthetic-refused")
                self.to_decoding(owned, uid=4)
                refs = self.refs()
                apply(self.s, "synthetic-refused")
                if reason == "specprefill_draft_model":
                    reason = "specprefill"
                self.assertIsNone(await self.cleanup(owned))
                self.assertEqual((owned._classified_state, owned._cleanup_outcome), (reason, "unsupported"))
                self.assertIs(self.s.running["synthetic-refused"], owned.request)
                self.assertEqual(self.s.batch_generator._find_uids([4]), {4: (2, 0)})
                self.assertEqual(self.refs(), refs)
                self.assertFalse(self.s._owned_waiting_cleanup_proofs)
                fake_mx.synchronize.assert_not_called()
                # An ordinary requeue (as _requeue_prefill_retry does) is not fenced.
                self.s.running.pop("synthetic-refused")
                owned.request.status = RequestStatus.WAITING
                self.s.waiting.append(owned.request)
                self.assertFalse(self.s._owned_waiting_fenced(owned.request))
                self.assertTrue(self.s._select_unfenced_waiting())

    async def test_image_request_is_refused_after_its_vlm_fields_are_cleared(self):
        owned = await self.admitted("synthetic-image", images=["synthetic-image-bytes"])
        self.assertFalse(owned._text_only)
        self.to_decoding(owned, uid=6)
        owned.request.images = None
        self.assertIsNone(await self.cleanup(owned))
        self.assertEqual(owned._classified_state, "not_text_only")

    async def test_natural_finish_and_absent_requests_are_slot_free_non_success(self):
        owned = await self.admitted("synthetic-finished")
        self.to_decoding(owned, uid=8)
        self.s._inflight_store_futures["synthetic-finished"] = object()
        self.assertIsNone(await self.cleanup(owned))
        self.assertEqual(owned._cleanup_outcome, "finished_before_cleanup")
        self.assertFalse(self.s._owned_waiting_cleanup_proofs)
        gone = await self.admitted("synthetic-absent")
        self.to_decoding(gone, uid=9)
        for table in (self.s.requests, self.s.running):
            table.pop("synthetic-absent")
        self.assertIsNone(await self.cleanup(gone))
        self.assertEqual(gone._cleanup_outcome, "absent")
        self.assertFalse(self.s._owned_waiting_cleanup_proofs)

    async def test_same_id_replacement_is_refused(self):
        owned = await self.admitted("synthetic-replaced")
        self.to_decoding(owned, uid=2)
        self.s.requests["synthetic-replaced"] = Request(request_id="synthetic-replaced", prompt="x", sampling_params=owned.request.sampling_params)
        self.assertIsNone(await self.cleanup(owned))
        self.assertEqual(owned._classified_state, "identity_mismatch")

    async def test_snapshot_or_sync_failure_leaves_the_request_running_untouched(self):
        for failure in ("snapshot", "sync"):
            with self.subTest(failure=failure):
                await self.asyncTearDown(); await self.asyncSetUp()
                owned = await self.admitted("synthetic-early-fail")
                self.to_decoding(owned, uid=4)
                refs = self.refs()
                if failure == "snapshot":
                    self.s._boundary_snapshot_store.cleanup_request.side_effect = TimeoutError("writer")
                else:
                    fake_mx.synchronize.side_effect = RuntimeError("device")
                self.assertIsNone(await self.cleanup(owned))
                self.assertEqual(owned._cleanup_outcome, "failed")
                self.assertIs(self.s.running["synthetic-early-fail"], owned.request)
                self.assertEqual(self.s.batch_generator._find_uids([4]), {4: (2, 0)})
                self.assertEqual(self.refs(), refs)
                self.assertIsNotNone(self.pcm.get_block_table("synthetic-early-fail"))
                self.assertEqual(self.s._owned_waiting_cleanup_proofs["synthetic-early-fail"].state, "failed")
                # A legacy abort afterwards still releases exactly this request.
                fake_mx.synchronize.side_effect = None
                self.s._boundary_snapshot_store.cleanup_request.side_effect = None
                self.assertTrue(self.s._do_abort_request("synthetic-early-fail"))
                self.assertNotIn("synthetic-early-fail", self.s.running)
                self.assertNotIn("synthetic-early-fail", self.s.requests)
                self.assertEqual(self.s.batch_generator._find_uids([4]), {})
                self.assertEqual(self.refs(), [r - 1 for r in refs])

    async def test_failure_after_first_mutation_still_completes_retirement(self):
        def failing_row(model, uid):
            raise RuntimeError("row")

        for failure in ("remove_raises", "partial_filter", "row_survives", "release_raises", "drop_steps_raise"):
            with self.subTest(failure=failure), patch.dict(
                namespace, {"_unregister_uid_row": failing_row} if failure == "drop_steps_raise" else {}
            ):
                await self.asyncTearDown(); await self.asyncSetUp()
                owned, other = await self.admitted("synthetic-late-fail"), await self.admitted("synthetic-u")
                self.to_decoding(other, uid=7)
                unrelated = self.refs()
                self.to_decoding(owned, uid=4)
                generator = self.s.batch_generator
                if failure == "drop_steps_raise":
                    # Every fallible step inside the drop fails; the drop and
                    # the requeue of the other request still complete.
                    self.s._boundary_snapshot_store.cleanup_all.side_effect = RuntimeError("snapshots")
                if failure in ("remove_raises", "drop_steps_raise"):
                    generator.remove_error = RuntimeError("filter")
                elif failure == "partial_filter":
                    generator.partial_error = RuntimeError("filter midway")
                elif failure == "row_survives":
                    generator.ignore_remove = True
                else:
                    self.pcm.release_for_eviction = Mock(side_effect=RuntimeError("release"))
                self.assertIsNone(await self.cleanup(owned))
                self.assertEqual(owned._cleanup_outcome, "failed")
                self.assertTrue(owned._cleanup_failures)
                self.assert_retired(owned, uid=4)  # no phantom running entry or table
                if failure == "release_raises":
                    # The generator is intact; the unrelated row keeps decoding.
                    self.assertIs(self.s.batch_generator, generator)
                    self.assertIs(self.s.running["synthetic-u"], other.request)
                    self.s._boundary_snapshot_store.cleanup_all.assert_not_called()
                    self.assertEqual(self.refs(), [r + 1 for r in unrelated])  # recorded leak, never a double release
                else:
                    # A generator whose remove failed or left the row is dropped,
                    # as the prefill-abort recovery does, and the other request
                    # is requeued to re-prefill instead of decoding on it.
                    self.assertIsNone(self.s.batch_generator)
                    self.assertNotIn("synthetic-u", self.s.running)
                    self.assertIs(self.s.waiting[0], other.request)
                    # Its row is forgotten, so a fresh generator reusing uid 7
                    # cannot be confused with it by a later abort.
                    self.assertNotIn("synthetic-u", self.s.request_id_to_uid)
                    self.assertNotIn(7, self.s.uid_to_request_id)
                    self.assertEqual(other.request.status, RequestStatus.WAITING)
                    self.s._boundary_snapshot_store.cleanup_all.assert_called_once_with()
                    self.assertNotIn("generator_drop", owned._cleanup_failures)
                    self.assertEqual(self.refs(), unrelated)
                    if failure == "drop_steps_raise":
                        self.assertIn("snapshot_cleanup_all", owned._cleanup_failures)
                        self.assertIn("uid_row", owned._cleanup_failures)
                # A legacy abort of the retired id afterwards finds nothing and
                # releases nothing twice.
                refs = self.refs()
                self.assertFalse(self.s._do_abort_request("synthetic-late-fail"))
                self.assertEqual(self.refs(), refs)

    async def test_deadline_after_the_fact_consumes_the_record(self):
        owned = await self.admitted("synthetic-late")
        self.to_decoding(owned, uid=3)
        original = self.s._cleanup_owned_request

        def slow(handle):
            fact = original(handle)
            handle._stream_effective_deadline = time.monotonic() - 1
            return fact

        self.s._cleanup_owned_request = slow
        self.assertIsNone(await self.cleanup(owned))
        self.assertEqual(owned._cleanup_outcome, "deadline")
        self.assertFalse(self.s._owned_waiting_cleanup_proofs)
        # The late fact is not left on the handle for a later stage to read.
        self.assertIsNone(owned._cleanup_fact)

    async def test_a_late_consume_mismatch_keeps_its_own_outcome(self):
        owned = await self.admitted("synthetic-late-mismatch")
        self.to_decoding(owned, uid=3)
        original = self.s._cleanup_owned_request

        def mismatched_and_slow(handle):
            fact = original(handle)
            self.s._owned_waiting_cleanup_proofs.pop("synthetic-late-mismatch")
            handle._stream_effective_deadline = time.monotonic() - 1
            return fact

        self.s._cleanup_owned_request = mismatched_and_slow
        self.assertIsNone(await self.cleanup(owned))
        self.assertEqual(owned._cleanup_outcome, "consume_mismatch")
        self.assertIsNone(owned._cleanup_fact)

    async def test_repeated_calls_join_one_future(self):
        owned = await self.admitted("synthetic-join")
        self.to_decoding(owned, uid=3)
        first, second = await asyncio.gather(self.cleanup(owned), self.cleanup(owned))
        self.assertIs(first, second)
        self.assertIs(type(first), RunningCleaned)
        fake_mx.synchronize.assert_called_once()

    def test_step_failure_marks_the_scheduler_dirty_and_eviction_clears(self):
        s = self.s
        s._step_unguarded = Mock(side_effect=RuntimeError("step"))
        with self.assertRaises(RuntimeError):
            s.step()
        self.assertTrue(s._step_failed)
        s._step_unguarded = Mock(return_value="output")
        s._step_failed = False
        self.assertEqual(s.step(), "output")
        self.assertFalse(s._step_failed)

        class Interrupted(BaseException):
            pass

        s._step_unguarded = Mock(side_effect=Interrupted())
        with self.assertRaises(Interrupted):
            s.step()
        self.assertTrue(s._step_failed)
        s._prefill_eviction_awaiting.add("synthetic-evict")
        s.prefill_eviction_finished("synthetic-evict")
        self.assertFalse(s._prefill_eviction_awaiting)

    def test_published_eviction_is_awaited_until_the_loop_finishes_it(self):
        s = self.s
        output = SimpleNamespace(prefill_eviction_request=None, has_work=False)
        s._publish_prefill_eviction(output)
        self.assertEqual((output.prefill_eviction_request, output.has_work), (None, False))
        paused = SimpleNamespace(request_id="synthetic-evict")
        s._pending_prefill_eviction_request = paused
        s._publish_prefill_eviction(output)
        self.assertIs(output.prefill_eviction_request, paused)
        self.assertTrue(output.has_work)
        self.assertIsNone(s._pending_prefill_eviction_request)
        self.assertEqual(s._prefill_eviction_awaiting, {"synthetic-evict"})
        # The step body hands off through this helper exactly once.
        step_body = next(m for m in cls.body if isinstance(m, ast.FunctionDef) and m.name == "_step_unguarded")
        calls = [
            n for n in ast.walk(step_body)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "_publish_prefill_eviction"
        ]
        self.assertEqual(len(calls), 1)

    async def test_fail_all_requests_clears_the_dirty_flag_and_eviction_markers(self):
        owned = await self.admitted("synthetic-failed-all")
        self.to_decoding(owned, uid=3)
        s = self.s
        s._step_failed = True
        s._prefill_eviction_awaiting.update({"synthetic-failed-all", "synthetic-other"})
        self.assertIn("synthetic-failed-all", s.fail_all_requests())
        self.assertFalse(s._step_failed)
        self.assertFalse(s._prefill_eviction_awaiting)
        self.assertIsNone(s.batch_generator)

    async def test_unclassified_cancelled_running_request_is_never_parked_after_a_real_requeue(self):
        async def barrier_first(owned):
            self.assertIsNotNone(await self.core._owned_stream_barrier(owned, time.monotonic() + 3))
            self.assertIsNone(await self.cleanup(owned))
            self.assertEqual(owned._cleanup_outcome, "prior_stream_barrier")

        async def expired_pin(owned):
            owned.pin_stream_deadline(self.core, time.monotonic() - 1)
            self.assertIsNone(await self.cleanup(owned))
            self.assertFalse(owned._cleanup_started)

        async def classification_error(owned):
            self.s.batch_generator._find_uids = Mock(side_effect=RuntimeError("uids"))
            self.assertIsNone(await self.cleanup(owned))
            self.assertEqual(
                (owned._classified_state, owned._cleanup_outcome), ("classification_failed", "failed")
            )
            self.assertFalse(self.s._owned_waiting_cleanup_proofs)

        def reschedule_running(s, owned):
            # The generator-reset recovery moves running requests to waiting.
            s._reschedule_running_requests()

        def prefill_retry(s, owned):
            # The OOM catch site tears the request down, then requeues it.
            rid = owned.binding.request_id
            uid = s.request_id_to_uid.pop(rid)
            s.uid_to_request_id.pop(uid)
            s.running.pop(rid)
            s.requests.pop(rid)
            s.batch_generator.stages[2].remove(uid)
            s._requeue_prefill_retry(owned.request)

        cases = [
            (name, refuse, path)
            for name, refuse in [
                ("barrier_first", barrier_first),
                ("expired_pin", expired_pin),
                ("classification_error", classification_error),
            ]
            for path in (reschedule_running, prefill_retry)
        ]
        for name, refuse, requeue in cases:
            with self.subTest(case=name, requeue=requeue.__name__):
                await self.asyncTearDown(); await self.asyncSetUp()
                owned = await self.admitted("synthetic-requeued")
                self.to_decoding(owned, uid=4)
                await refuse(owned)
                self.assertTrue(owned._cancelled)
                self.assertIs(self.s.running["synthetic-requeued"], owned.request)
                requeue(self.s, owned)
                self.assertIs(self.s.waiting[0], owned.request)
                self.assertEqual(owned.request.status, RequestStatus.WAITING)
                self.assertFalse(self.s._owned_waiting_fenced(owned.request))
                self.assertTrue(self.s._select_unfenced_waiting())
                if owned._classified_state is None:
                    # Without the started marker this handle would be parked.
                    owned._started = False
                    self.assertTrue(self.s._owned_waiting_fenced(owned.request))

    async def test_actual_scheduling_marks_started_before_the_first_compute_call(self):
        class Halt(BaseException):
            pass

        # The external (whole-prompt) prefill entry and the chunked entry.
        for entry in ("_do_external_prefill", "_begin_prefill"):
            with self.subTest(entry=entry):
                await self.asyncTearDown(); await self.asyncSetUp()
                s = self.s
                owned = await self.admitted("synthetic-scheduled")
                self.assertFalse(owned._started)
                seen = []

                def first_compute(request, *args, **kwargs):
                    seen.append((request is owned.request, owned._started))
                    raise Halt()

                s._prefill_gate_open.return_value = True
                s._prepare_prefix_cache_for_request = Mock(return_value=None)
                s._build_sampler_and_processors = Mock(return_value=(None, []))
                s._build_state_machine = Mock()
                s._preflight_memory_check = Mock(return_value=None)
                s._do_external_prefill = Mock(side_effect=AssertionError("unexpected entry"))
                s._begin_prefill = Mock(side_effect=AssertionError("unexpected entry"))
                setattr(s, entry, Mock(side_effect=first_compute))
                s._decode_fairness = False
                s._is_harmony_model = False
                s._detect_needs_think_prefix = Mock(return_value=False)
                chunked = entry == "_begin_prefill"
                s.config = SimpleNamespace(
                    max_num_seqs=1, chunked_prefill=chunked, prefill_step_size=0 if chunked else 2048
                )
                # The scheduler's own except clauses name these; none matches Halt.
                stubs = {
                    name: type(name, (Exception,), {})
                    for name in ("_PrefillAbortedError", "_PrefillEvictionNeeded", "PrefillMemoryExceededError")
                }
                stubs["Scheduler"] = SimpleNamespace(
                    _clear_cache=lambda scheduler: None, _packed_prefill_ready=lambda scheduler: False
                )
                with patch.dict(namespace, stubs), self.assertRaises(Halt):
                    s._schedule_waiting()
                self.assertEqual(seen, [(True, True)])

    async def test_prefix_fence_put_back_leaves_a_cancelled_request_unstarted_and_fenced(self):
        s = self.s
        owned = await self.admitted("synthetic-fenced")
        real_prepare = s._prepare_prefix_cache_for_request

        class Halt(BaseException):
            pass

        observed = []

        def cancel_then_prepare(request):
            # Cancel lands after the queue pop, before the prefix fence.
            owned.cancel()
            observed.append(s._owned_waiting_fenced(request))
            if not observed[-1]:
                raise Halt()  # it would go on to compute; stop the pass here
            return real_prepare(request)

        s._prefill_gate_open.return_value = True
        s._prepare_prefix_cache_for_request = cancel_then_prepare
        s._do_external_prefill = Mock()
        s._build_sampler_and_processors = Mock(return_value=(None, []))
        unmatched = {
            name: type(name, (Exception,), {})
            for name in ("_PrefillAbortedError", "_PrefillEvictionNeeded", "PrefillMemoryExceededError")
        }
        with patch.dict(namespace, unmatched):
            try:
                s._schedule_waiting()
            except Halt:
                pass
        self.assertEqual(observed, [True])
        s._do_external_prefill.assert_not_called()
        s._build_sampler_and_processors.assert_not_called()
        self.assertFalse(owned._started)
        self.assertIs(s.waiting[0], owned.request)
        self.assertTrue(s._owned_waiting_fenced(owned.request))

    async def test_same_id_replacement_with_a_store_future_is_an_identity_mismatch(self):
        owned = await self.admitted("synthetic-replaced")
        self.to_decoding(owned, uid=2)
        self.s.requests["synthetic-replaced"] = Request(
            request_id="synthetic-replaced", prompt="x", sampling_params=owned.request.sampling_params
        )
        self.s._inflight_store_futures["synthetic-replaced"] = object()
        self.assertIsNone(await self.cleanup(owned))
        self.assertEqual(
            (owned._classified_state, owned._cleanup_outcome),
            ("identity_mismatch", "identity_or_admission_unavailable"),
        )

    async def test_a_proof_held_by_another_handle_is_reported_stale(self):
        owned = await self.admitted("synthetic-stale")
        self.to_decoding(owned, uid=3)
        refs = self.refs()
        foreign = namespace["_WaitingCleanupProof"](SimpleNamespace(), owned.request, "synthetic-generation")
        self.s._owned_waiting_cleanup_proofs["synthetic-stale"] = foreign
        self.assertIsNone(await self.cleanup(owned))
        self.assertEqual(owned._cleanup_outcome, "stale_proof")
        self.assertIs(self.s._owned_waiting_cleanup_proofs["synthetic-stale"], foreign)
        self.assertIs(self.s.running["synthetic-stale"], owned.request)
        self.assertEqual(self.refs(), refs)
        fake_mx.synchronize.assert_not_called()

    def test_text_only_bit_never_truth_tests_an_array(self):
        class ArrayLike:
            def __bool__(self):
                raise ValueError("ambiguous truth value")

        empty = dict(
            images=[], videos=(), vlm_image_hash=None, vlm_cache_key_ranges={},
            vlm_inputs_embeds=None, vlm_extra_kwargs=None,
        )
        self.assertTrue(_admitted_text_only(SimpleNamespace(**empty)))
        self.assertFalse(_admitted_text_only(SimpleNamespace(**{**empty, "vlm_inputs_embeds": ArrayLike()})))
        self.assertFalse(_admitted_text_only(SimpleNamespace(**{**empty, "images": ["synthetic"]})))

    async def test_deadline_inside_the_job_refuses_before_any_mutation(self):
        owned = await self.admitted("synthetic-expiring")
        self.to_decoding(owned, uid=4)
        refs = self.refs()
        original = self.s._cleanup_owned_running

        def expire_then_run(handle, request, rid, state):
            handle._stream_effective_deadline = time.monotonic() - 1
            return original(handle, request, rid, state)

        self.s._cleanup_owned_running = expire_then_run
        self.assertIsNone(await self.cleanup(owned))
        self.assertEqual(owned._cleanup_outcome, "failed")
        self.assertIs(self.s.running["synthetic-expiring"], owned.request)
        self.assertEqual(self.s.batch_generator._find_uids([4]), {4: (2, 0)})
        self.assertEqual(self.refs(), refs)
        fake_mx.synchronize.assert_not_called()
        self.s._boundary_snapshot_store.cleanup_request.assert_not_called()

    async def test_decoding_and_chunked_states_must_match_exactly(self):
        damages = {
            "status_not_running": lambda s, request: setattr(request, "status", RequestStatus.WAITING),
            "batch_uid_differs": lambda s, request: setattr(request, "batch_uid", 99),
            "uid_shared_by_another_id": lambda s, request: s.request_id_to_uid.__setitem__("synthetic-shadow", 4),
            "uid_maps_to_another_id": lambda s, request: s.uid_to_request_id.__setitem__(4, "synthetic-shadow"),
            "also_prefilling": lambda s, request: s.prefilling.append(request),
        }
        for name, damage in damages.items():
            with self.subTest(case=name):
                await self.asyncTearDown(); await self.asyncSetUp()
                owned = await self.admitted("synthetic-exact")
                self.to_decoding(owned, uid=4)
                damage(self.s, owned.request)
                refs = self.refs()
                self.assertIsNone(await self.cleanup(owned))
                self.assertEqual(
                    (owned._classified_state, owned._cleanup_outcome), ("unsupported_state", "unsupported")
                )
                self.assertEqual(self.refs(), refs)
                self.assertFalse(self.s._owned_waiting_cleanup_proofs)
        with self.subTest(case="listed_twice_in_prefilling"):
            await self.asyncTearDown(); await self.asyncSetUp()
            owned = await self.admitted("synthetic-exact")
            self.to_chunked(owned)
            self.s.prefilling.append(owned.request)
            self.assertIsNone(await self.cleanup(owned))
            self.assertEqual(owned._classified_state, "unsupported_state")
            self.assertIn("synthetic-exact", self.s._prefill_states)

    async def test_a_non_pristine_queued_request_is_classified_and_never_fenced(self):
        owned = await self.admitted("synthetic-touched")
        owned._waiting_pristine = False
        self.assertIsNone(await self.cleanup(owned))
        self.assertEqual(
            (owned._classified_state, owned._cleanup_outcome), ("non_pristine_waiting", "unsupported")
        )
        self.assertTrue(owned._cancelled)
        self.assertFalse(self.s._owned_waiting_fenced(owned.request))
        self.assertTrue(self.s._select_unfenced_waiting())

    async def test_running_cleanup_at_capacity_is_refused_without_mutation(self):
        owned = await self.admitted("synthetic-full")
        self.to_decoding(owned, uid=4)
        refs = self.refs()
        proof = namespace["_WaitingCleanupProof"]
        for i in range(RunningScheduler._MAX_OWNED_WAITING_PROOFS):
            self.s._owned_waiting_cleanup_proofs[f"synthetic-held-{i}"] = proof(
                SimpleNamespace(), None, "synthetic-generation"
            )
        self.assertIsNone(await self.cleanup(owned))
        self.assertEqual((owned._classified_state, owned._cleanup_outcome), ("decoding", "capacity_refused"))
        self.assertIs(self.s.running["synthetic-full"], owned.request)
        self.assertEqual(self.refs(), refs)
        self.assertNotIn("synthetic-full", self.s._owned_waiting_cleanup_proofs)
        fake_mx.synchronize.assert_not_called()

    async def test_prefill_to_decode_transition_after_cancel_is_cleaned_as_decoding(self):
        owned = await self.admitted("synthetic-transition")
        self.to_chunked(owned)
        owned.cancel()
        s, request, rid = self.s, owned.request, "synthetic-transition"
        # What _insert_prefilled_request does when the last chunk completes.
        s._prefill_states.pop(rid)
        s.prefilling.remove(request)
        request.status = RequestStatus.RUNNING
        request.batch_uid = 6
        s.running[rid] = request
        s.request_id_to_uid[rid] = 6
        s.uid_to_request_id[6] = rid
        s.batch_generator.stages[2].append(6)
        fact = await self.cleanup(owned)
        self.assertEqual((type(fact), fact.state), (RunningCleaned, "decoding"))
        self.assert_retired(owned, uid=6)

    async def test_caller_cancellation_while_the_running_job_runs_keeps_its_future(self):
        owned = await self.admitted("synthetic-held")
        self.to_decoding(owned, uid=4)
        fixture = self.base.fixture.fixture
        entered, release = fixture.event(), fixture.event()

        def held(rid, **kwargs):
            entered.set()
            release.wait(2)

        self.s._boundary_snapshot_store.cleanup_request.side_effect = held
        task = asyncio.create_task(self.cleanup(owned))
        fixture.tasks.append(task)
        await fixture.until(entered.is_set)
        original = owned._cleanup_future
        task.cancel()
        self.assertIsNone(await task)
        self.assertFalse(original.cancelled())
        self.assertFalse(original.done())
        self.assertIs(self.s.running["synthetic-held"], owned.request)
        release.set()
        fact = await self.cleanup(owned)
        self.assertIs(type(fact), RunningCleaned)
        self.assertIs(owned._cleanup_future, original)
        self.s._boundary_snapshot_store.cleanup_request.assert_called_once()
        self.assert_retired(owned, uid=4)

    async def test_a_cleaned_id_is_reusable_only_after_its_collector_is_released(self):
        owned = await self.admitted("synthetic-reused")
        self.to_decoding(owned, uid=3)
        self.s._boundary_cache_snapshots["synthetic-reused"] = object()
        self.assertIs(type(await self.cleanup(owned)), RunningCleaned)
        refs = self.refs()
        self.assertFalse(self.s._do_abort_request("synthetic-reused"))
        self.assertEqual(self.refs(), refs)
        # This stage releases no collector ownership, so the engine still
        # refuses the id; releasing it is the coordinator's job (slice 27).
        with self.assertRaisesRegex(ValueError, "owned_admission_request_id_in_use"):
            await self.admitted("synthetic-reused")
        self.core._output_collectors.pop("synthetic-reused")
        self.core._finished_events.pop("synthetic-reused")
        again = await self.admitted("synthetic-reused")
        self.assertIsNone(self.pcm.get_block_table("synthetic-reused"))
        self.assertNotIn("synthetic-reused", self.s._boundary_cache_snapshots)
        self.to_decoding(again, uid=5)
        fact = await self.cleanup(again)
        self.assertIs(fact.handle, again)
        self.assert_retired(again, uid=5)


if __name__ == "__main__":
    unittest.main()
