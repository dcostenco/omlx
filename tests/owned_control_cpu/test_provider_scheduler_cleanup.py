# SPDX-License-Identifier: Apache-2.0
"""CPU-only exact scheduler-method tests; no server/MLX import or stop proof."""

import ast
import concurrent.futures
from collections import deque
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import threading
from types import MethodType, SimpleNamespace
from typing import Any
import unittest
from unittest.mock import Mock

from omlx.owned_admission import OwnedAdmission

ROOT = Path(os.environ["OMLX_CONTROL_ROOT"])
SOURCE = ROOT / "omlx/scheduler.py"
tree = ast.parse(SOURCE.read_text())
scheduler = next(
    n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler"
)
names = {
    "_track_deferred_cleanup",
    "_untrack_deferred_cleanup",
    "_drain_tracked_deferred_cleanup",
    "_drain_pending_async_removes",
    # 28: the drain records a failed row removal of owned no-retry work.
    "_owned_no_retry",
}
methods = [
    n for n in scheduler.body if isinstance(n, ast.FunctionDef) and n.name in names
]
assert len(methods) == len(names), "exact scheduler methods must exist"
tracking = next(
    n
    for n in tree.body
    if isinstance(n, ast.ClassDef) and n.name == "_DeferredCleanupTracking"
)
limit = next(
    n
    for n in scheduler.body
    if isinstance(n, ast.Assign)
    and any(
        isinstance(t, ast.Name) and t.id == "_MAX_TRACKED_DEFERRED_CLEANUPS"
        for t in n.targets
    )
)
ns = dict(
    __name__=__name__,
    Any=Any,
    dataclass=dataclass,
    deque=deque,
    concurrent=concurrent,
    logger=logging.getLogger("cpu-cleanup"),
    mx=SimpleNamespace(synchronize=Mock()),
    _safe_sync_stream=Mock(),
    is_command_buffer_error=lambda error: False,
    _unregister_uid_row=Mock(),
    OwnedAdmission=OwnedAdmission,
)
exec(
    compile(
        ast.fix_missing_locations(
            ast.Module(
                body=[
                    tracking,
                    ast.ClassDef(
                        name="TestedScheduler",
                        bases=[],
                        keywords=[],
                        body=[limit, *methods],
                        decorator_list=[],
                    ),
                ],
                type_ignores=[],
            )
        ),
        str(SOURCE),
        "exec",
    ),
    ns,
)
TestedScheduler = ns["TestedScheduler"]


class SchedulerCleanupTests(unittest.TestCase):
    def setUp(self):
        self.sync = ns["mx"].synchronize = Mock()
        ns["_safe_sync_stream"] = Mock()
        ns["_unregister_uid_row"] = Mock()
        self.s = TestedScheduler()
        self.s._pending_async_removes = deque()
        self.s._tracked_deferred_cleanups = {}
        self.s._inflight_store_futures = {}
        self.s._inflight_store_info = {}
        self.s.requests = {}
        self.s.uid_to_request_id = {}
        self.s.request_id_to_uid = {}
        self.s._stream = object()
        self.s.model = SimpleNamespace(unregister_rope_delta=Mock())
        self.s._remove_uid_from_active_batch = Mock()
        self.s._boundary_snapshot_store = SimpleNamespace(cleanup_request=Mock())
        self.s._clear_request_admission_bookkeeping = Mock()
        self.s._store_cache_gate = SimpleNamespace(note_done=Mock())
        self.s._schedule_deferred_metal_clear = Mock()
        # Upstream (0.7.1.dev2+) restarts its memory-admission timer when a
        # drain completes work; that bookkeeping is not the cleanup under test.
        self.s._note_memory_admission_progress = Mock()
        self.req, self.future, self.entry = self.add("original", 17)
        self.token = self.s._track_deferred_cleanup("original")

    def add(self, rid, uid):
        req = SimpleNamespace(_extracted_cache=object(), prompt_cache=object())
        future = concurrent.futures.Future()
        entry = (uid, rid, future)
        self.s.requests[rid] = req
        self.s._pending_async_removes.append(entry)
        self.s._inflight_store_futures[rid] = future
        self.s._inflight_store_info[rid] = object()
        self.s.uid_to_request_id[uid] = rid
        self.s.request_id_to_uid[rid] = uid
        return req, future, entry

    def assert_retained(self):
        self.assertIs(self.s._pending_async_removes[0], self.entry)
        self.assertIs(self.s.requests["original"], self.req)
        self.assertIs(self.s._inflight_store_futures["original"], self.future)
        self.assertEqual(self.s.uid_to_request_id[17], "original")
        self.assertIsNotNone(self.req._extracted_cache)
        self.assertIsNotNone(self.req.prompt_cache)

    def test_unfinished_store_is_retained_without_cleanup(self):
        self.assertFalse(self.s._drain_pending_async_removes())
        self.assert_retained()
        self.assertEqual(self.token.state, "pending")
        self.sync.assert_not_called()
        self.s._store_cache_gate.note_done.assert_not_called()

    def test_success_publishes_after_cleanup_and_releases_once(self):
        self.future.set_result(None)
        order = []
        self.sync.side_effect = lambda stream: order.append("sync")
        self.s._remove_uid_from_active_batch.side_effect = lambda uid: order.append(
            "remove"
        )
        self.s._boundary_snapshot_store.cleanup_request.side_effect = (
            lambda rid, **kwargs: order.append("snapshot")
        )
        self.s._store_cache_gate.note_done.side_effect = lambda: order.append("gate")
        self.assertTrue(self.s._drain_pending_async_removes())
        self.assertEqual(order, ["sync", "remove", "snapshot", "gate"])
        self.assertEqual(self.token.state, "cleaned")
        self.assertNotIn("original", self.s.requests)
        self.assertIsNone(self.req._extracted_cache)
        self.sync.assert_called_once_with(self.s._stream)
        self.assertFalse(self.s._drain_pending_async_removes())
        self.s._store_cache_gate.note_done.assert_called_once()
        self.s._untrack_deferred_cleanup("original", self.token)
        self.assertFalse(self.s._tracked_deferred_cleanups)

    def test_failed_cleanup_retains_identity_and_is_never_retried(self):
        stages = [
            self.sync,
            self.s._remove_uid_from_active_batch,
            self.s.model.unregister_rope_delta,
            ns["_unregister_uid_row"],
            self.s._boundary_snapshot_store.cleanup_request,
            self.s._clear_request_admission_bookkeeping,
            self.s._store_cache_gate.note_done,
        ]
        for index in range(len(stages)):
            with self.subTest(stage=index):
                self.setUp()
                stage = [
                    self.sync,
                    self.s._remove_uid_from_active_batch,
                    self.s.model.unregister_rope_delta,
                    ns["_unregister_uid_row"],
                    self.s._boundary_snapshot_store.cleanup_request,
                    self.s._clear_request_admission_bookkeeping,
                    self.s._store_cache_gate.note_done,
                ][index]
                stage.side_effect = RuntimeError("synthetic_cleanup_failure")
                self.future.set_result(None)
                self.assertFalse(self.s._drain_pending_async_removes())
                self.assertEqual(self.token.state, "failed")
                self.assert_retained()
                calls = stage.call_count
                self.assertFalse(self.s._drain_pending_async_removes())
                self.assertEqual(stage.call_count, calls)
                with self.assertRaises(ValueError):
                    self.s._untrack_deferred_cleanup("original", self.token)

    def test_missing_stream_and_no_stream_exception_are_failures(self):
        for missing in [True, False]:
            with self.subTest(missing=missing):
                self.setUp()
                self.future.set_result(None)
                if missing:
                    self.s._stream = None
                else:
                    self.sync.side_effect = RuntimeError("no Stream in current thread")
                self.assertFalse(self.s._drain_pending_async_removes())
                self.assertEqual(self.token.state, "failed")
                self.assert_retained()

    def test_worker_failure_or_cancellation_is_sticky(self):
        for cancel in [False, True]:
            with self.subTest(cancel=cancel):
                self.setUp()
                if cancel:
                    self.future.cancel()
                else:
                    self.future.set_exception(RuntimeError("synthetic_worker_failure"))
                self.assertFalse(self.s._drain_pending_async_removes())
                self.assertEqual(self.token.state, "failed")
                self.assert_retained()
                self.sync.assert_not_called()

    def test_uid_reused_by_other_request_is_not_removed(self):
        self.s.uid_to_request_id[17] = "unrelated"
        self.s.request_id_to_uid["unrelated"] = 17
        self.future.set_result(None)
        self.assertTrue(self.s._drain_pending_async_removes())
        self.assertEqual(self.s.uid_to_request_id[17], "unrelated")
        self.assertEqual(self.s.request_id_to_uid["unrelated"], 17)
        self.s._remove_uid_from_active_batch.assert_not_called()
        ns["_unregister_uid_row"].assert_not_called()

    def test_a_tracked_store_only_entry_with_no_batch_row_cleans(self):
        # Upstream 0.7.1.dev2+ queues (None, request_id, future) for a store
        # with no batch row. request_id_to_uid.get(rid) == None must not lead
        # the tracked drain to delete an absent mapping (KeyError -> failed).
        req = SimpleNamespace(_extracted_cache=object(), prompt_cache=object())
        future = concurrent.futures.Future()
        self.s.requests["store-only"] = req
        self.s._pending_async_removes.append((None, "store-only", future))
        self.s._inflight_store_futures["store-only"] = future
        self.s._inflight_store_info["store-only"] = object()
        token = self.s._track_deferred_cleanup("store-only")
        self.future.set_result(None)
        future.set_result(None)
        self.s._drain_pending_async_removes()
        self.assertEqual(token.state, "cleaned")
        self.assertNotIn("store-only", self.s.requests)
        self.assertNotIn("store-only", self.s._inflight_store_futures)
        self.assertNotIn(None, [c.args[0] for c in self.s._remove_uid_from_active_batch.call_args_list])
        self.assertEqual(self.token.state, "cleaned")

    def test_reused_request_id_or_replaced_future_does_not_authorize_cleanup(self):
        for replace_request in [False, True]:
            with self.subTest(replace_request=replace_request):
                self.setUp()
                self.future.set_result(None)
                if replace_request:
                    replacement = SimpleNamespace(
                        _extracted_cache=object(), prompt_cache=object()
                    )
                    self.s.requests["original"] = replacement
                else:
                    replacement = concurrent.futures.Future()
                    self.s._inflight_store_futures["original"] = replacement
                self.assertFalse(self.s._drain_pending_async_removes())
                self.assertEqual(self.token.state, "failed")
                self.assertIs(self.s._pending_async_removes[0], self.entry)
                self.assertIsNotNone(self.req._extracted_cache)
                self.sync.assert_not_called()

    def test_retained_failure_allows_unrelated_request_to_drain(self):
        self.future.set_exception(RuntimeError("synthetic_worker_failure"))
        other, future, _ = self.add("unrelated", 19)
        future.set_result(None)
        self.assertTrue(self.s._drain_pending_async_removes())
        self.assert_retained()
        self.assertNotIn("unrelated", self.s.requests)
        self.assertIsNone(other._extracted_cache)
        self.s._remove_uid_from_active_batch.assert_called_once_with(19)
        self.s._store_cache_gate.note_done.assert_called_once()

    def test_arm_requires_unique_present_request_and_exact_future(self):
        with self.assertRaises(ValueError):
            self.s._track_deferred_cleanup("original")
        with self.assertRaises(ValueError):
            self.s._track_deferred_cleanup("missing")
        _, _, entry = self.add("ambiguous", 20)
        self.s._pending_async_removes.append(entry)
        with self.assertRaises(ValueError):
            self.s._track_deferred_cleanup("ambiguous")
        self.add("wrong-future", 21)
        self.s._inflight_store_futures["wrong-future"] = concurrent.futures.Future()
        with self.assertRaises(ValueError):
            self.s._track_deferred_cleanup("wrong-future")

    def test_tracking_is_bounded_and_stale_token_cannot_release(self):
        self.s._MAX_TRACKED_DEFERRED_CLEANUPS = 1
        self.add("overflow", 20)
        with self.assertRaises(ValueError):
            self.s._track_deferred_cleanup("overflow")
        self.future.set_result(None)
        self.s._drain_pending_async_removes()
        with self.assertRaises(ValueError):
            self.s._untrack_deferred_cleanup("original", object())

    def test_ordinary_untracked_cleanup_keeps_existing_behavior(self):
        del self.s._tracked_deferred_cleanups["original"]
        self.future.set_result(None)
        ns["_safe_sync_stream"].side_effect = RuntimeError("synthetic_sync_failure")
        self.assertTrue(self.s._drain_pending_async_removes())
        self.assertNotIn("original", self.s.requests)
        self.sync.assert_not_called()

    def test_extraction_does_not_execute_imports_or_enable_stop_capability(self):
        for n in methods:
            self.assertFalse(
                any(isinstance(x, (ast.Import, ast.ImportFrom)) for x in ast.walk(n))
            )
        self.assertNotIn(
            "backend_stopped",
            {x.id for n in methods for x in ast.walk(n) if isinstance(x, ast.Name)},
        )
        # Only exact method bodies are executed, never Scheduler construction,
        # engine imports, model generation or the actual MLX synchronization.
        self.assertEqual(
            Path(
                ns["TestedScheduler"]._drain_pending_async_removes.__code__.co_filename
            ),
            SOURCE,
        )

    def actual_snapshot_store(self, acquired=True, deletion_error=False, safe=True):
        # Execute the actual method with CPU-only dependencies. rmtree is always
        # a Mock; this harness never deletes any snapshot file.
        source = ROOT / "omlx/cache/boundary_snapshot_store.py"
        tree = ast.parse(source.read_text())
        method = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "cleanup_request"
        )
        deletion = Mock(
            side_effect=(
                OSError("synthetic_deletion_failure") if deletion_error else None
            )
        )
        local = dict(shutil=SimpleNamespace(rmtree=deletion), logger=Mock())
        exec(
            compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"),
            local,
        )
        store = SimpleNamespace(
            _pending_cond=threading.Condition(),
            _pending_writes={},
            _registry_lock=threading.Lock(),
            _file_registry={},
            _writer_busy=Mock(),
            _CLEANUP_REQUEST_TIMEOUT_S=0.01,
            _cancelled_lock=threading.Lock(),
            _cancelled_requests={},
            _request_dir=lambda rid: SimpleNamespace(exists=lambda: True),
            _is_safe_snapshot_path=lambda path: safe,
        )
        store._writer_busy.acquire.return_value = acquired
        store.cleanup_request = MethodType(local["cleanup_request"], store)
        return store, deletion

    def test_actual_snapshot_helper_failures_cannot_publish_success(self):
        for acquired, deletion_error, safe in [
            (False, False, True),
            (True, True, True),
            (True, False, False),
        ]:
            with self.subTest(
                acquired=acquired, deletion_error=deletion_error, safe=safe
            ):
                self.setUp()
                self.future.set_result(None)
                store, deletion = self.actual_snapshot_store(
                    acquired, deletion_error, safe
                )
                self.s._boundary_snapshot_store = store
                self.assertFalse(self.s._drain_pending_async_removes())
                self.assertEqual(self.token.state, "failed")
                self.assert_retained()
                self.s._store_cache_gate.note_done.assert_not_called()
                before = deletion.call_count
                store._writer_busy.acquire.return_value = True
                deletion.side_effect = None
                self.assertFalse(self.s._drain_pending_async_removes())
                self.assertEqual(deletion.call_count, before)

    def test_actual_snapshot_helper_strict_success_releases_writer(self):
        store, deletion = self.actual_snapshot_store()
        store.cleanup_request("synthetic", strict=True)
        deletion.assert_called_once()
        store._writer_busy.release.assert_called_once()

    def test_actual_snapshot_helper_legacy_swallows_errors_as_before(self):
        for acquired in [False, True]:
            with self.subTest(acquired=acquired):
                store, deletion = self.actual_snapshot_store(
                    acquired, deletion_error=True
                )
                self.assertIsNone(store.cleanup_request("synthetic"))
                deletion.assert_called_once()
                self.assertEqual(store._writer_busy.release.call_count, int(acquired))


if __name__ == "__main__":
    unittest.main()
