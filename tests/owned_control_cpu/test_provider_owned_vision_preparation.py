# SPDX-License-Identifier: Apache-2.0
"""Actual VLM prompt/admit methods; stubbed encoder, device and store."""

import asyncio
import ast
from concurrent.futures import ThreadPoolExecutor
import contextlib
import functools
import importlib
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import test_provider_owned_admission_stage as stage_fixture
import test_provider_owned_admission_plumbing as plumbing
from omlx import owned_preparation as prep
from omlx.owned_admission import OwnedAdmission


class VisionPreparationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = stage_fixture.AdmissionStageTests()
        await self.fixture.asyncSetUp()
        self.core = self.fixture.core
        self.pool = self.fixture.pool
        self.calls = []
        self.active_stream = None
        self.entered = threading.Event()
        self.release = threading.Event()
        self.sync_error = None
        self.hold_sync = False
        self.change_binding = False
        self.context_enter_error = self.context_exit_error = None
        self.error = ValueError("synthetic encoder failed after enqueue")
        self.worker_thread = None
        host = plumbing.VLMHost(
            SimpleNamespace(
                engine=self.core,
                _mlx_executor=self.core._mlx_executor,
                add_request=self.core.add_request,
            )
        )
        engine_class = type("OwnedVLM", (plumbing.VLMNew, plumbing.VLMHost), {})
        self.engine = engine_class.__new__(engine_class)
        self.engine.__dict__.update(host.__dict__)
        self.engine._process_chat_messages = self.process
        self.engine.preflight_chat = self.fixture.engine.preflight_chat
        self.pool.engine = self.engine
        self.engine.core = self.core
        self.namespace = plumbing.VLMNew._stream_chat_prompt.__globals__
        self.saved = {
            key: self.namespace.get(key)
            for key in ("mx", "functools", "prepare_owned_vision")
        }
        self.namespace.update(
            mx=SimpleNamespace(stream=self.stream, synchronize=self.synchronize),
            functools=functools,
        )
        if (
            "prepare_owned_vision"
            in plumbing.VLMNew._stream_chat_prompt.__code__.co_names
        ):
            self.namespace["prepare_owned_vision"] = importlib.import_module(
                "omlx.owned_vision_preparation"
            ).prepare_owned_vision
        self.stage = prep.make_admission_stage(
            lambda: self.pool, stage_fixture.prepare_chat, prep.owned_request_refusal
        )
        self.tasks = []

    async def asyncTearDown(self):
        self.release.set()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.namespace.update(self.saved)
        await self.fixture.asyncTearDown()

    @contextlib.contextmanager
    def stream(self, stream):
        self.calls.append(("enter", stream, threading.get_ident()))
        if self.context_enter_error:
            raise self.context_enter_error
        previous, self.active_stream = self.active_stream, stream
        try:
            yield
        finally:
            self.active_stream = previous
            self.calls.append(("exit", stream, threading.get_ident()))
            if self.context_exit_error:
                raise self.context_exit_error

    def process(self, messages, tools, kwargs):
        self.worker_thread = threading.get_ident()
        self.calls.append(("encode", self.active_stream, self.worker_thread))
        if self.change_binding:
            self.core._mlx_stream = object()
        if self.error:
            raise self.error
        return plumbing.VLMHost._process_chat_messages(
            self.engine, messages, tools, kwargs
        )

    def synchronize(self, stream):
        self.calls.append(("sync", stream, threading.get_ident()))
        self.entered.set()
        if self.hold_sync and not self.release.wait(5):
            raise TimeoutError("test did not release simulated device")
        if self.sync_error:
            raise self.sync_error

    def launch(self, *, seconds=3):
        self.execution = self.fixture.execution(seconds=seconds)
        task = asyncio.create_task(self.stage(self.execution))
        self.tasks.append(task)
        return task

    async def test_encoder_failure_synchronizes_captured_stream_before_lease_release(
        self,
    ):
        stream = self.core._mlx_stream
        task = self.launch()
        with self.assertRaises(stage_fixture.coord.EngineStageUnavailable):
            await task
        self.assertEqual(
            [call[0] for call in self.calls], ["enter", "encode", "sync", "exit"]
        )
        self.assertTrue(all(call[1] is stream for call in self.calls))
        self.assertTrue(all(call[2] == self.worker_thread for call in self.calls))
        self.assertNotEqual(self.worker_thread, threading.get_ident())
        self.assertEqual(self.pool.releases, ["m"])

    async def test_failed_synchronization_retains_lease_and_cannot_report_not_admitted(
        self,
    ):
        self.sync_error = RuntimeError("synthetic device synchronization failed")
        task = self.launch()
        with self.assertRaises(prep.ExecutionUnresolved):
            await task
        self.assertEqual(self.pool.releases, [])
        self.assertIsNotNone(self.execution.lease)
        self.assertFalse(self.core.scheduler.requests)

    async def test_cancellation_retains_lease_until_simulated_gpu_wait_settles(self):
        self.hold_sync = True
        task = self.launch()
        await self.fixture.until(lambda: self.entered.is_set() or task.done())
        self.assertFalse(task.done(), "Python worker failure is not GPU stop proof")
        task.cancel()
        await self.fixture.until(lambda: bool(self.execution.cancellations))
        task.cancel()
        await self.fixture.until(lambda: len(self.execution.cancellations) >= 2)
        self.assertFalse(task.done())
        self.assertEqual(self.pool.releases, [])
        self.release.set()
        with self.assertRaises(stage_fixture.coord.EngineStageUnavailable):
            await task
        self.assertEqual(self.pool.releases, ["m"])

    async def test_binding_drift_during_encoder_failure_retains_lease(self):
        self.change_binding = True
        task = self.launch()
        with self.assertRaises(prep.ExecutionUnresolved):
            await task
        self.assertEqual(self.pool.releases, [])
        self.assertIsNotNone(self.execution.lease)

    async def test_ordinary_chat_preparation_keeps_its_existing_path(self):
        with self.assertRaises(ValueError) as caught:
            await self.engine._stream_chat_prompt([], None, {})
        self.assertIs(caught.exception, self.error)
        self.assertEqual([call[0] for call in self.calls], ["encode"])

    async def test_success_synchronizes_before_scheduler_insertion(self):
        self.error = None
        handle = OwnedAdmission(
            self.core, "synthetic-vision-success", time.monotonic() + 3
        )
        await self.engine.admit_owned_chat(
            [], admission=handle, request_id=handle.binding.request_id
        )
        self.assertEqual(
            [call[0] for call in self.calls], ["enter", "encode", "sync", "exit"]
        )
        self.assertFalse(handle.preparation_unresolved)
        self.assertIn(handle.binding.request_id, self.core.scheduler.requests)

    async def test_preparation_error_is_preserved_when_synchronization_succeeds(self):
        handle = OwnedAdmission(
            self.core, "synthetic-vision-error", time.monotonic() + 3
        )
        with self.assertRaises(ValueError) as caught:
            await self.engine.admit_owned_chat(
                [], admission=handle, request_id=handle.binding.request_id
            )
        self.assertIs(caught.exception, self.error)
        self.assertFalse(handle.preparation_unresolved)
        self.assertFalse(self.core.scheduler.requests)

    async def test_deadline_during_sync_fences_insertion_but_keeps_the_lease(self):
        self.hold_sync = True
        task = self.launch(seconds=0.04)
        await self.fixture.until(lambda: self.entered.is_set())
        await self.fixture.until(lambda: self.execution.preparation_handle.cancelled)
        self.assertFalse(task.done())
        self.assertEqual(self.pool.releases, [])
        self.release.set()
        with self.assertRaises(stage_fixture.coord.EngineStageUnavailable):
            await task
        self.assertEqual(self.pool.releases, ["m"])
        self.assertFalse(self.core.scheduler.requests)

    async def test_context_entry_failure_retains_lease(self):
        self.context_enter_error = RuntimeError("synthetic context entry failed")
        with self.assertRaises(prep.ExecutionUnresolved):
            await self.launch()
        self.assertEqual(self.pool.releases, [])
        self.assertEqual([call[0] for call in self.calls], ["enter"])

    async def test_context_exit_failure_retains_lease(self):
        self.context_exit_error = RuntimeError("synthetic context exit failed")
        with self.assertRaises(prep.ExecutionUnresolved):
            await self.launch()
        self.assertEqual(self.pool.releases, [])

    async def test_coordinator_publishes_unresolved_without_finishing_after_failed_sync(
        self,
    ):
        self.sync_error = RuntimeError("synthetic device sync failed")
        store = SimpleNamespace(
            claim=Mock(
                return_value=SimpleNamespace(request_id="synthetic-coordinator-owned")
            ),
            verify_claim=Mock(),
            finish=Mock(),
        )
        coordinator = stage_fixture.coord.ControlCoordinator(
            store, lambda model: "digest", self.stage, lock=object()
        )
        execution = coordinator.start(
            ("owner", "job", "operation"),
            request_sha256="request-digest",
            model_id="m",
            settings_sha256="digest",
            deadline=time.monotonic() + 3,
            request=stage_fixture.request(),
        )
        self.tasks.append(execution.task)
        await execution.task
        events = []
        while not execution._events.empty():
            events.append((await execution.next_event())[0])
        self.assertEqual(events, ["claimed", "unresolved"])
        store.finish.assert_not_called()
        self.assertEqual(self.pool.releases, [])
        self.assertIsNotNone(execution.lease)

    async def test_foreign_dispatch_executor_is_refused_before_encoder_work(self):
        self.error = None
        handle = OwnedAdmission(
            self.core, "synthetic-foreign-executor", time.monotonic() + 3
        )
        executor = ThreadPoolExecutor(max_workers=1)
        self.engine._engine._mlx_executor = executor
        try:
            with self.assertRaisesRegex(
                ValueError, "owned_preparation_executor_changed"
            ):
                await self.engine.admit_owned_chat(
                    [], admission=handle, request_id=handle.binding.request_id
                )
            self.assertFalse(handle.preparation_unresolved)
            self.assertEqual(self.calls, [])
            self.assertFalse(self.core.scheduler.requests)
        finally:
            executor.shutdown(wait=True)

    def test_ast_harness_dependencies_are_imported_in_the_real_module(self):
        tree = ast.parse((stage_fixture.ROOT / "omlx/engine/vlm.py").read_text())
        imports = {
            alias.asname or alias.name.split(".")[0]
            for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        self.assertTrue({"functools", "mx", "prepare_owned_vision"} <= imports)


if __name__ == "__main__":
    unittest.main()
