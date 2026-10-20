# SPDX-License-Identifier: Apache-2.0
"""CPU stream-stage proof: actual core methods, real executor, device stub."""

import ast
import asyncio
from concurrent.futures import Future
from threading import get_ident
import time
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock

import test_provider_owned_admission as admission_fixture
from omlx.owned_admission import OwnedAdmission, RunningCleaned, StreamSynchronized, WaitingCleaned

SOURCE = admission_fixture.ROOT / "omlx/engine_core.py"
tree = ast.parse(SOURCE.read_text())
cls = next(
    n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "EngineCore"
)
methods = [
    n
    for n in cls.body
    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    and n.name in {"_synchronize_owned_stream", "_owned_stream_barrier"}
]
assert len(methods) == 2
namespace = dict(
    admission_fixture.ns,
    time=time,
    StreamSynchronized=StreamSynchronized,
    WaitingCleaned=WaitingCleaned,
    RunningCleaned=RunningCleaned,
    mx=SimpleNamespace(synchronize=Mock()),
)
exec(compile(ast.Module(body=methods, type_ignores=[]), str(SOURCE), "exec"), namespace)


class StreamBarrierTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = admission_fixture.OwnedAdmissionTests()
        await self.fixture.asyncSetUp()
        self.core = self.fixture.core
        self.calls = []

        def sync(stream):
            self.calls.append((stream, get_ident()))

        self.sync = namespace["mx"].synchronize = Mock(side_effect=sync)
        for name in ["_synchronize_owned_stream", "_owned_stream_barrier"]:
            setattr(self.core, name, MethodType(namespace[name], self.core))

    async def asyncTearDown(self):
        await self.fixture.asyncTearDown()

    async def admitted(self):
        handle = OwnedAdmission(
            self.core, "synthetic-stream-owned", time.monotonic() + 2
        )
        await self.core.add_request(
            "synthetic", request_id=handle.binding.request_id, _owned_admission=handle
        )
        return handle

    def launch(self, handle, deadline):
        task = asyncio.create_task(self.core._owned_stream_barrier(handle, deadline))
        self.fixture.tasks.append(task)
        return task

    async def test_actual_sync_runs_on_bound_executor_stream_and_does_not_stop(self):
        handle = await self.admitted()
        fact = await self.core._owned_stream_barrier(handle, time.monotonic() + 2)
        self.assertIs(fact.handle, handle)
        self.assertIs(fact.future, handle._stream_future)
        self.assertEqual(fact.generation_id, self.core._prism_control_generation_id)
        self.assertEqual(self.calls[0][0], handle.binding.stream)
        self.assertNotEqual(self.calls[0][1], get_ident())
        self.assertIn(handle.binding.request_id, self.core.scheduler.requests)
        self.assertFalse(hasattr(fact, "backend_stopped"))
        self.core.scheduler.abort_request.assert_not_called()
        self.core._cleanup_request.assert_not_called()

    async def test_queued_sync_cancellation_keeps_future_and_retry_dispatches_once(
        self,
    ):
        handle = await self.admitted()
        release, _ = await self.fixture.block_queue()
        task = self.launch(handle, time.monotonic() + 2)
        await self.fixture.until(lambda: handle._stream_future is not None)
        original = handle._stream_future
        task.cancel()
        self.assertIsNone(await task)
        self.assertFalse(original.cancelled())
        self.assertFalse(original.done())
        self.sync.assert_not_called()
        release.set()
        fact = await self.core._owned_stream_barrier(handle, time.monotonic() + 2)
        self.assertIs(fact.future, original)
        self.sync.assert_called_once_with(handle.binding.stream)

    async def test_running_sync_cancellation_keeps_owned_work(self):
        handle = await self.admitted()
        entered, release = self.fixture.event(), self.fixture.event()

        def held(stream):
            entered.set()
            release.wait(2)

        self.sync.side_effect = held
        task = self.launch(handle, time.monotonic() + 2)
        await self.fixture.until(entered.is_set)
        task.cancel()
        self.assertIsNone(await task)
        self.assertFalse(handle._stream_future.done())
        self.assertFalse(handle._stream_future.cancelled())
        release.set()
        self.assertIsNotNone(
            await self.core._owned_stream_barrier(handle, time.monotonic() + 2)
        )
        self.assertEqual(self.sync.call_count, 1)

    async def test_deadline_cannot_be_extended_after_stalled_sync(self):
        handle = await self.admitted()
        entered, release = self.fixture.event(), self.fixture.event()
        self.sync.side_effect = lambda stream: (entered.set(), release.wait(2))
        deadline = time.monotonic() + 0.05
        task = self.launch(handle, deadline)
        await self.fixture.until(entered.is_set)
        self.assertIsNone(await task)
        original = handle._stream_future
        self.assertFalse(original.cancelled())
        self.assertFalse(original.done())
        release.set()
        self.assertIsNone(
            await self.core._owned_stream_barrier(handle, time.monotonic() + 2)
        )
        self.assertIs(handle._stream_future, original)
        self.assertEqual(handle._stream_pinned_deadline, deadline)
        self.assertEqual(self.sync.call_count, 1)

    async def test_no_stream_sync_error_is_sticky_without_legacy_tolerance(self):
        handle = await self.admitted()
        self.sync.side_effect = RuntimeError("no Stream in current thread")
        self.assertIsNone(
            await self.core._owned_stream_barrier(handle, time.monotonic() + 2)
        )
        original = handle._stream_future
        self.assertIsNone(
            await self.core._owned_stream_barrier(handle, time.monotonic() + 2)
        )
        self.assertIs(handle._stream_future, original)
        self.sync.assert_called_once()

    async def test_foreign_generation_reference_and_missing_stream_refuse(self):
        for attribute in [
            "_prism_control_generation_id",
            "_mlx_stream",
            "scheduler",
            "_mlx_executor",
        ]:
            with self.subTest(attribute=attribute):
                handle = OwnedAdmission(
                    self.core, "synthetic-" + attribute, time.monotonic() + 2
                )
                await self.core.add_request(
                    "synthetic",
                    request_id=handle.binding.request_id,
                    _owned_admission=handle,
                )
                original = getattr(self.core, attribute)
                setattr(
                    self.core,
                    attribute,
                    None if attribute == "_mlx_stream" else object(),
                )
                try:
                    self.assertIsNone(
                        await self.core._owned_stream_barrier(
                            handle, time.monotonic() + 2
                        )
                    )
                    self.assertIsNone(handle._stream_future)
                finally:
                    setattr(self.core, attribute, original)
        self.sync.assert_not_called()

    async def test_changed_nonce_after_executor_callback_before_host_acceptance(self):
        handle = await self.admitted()
        original_submit = self.fixture.executor.submit

        def submit(function, *args, **kwargs):
            def after_sync():
                result = function(*args, **kwargs)
                self.core._prism_control_generation_id = "synthetic-replaced-after-sync"
                return result

            return original_submit(after_sync)

        self.fixture.executor.submit = submit
        self.assertIsNone(
            await self.core._owned_stream_barrier(handle, time.monotonic() + 2)
        )
        self.assertTrue(handle._stream_future.done())
        self.assertIsNone(handle._stream_future.exception())
        self.sync.assert_called_once()

    async def test_concurrent_retries_share_one_synchronization_future(self):
        handle = await self.admitted()
        deadline = time.monotonic() + 2
        facts = await asyncio.gather(
            *(self.core._owned_stream_barrier(handle, deadline) for _ in range(3))
        )
        self.assertTrue(all(fact is not None for fact in facts))
        self.assertTrue(all(fact.future is handle._stream_future for fact in facts))
        self.sync.assert_called_once()

    async def test_rejected_submission_is_sticky_and_never_retried(self):
        handle = await self.admitted()
        original_submit = self.fixture.executor.submit
        rejected = Mock(side_effect=RuntimeError("synthetic_rejected_submission"))
        self.fixture.executor.submit = rejected
        self.assertIsNone(
            await self.core._owned_stream_barrier(handle, time.monotonic() + 2)
        )
        self.fixture.executor.submit = original_submit
        self.assertIsNone(
            await self.core._owned_stream_barrier(handle, time.monotonic() + 2)
        )
        rejected.assert_called_once()
        self.sync.assert_not_called()

    async def test_shorter_expired_deadline_refuses_new_fact_after_success(self):
        handle = await self.admitted()
        fact = await self.core._owned_stream_barrier(handle, time.monotonic() + 2)
        self.assertIsNotNone(fact)
        self.assertIsNone(
            await self.core._owned_stream_barrier(handle, time.monotonic() - 1)
        )
        self.assertIsNone(
            await self.core._owned_stream_barrier(handle, time.monotonic() + 20)
        )
        self.sync.assert_called_once()

    async def test_replaced_future_refuses_success_and_does_not_redispatch(self):
        handle = await self.admitted()
        self.assertIsNotNone(
            await self.core._owned_stream_barrier(handle, time.monotonic() + 2)
        )
        fake = Future()
        fake.set_result(handle.binding.stream)
        handle._stream_future = fake
        self.assertIsNone(
            await self.core._owned_stream_barrier(handle, time.monotonic() + 2)
        )
        self.sync.assert_called_once()

    async def test_joins_admission_before_sync_dispatch_and_preserves_other_request(
        self,
    ):
        release, _ = await self.fixture.block_queue()
        handle = OwnedAdmission(
            self.core, "synthetic-pending-admission", time.monotonic() + 2
        )
        admission_task = self.fixture.launch(handle)
        await self.fixture.until(lambda: handle.future is not None)
        other = object()
        self.core.scheduler.requests["synthetic-other"] = other
        syncing = self.launch(handle, time.monotonic() + 2)
        await asyncio.sleep(0.01)
        self.assertIsNone(handle._stream_future)
        self.sync.assert_not_called()
        release.set()
        await admission_task
        self.assertIsNotNone(await syncing)
        self.assertIs(self.core.scheduler.requests["synthetic-other"], other)
        self.core.scheduler.abort_request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
