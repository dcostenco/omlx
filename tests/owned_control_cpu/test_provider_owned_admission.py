# SPDX-License-Identifier: Apache-2.0
"""CPU ownership/join proof with actual core method and real executor threads."""

import ast
import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
import importlib.abc
import logging
import os
from pathlib import Path
import sys
from threading import Event, Lock, get_ident
from time import monotonic
from types import MethodType, SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union
import unittest
from unittest.mock import Mock
import uuid

ROOT = Path(os.environ["OMLX_CONTROL_ROOT"])
sys.path.insert(0, str(ROOT))


class NoEngineImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (
            fullname == "mlx"
            or fullname.startswith("mlx.")
            or fullname == "omlx.server"
        ):
            raise RuntimeError("real_engine_import_forbidden")


sys.meta_path.insert(0, NoEngineImports())
from omlx.owned_admission import OwnedAdmission
from omlx.request import Request, SamplingParams

source = ROOT / "omlx/engine_core.py"
tree = ast.parse(source.read_text())
cls = next(
    n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "EngineCore"
)
node = next(
    n
    for n in cls.body
    if isinstance(n, ast.AsyncFunctionDef) and n.name == "add_request"
)
ns = dict(
    asyncio=asyncio,
    Any=Any,
    Dict=Dict,
    List=List,
    Optional=Optional,
    Tuple=Tuple,
    Union=Union,
    OwnedAdmission=OwnedAdmission,
    Request=Request,
    SamplingParams=SamplingParams,
    uuid=uuid,
    logger=logging.getLogger("owned-test"),
    RequestOutputCollector=lambda **kw: object(),
    RequestStreamState=lambda **kw: object(),
)
exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), ns)
init_node = next(
    n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
)
init_ns = dict(
    ns,
    EngineConfig=lambda: SimpleNamespace(scheduler_config=None),
    share_command_queue=lambda stream: object(),
    SchedulerConfig=lambda: object(),
    Scheduler=lambda **kwargs: SimpleNamespace(requests={}),
    get_registry=lambda: SimpleNamespace(acquire=Mock()),
    concurrent=SimpleNamespace(
        futures=SimpleNamespace(ThreadPoolExecutor=ThreadPoolExecutor)
    ),
    threading=SimpleNamespace(Lock=Lock),
    mx=SimpleNamespace(
        default_device=lambda: object(), new_thread_local_stream=lambda device: object()
    ),
)
exec(
    compile(ast.Module(body=[init_node], type_ignores=[]), str(source), "exec"), init_ns
)


class OwnedAdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="cpu-owned"
        )
        self.events = []
        self.tasks = []
        self.scheduler = SimpleNamespace(
            requests={}, _specprefill_draft_model=None, abort_request=Mock()
        )
        self.insert_threads = []

        def insert(request):
            self.insert_threads.append(get_ident())
            self.scheduler.requests[request.request_id] = request

        self.scheduler.add_request = Mock(side_effect=insert)
        self.core = SimpleNamespace(
            _engine_id="synthetic-generation",
            _prism_control_generation_id=str(uuid.uuid4()),
            scheduler=self.scheduler,
            _mlx_executor=self.executor,
            _mlx_stream=object(),
            _closed=False,
            _closing=False,
            _output_collectors={},
            _stream_states={},
            _finished_events={},
            config=SimpleNamespace(stream_interval=1),
            _wake_engine_loop=Mock(),
        )

        def cleanup(rid):
            self.core._output_collectors.pop(rid, None)
            self.core._stream_states.pop(rid, None)
            self.core._finished_events.pop(rid, None)

        self.core._cleanup_request = Mock(side_effect=cleanup)
        self.core.add_request = MethodType(ns["add_request"], self.core)

    async def asyncTearDown(self):
        for event in self.events:
            event.set()
        for task in self.tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.executor.shutdown(wait=True)

    def event(self):
        event = Event()
        self.events.append(event)
        return event

    async def until(self, predicate):
        async with asyncio.timeout(2):
            while not predicate():
                await asyncio.sleep(0.001)

    def launch(self, handle):
        task = asyncio.create_task(
            self.core.add_request(
                "synthetic",
                request_id=handle.binding.request_id,
                _owned_admission=handle,
            )
        )
        self.tasks.append(task)
        return task

    async def block_queue(self):
        entered, release = self.event(), self.event()

        def block():
            entered.set()
            release.wait(2)

        future = self.executor.submit(block)
        await self.until(entered.is_set)
        return release, future

    async def test_cancellation_during_insert_retains_late_admission_and_collectors(
        self,
    ):
        entered, release = self.event(), self.event()
        original = self.scheduler.add_request.side_effect

        def held(request):
            entered.set()
            release.wait(2)
            original(request)

        self.scheduler.add_request.side_effect = held
        handle = OwnedAdmission(self.core, "synthetic-owned", monotonic() + 2)
        task = self.launch(handle)
        await self.until(entered.is_set)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(handle.future.done())
        self.assertFalse(handle.future.cancelled())
        self.assertIn("synthetic-owned", self.core._output_collectors)
        self.scheduler.abort_request.assert_not_called()
        self.core._cleanup_request.assert_not_called()
        self.assertIsNone(await handle.join(self.core, monotonic() + 0.02))
        release.set()
        joined = await handle.join(self.core, monotonic() + 2)
        self.assertIs(joined.handle, handle)
        self.assertIs(joined.future, handle.future)
        self.assertTrue(joined.insertion_attempted)
        self.assertTrue(joined.insertion_returned)
        self.assertFalse(joined.admission_failed)
        self.assertIs(self.scheduler.requests["synthetic-owned"], handle.request)
        self.assertNotEqual(self.insert_threads[0], get_ident())
        self.assertFalse(hasattr(joined, "backend_stopped"))

    async def test_cancellation_wins_before_queued_worker_attempt(self):
        release, _ = await self.block_queue()
        handle = OwnedAdmission(self.core, "synthetic-queued", monotonic() + 2)
        task = self.launch(handle)
        await self.until(lambda: handle.future is not None)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(handle.future.cancelled())
        self.assertIsNone(await handle.join(self.core, monotonic() + 0.02))
        self.assertFalse(handle.future.cancelled())
        release.set()
        joined = await handle.join(self.core, monotonic() + 2)
        self.assertFalse(joined.insertion_attempted)
        self.assertFalse(joined.insertion_returned)
        self.scheduler.add_request.assert_not_called()
        self.scheduler.abort_request.assert_not_called()

    async def test_original_deadline_expires_in_queue_without_insertion(self):
        release, _ = await self.block_queue()
        handle = OwnedAdmission(self.core, "synthetic-expired", monotonic() + 0.1)
        task = self.launch(handle)
        with self.assertRaises(TimeoutError):
            await task
        self.assertFalse(handle.future.cancelled())
        self.scheduler.abort_request.assert_not_called()
        self.core._cleanup_request.assert_not_called()
        self.assertIn("synthetic-expired", self.core._output_collectors)
        release.set()
        joined = await handle.join(self.core, monotonic() + 2)
        self.assertFalse(joined.insertion_attempted)
        self.scheduler.add_request.assert_not_called()

    async def test_expiry_inside_insertion_is_unknown_until_future_joins(self):
        entered, release = self.event(), self.event()

        def held(request):
            self.scheduler.requests[request.request_id] = request
            entered.set()
            release.wait(2)

        self.scheduler.add_request.side_effect = held
        handle = OwnedAdmission(self.core, "synthetic-partial", monotonic() + 0.1)
        task = self.launch(handle)
        await self.until(entered.is_set)
        with self.assertRaises(TimeoutError):
            await task
        self.assertFalse(handle.future.done())
        self.assertIn("synthetic-partial", self.scheduler.requests)
        self.assertIn("synthetic-partial", self.core._output_collectors)
        self.scheduler.abort_request.assert_not_called()
        self.assertIsNone(await handle.join(self.core, monotonic() + 0.02))
        release.set()
        self.assertTrue(
            (await handle.join(self.core, monotonic() + 2)).insertion_attempted
        )

    async def test_scheduler_failure_after_partial_insertion_is_distinguished(self):
        def partial(request):
            self.scheduler.requests[request.request_id] = request
            raise ValueError("synthetic_insert_failed")

        self.scheduler.add_request.side_effect = partial
        handle = OwnedAdmission(self.core, "synthetic-failed", monotonic() + 2)
        with self.assertRaisesRegex(ValueError, "synthetic_insert_failed"):
            await self.core.add_request(
                "synthetic", request_id="synthetic-failed", _owned_admission=handle
            )
        joined = await handle.join(self.core, monotonic() + 2)
        self.assertTrue(joined.insertion_attempted)
        self.assertFalse(joined.insertion_returned)
        self.assertTrue(joined.admission_failed)
        self.assertIn("synthetic-failed", self.scheduler.requests)
        self.scheduler.abort_request.assert_not_called()
        self.core._cleanup_request.assert_not_called()
        self.assertIn("synthetic-failed", self.core._output_collectors)

    async def test_join_cancellation_does_not_cancel_real_future(self):
        release, _ = await self.block_queue()
        handle = OwnedAdmission(self.core, "synthetic-join", monotonic() + 2)
        admission = self.launch(handle)
        await self.until(lambda: handle.future is not None)
        joining = asyncio.create_task(handle.join(self.core, monotonic() + 2))
        self.tasks.append(joining)
        await asyncio.sleep(0)
        joining.cancel()
        self.assertIsNone(await joining)
        self.assertFalse(handle.future.cancelled())
        release.set()
        await admission
        self.assertIsNotNone(await handle.join(self.core, monotonic() + 2))

    async def test_duplicate_handle_is_rejected_without_second_submission(self):
        handle = OwnedAdmission(self.core, "synthetic-once", monotonic() + 2)
        await self.core.add_request(
            "synthetic", request_id="synthetic-once", _owned_admission=handle
        )
        retained = handle.future
        with self.assertRaises(ValueError):
            await self.core.add_request(
                "synthetic", request_id="synthetic-once", _owned_admission=handle
            )
        self.assertIs(handle.future, retained)
        self.assertEqual(self.scheduler.add_request.call_count, 1)

    async def test_foreign_core_forged_handle_and_request_id_are_rejected(self):
        handle = OwnedAdmission(self.core, "synthetic-binding", monotonic() + 2)
        for invalid in [{}, SimpleNamespace(binding=handle.binding)]:
            with self.assertRaises(ValueError):
                await self.core.add_request(
                    "synthetic",
                    request_id="synthetic-binding",
                    _owned_admission=invalid,
                )
        with self.assertRaises(ValueError):
            await self.core.add_request(
                "synthetic", request_id="wrong", _owned_admission=handle
            )
        foreign = SimpleNamespace(**vars(self.core))
        self.assertIsNone(await handle.join(foreign, monotonic() + 2))
        self.assertFalse(self.core._output_collectors)
        self.assertIsNone(handle.future)

    async def test_changed_generation_executor_scheduler_or_stream_refuses_join(self):
        for attribute in [
            "_engine_id",
            "_prism_control_generation_id",
            "_mlx_executor",
            "scheduler",
            "_mlx_stream",
        ]:
            with self.subTest(attribute=attribute):
                rid = "synthetic-" + attribute
                handle = OwnedAdmission(self.core, rid, monotonic() + 2)
                await self.core.add_request(
                    "synthetic", request_id=rid, _owned_admission=handle
                )
                old = getattr(self.core, attribute)
                setattr(self.core, attribute, object())
                try:
                    self.assertIsNone(await handle.join(self.core, monotonic() + 2))
                finally:
                    setattr(self.core, attribute, old)

    async def test_missing_or_replaced_future_has_no_join_fact(self):
        handle = OwnedAdmission(self.core, "synthetic-tamper", monotonic() + 2)
        self.assertIsNone(await handle.join(self.core, monotonic() + 2))
        await self.core.add_request(
            "synthetic", request_id="synthetic-tamper", _owned_admission=handle
        )
        replacement = Future()
        replacement.set_result(None)
        handle._future = replacement
        self.assertIsNone(await handle.join(self.core, monotonic() + 2))

    async def test_rejected_submit_cannot_be_retried_or_prove_joined(self):
        handle = OwnedAdmission(self.core, "synthetic-rejected", monotonic() + 2)
        self.executor.shutdown(wait=True)
        with self.assertRaises(RuntimeError):
            await self.core.add_request(
                "synthetic", request_id="synthetic-rejected", _owned_admission=handle
            )
        self.assertIsNone(await handle.join(self.core, monotonic() + 2))
        self.assertIsNotNone(handle.request)
        self.assertIsNone(handle.future)
        self.scheduler.abort_request.assert_not_called()

    async def test_ordinary_cancellation_keeps_existing_abort_and_cleanup(self):
        entered, release = self.event(), self.event()

        def held(request):
            entered.set()
            release.wait(2)

        self.scheduler.add_request.side_effect = held
        task = asyncio.create_task(
            self.core.add_request("synthetic", request_id="synthetic-ordinary")
        )
        self.tasks.append(task)
        await self.until(entered.is_set)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.scheduler.abort_request.assert_called_once_with("synthetic-ordinary")
        self.core._cleanup_request.assert_called_once_with("synthetic-ordinary")
        self.assertFalse(self.core._output_collectors)

    async def test_unrelated_request_is_not_aborted_or_modified(self):
        other = Request(
            request_id="synthetic-other",
            prompt="other",
            sampling_params=SamplingParams(),
        )
        self.scheduler.requests[other.request_id] = other
        handle = OwnedAdmission(self.core, "synthetic-owned", monotonic() + 2)
        await self.core.add_request(
            "synthetic", request_id="synthetic-owned", _owned_admission=handle
        )
        self.assertIsNotNone(await handle.join(self.core, monotonic() + 2))
        self.assertIs(self.scheduler.requests[other.request_id], other)
        self.scheduler.abort_request.assert_not_called()

    async def test_expired_or_nonfinite_deadlines_are_rejected_before_setup(self):
        for deadline in [monotonic() - 1, float("inf"), float("nan"), True]:
            with self.assertRaises(ValueError):
                OwnedAdmission(self.core, "synthetic-invalid", deadline)
        handle = OwnedAdmission(
            self.core, "synthetic-expired-setup", monotonic() + 0.01
        )
        await asyncio.sleep(0.02)
        with self.assertRaises(TimeoutError):
            await self.core.add_request(
                "synthetic",
                request_id=handle.binding.request_id,
                _owned_admission=handle,
            )
        self.assertFalse(self.core._output_collectors)
        self.assertIsNone(handle.future)

    async def test_actual_constructor_nonce_stays_unique_with_reused_legacy_label(self):
        cores = [SimpleNamespace(), SimpleNamespace()]
        try:
            for core in cores:
                init_ns["__init__"](
                    core, object(), object(), engine_id="synthetic-reused-label"
                )
                self.assertEqual(core._engine_id, "synthetic-reused-label")
                self.assertEqual(
                    uuid.UUID(core._prism_control_generation_id).version, 4
                )
            handles = [
                OwnedAdmission(core, "synthetic-same-id", monotonic() + 2)
                for core in cores
            ]
            self.assertNotEqual(
                handles[0].binding.generation_id, handles[1].binding.generation_id
            )
            self.assertEqual(
                handles[0].binding.generation_id, cores[0]._prism_control_generation_id
            )
            self.assertIsNone(await handles[0].join(cores[1], monotonic() + 2))
        finally:
            for core in cores:
                executor = getattr(core, "_mlx_executor", None)
                if executor is not None:
                    executor.shutdown(wait=True)


if __name__ == "__main__":
    unittest.main()
