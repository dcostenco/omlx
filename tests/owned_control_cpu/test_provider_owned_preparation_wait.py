# SPDX-License-Identifier: Apache-2.0
"""Preparation must settle real executor work before its caller releases a lease."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from time import monotonic
from types import SimpleNamespace
import unittest
from unittest import mock

from omlx.owned_preparation_wait import wait_owned_preparation


class PreparationWaitTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.started = Event()
        self.release = Event()
        self.tasks = []
        self.execution = SimpleNamespace(deadline=monotonic() + 5, cancellations=[])
        self.handle = SimpleNamespace(cancelled=False)
        self.handle.cancel = lambda: setattr(self.handle, "cancelled", True)
        self.completed = False
        self.admitted = False
        self.lease_released = False

    async def asyncTearDown(self):
        self.release.set()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.executor.shutdown(wait=True)

    async def until(self, predicate):
        async with asyncio.timeout(2):
            while not predicate():
                await asyncio.sleep(0.001)

    def work(self):
        self.started.set()
        if not self.release.wait(3):
            raise TimeoutError("test did not release worker")
        self.completed = True
        return "prepared"

    async def prepare_and_admit(self, *, fail=False):
        result = await asyncio.get_running_loop().run_in_executor(
            self.executor, self.work
        )
        if fail:
            raise RuntimeError("preparation failed after work")
        if self.handle.cancelled:
            raise asyncio.CancelledError()
        self.admitted = True
        return result

    def launch(self, *, fail=False):
        async def owner():
            try:
                return await wait_owned_preparation(
                    self.execution, self.handle, self.prepare_and_admit(fail=fail)
                )
            finally:
                self.lease_released = True

        task = asyncio.create_task(owner())
        self.tasks.append(task)
        return task

    async def test_success_returns_result_and_preserves_admission(self):
        task = self.launch()
        await self.until(self.started.is_set)
        self.assertFalse(self.lease_released)
        self.release.set()
        self.assertEqual(await task, "prepared")
        self.assertTrue(self.admitted)
        self.assertTrue(self.lease_released)
        self.assertEqual(self.execution.cancellations, [])

    async def test_repeated_cancellation_does_not_abandon_running_executor(self):
        task = self.launch()
        await self.until(self.started.is_set)
        for count in range(1, 3):
            task.cancel()
            await self.until(lambda: len(self.execution.cancellations) == count)
            self.assertTrue(self.handle.cancelled)
            self.assertFalse(task.done())
            self.assertFalse(self.completed)
            self.assertFalse(self.lease_released)
        self.release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(self.completed)
        self.assertFalse(self.admitted)
        self.assertTrue(self.lease_released)

    async def test_deadline_fences_admission_but_does_not_abandon_worker(self):
        self.execution.deadline = monotonic() + 0.03
        with mock.patch(
            "omlx.owned_preparation_wait.asyncio.wait", wraps=asyncio.wait
        ) as waits:
            task = self.launch()
            await self.until(self.started.is_set)
            await self.until(lambda: self.handle.cancelled)
            await asyncio.sleep(0.02)
            self.assertFalse(task.done())
            self.assertFalse(self.lease_released)
            self.assertLessEqual(waits.call_count, 2, "expired admission must not spin")
            self.release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(self.completed)
        self.assertFalse(self.admitted)

    async def test_preparation_exception_is_preserved_after_cancel(self):
        task = self.launch(fail=True)
        await self.until(self.started.is_set)
        task.cancel()
        await self.until(lambda: self.handle.cancelled)
        self.assertFalse(self.lease_released)
        self.release.set()
        with self.assertRaisesRegex(RuntimeError, "preparation failed after work"):
            await task
        self.assertTrue(self.completed)
        self.assertFalse(self.admitted)

    async def test_cancellation_while_executor_job_is_queued_still_joins_it(self):
        blocker = self.executor.submit(self.release.wait, 3)
        task = self.launch()
        await self.until(lambda: self.executor._work_queue.qsize() == 1)
        task.cancel()
        await self.until(lambda: self.handle.cancelled)
        self.assertFalse(task.done())
        self.assertFalse(self.lease_released)
        self.release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(blocker.done())
        self.assertTrue(self.completed)
        self.assertFalse(self.admitted)
