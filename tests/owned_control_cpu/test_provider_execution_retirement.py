# SPDX-License-Identifier: Apache-2.0
"""Completed coordinator references, not GPU erasure or whole-process retention.

Runs the real coordinator/task/queue with a stub store and engine stage. Weak
references distinguish releasing payloads from merely forgetting an identity.
Unresolved and pending work must keep its existing ownership.
"""

import asyncio
import gc
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import weakref

import test_provider_control_routes as support  # CPU import boundary
from omlx.api import prism_control_coordinator as coord
from omlx.api.prism_control_store import ControlConflict


class Payload:
    pass


class ExecutionRetirementTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.store = SimpleNamespace(
            claim=Mock(return_value=SimpleNamespace(request_id="synthetic")),
            verify_claim=Mock(),
            finish=Mock(return_value=SimpleNamespace(state="completed")),
        )
        self.gate = asyncio.Event()
        self.entered = asyncio.Event()
        self.outcome = "completed"
        self.raise_unresolved = False
        self.keep_lease = False
        self.cancel_after_terminal = False
        self.early_terminal = False
        self.refs = []
        self.coordinator = coord.ControlCoordinator(
            self.store, lambda model: "digest", self.stage, lock=object()
        )

    async def stage(self, execution):
        # Synthetic image/array sentinels: no model, device or image admission.
        payload = Payload()
        handle = SimpleNamespace(payload=payload)
        self.refs.append(weakref.ref(payload))
        execution.preparation_handle = handle
        execution.publish_fence(lambda: handle)
        execution.final_output = payload
        execution.cleanup_task = asyncio.create_task(asyncio.sleep(0, result=payload))
        await execution.cleanup_task
        execution.lease = object()
        if self.early_terminal:
            execution.lease = None
            execution.emit("terminal", SimpleNamespace(state="synthetic-early"))
        self.entered.set()
        await self.gate.wait()
        if not self.keep_lease:
            execution.lease = None
        if self.raise_unresolved:
            raise RuntimeError("synthetic unresolved engine stage")
        execution.emit("delta", {"text": "retained output"})
        if self.cancel_after_terminal:
            execution.cancellations.append(None)
        return self.outcome, ("synthetic-proof",)

    def start(self):
        request = Payload()
        self.refs.append(weakref.ref(request))
        return self.coordinator.start(
            ("owner", "job", "operation"),
            request_sha256="request-digest",
            model_id="m",
            settings_sha256="digest",
            deadline=time.monotonic() + 3,
            request=request,
        )

    async def settle(self, execution):
        self.gate.set()
        try:
            await execution.task
        except asyncio.CancelledError:
            self.assertTrue(self.cancel_after_terminal)
        # The done callback must finish even when a slow SSE reader retains
        # execution and its task for longer than the coordinator does.
        await asyncio.sleep(0)

    def assert_retired(self, execution):
        for name in (
            "request",
            "preparation_handle",
            "_fence",
            "cleanup_task",
            "final_output",
        ):
            self.assertIsNone(getattr(execution, name, None), name)
        gc.collect()
        self.assertTrue(all(ref() is None for ref in self.refs))

    def assert_owned(self, execution):
        self.assertIsNotNone(execution.request)
        self.assertIsNotNone(execution.preparation_handle)
        self.assertIsNotNone(execution._fence)
        self.assertIsNotNone(execution.cleanup_task)
        gc.collect()
        self.assertTrue(all(ref() is not None for ref in self.refs))

    async def test_terminal_receipts_retire_payloads_without_consuming_queued_output(
        self,
    ):
        for outcome in ("completed", "failed", "cancelled", "not_started"):
            with self.subTest(outcome=outcome):
                self.outcome = outcome
                self.refs = []
                self.store.finish.return_value = SimpleNamespace(state=outcome)
                execution = self.start()
                await self.settle(execution)
                self.assert_retired(execution)
                self.assertNotIn(execution.identity, self.coordinator._executions)
                events = []
                while not execution._events.empty():
                    events.append(await execution.next_event())
                self.assertEqual(
                    [e for e, _ in events], ["claimed", "delta", "terminal"]
                )
                self.assertEqual(events[1][1], {"text": "retained output"})
                self.assertIs(events[-1][1], self.store.finish.return_value)

    async def test_pending_stage_keeps_payload_and_lease_until_settlement(self):
        execution = self.start()
        try:
            await asyncio.wait_for(self.entered.wait(), 3)
            self.assert_owned(execution)
            self.assertIsNotNone(execution.lease)
            self.coordinator.fence(execution.identity)
            self.assertTrue(execution.cancel_requested)
        finally:
            await self.settle(execution)
        self.assert_retired(execution)

    async def test_terminal_store_pending_keeps_payload_until_receipt(self):
        entered, release = threading.Event(), threading.Event()

        def finish(*args, **kwargs):
            entered.set()
            if not release.wait(3):
                raise RuntimeError("test store wait expired")
            return SimpleNamespace(state="completed")

        self.store.finish.side_effect = finish
        execution = self.start()
        self.gate.set()
        try:
            async with asyncio.timeout(3):
                while not entered.is_set():
                    await asyncio.sleep(0.001)
            self.assert_owned(execution)
        finally:
            release.set()
            await self.settle(execution)
        self.assert_retired(execution)

    async def test_store_conflict_preserves_unresolved_payloads(self):
        self.store.finish.side_effect = ControlConflict("synthetic")
        execution = self.start()
        await self.settle(execution)
        self.assert_owned(execution)
        self.assertEqual((await execution.next_event())[0], "claimed")
        self.assertEqual((await execution.next_event())[0], "delta")
        self.assertEqual((await execution.next_event())[0], "unresolved")

    async def test_engine_unresolved_with_lease_keeps_payloads(self):
        self.raise_unresolved = self.keep_lease = True
        execution = self.start()
        await self.settle(execution)
        self.assert_owned(execution)
        self.assertIsNotNone(execution.lease)
        self.store.finish.assert_not_called()

    async def test_terminal_with_unexpected_held_lease_does_not_retire(self):
        self.keep_lease = True
        execution = self.start()
        await self.settle(execution)
        self.assert_owned(execution)

    async def test_engine_unresolved_without_lease_keeps_payloads(self):
        self.raise_unresolved = True
        execution = self.start()
        await self.settle(execution)
        self.assertIsNone(execution.lease)
        self.assert_owned(execution)
        events = []
        while not execution._events.empty():
            events.append((await execution.next_event())[0])
        self.assertEqual(events, ["claimed", "unresolved"])
        self.store.finish.assert_not_called()

    async def test_replayed_cancellation_after_receipt_still_retires(self):
        self.cancel_after_terminal = True
        execution = self.start()
        await self.settle(execution)
        self.assertTrue(execution.task.cancelled())
        self.assert_retired(execution)

    async def test_pre_execution_refusal_releases_request_without_claiming_stop(self):
        self.store.claim.side_effect = ControlConflict("synthetic")
        execution = self.start()
        await self.settle(execution)
        self.assert_retired(execution)
        self.assertEqual(await execution.next_event(), ("refused", "conflict"))
        self.store.finish.assert_not_called()

    async def test_final_event_alone_does_not_retire_a_running_stage(self):
        self.early_terminal = True
        execution = self.start()
        try:
            await asyncio.wait_for(self.entered.wait(), 3)
            self.assertTrue(execution.final)
            self.assertIsNone(execution.lease)
            self.assertFalse(execution.task.done())
            self.assert_owned(execution)
        finally:
            await self.settle(execution)
        self.assert_retired(execution)

    async def test_task_cancelled_before_claim_gets_fallback_refusal_and_retires(self):
        execution = self.start()
        execution.task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await execution.task
        await asyncio.sleep(0)
        self.assert_retired(execution)
        self.assertEqual(await execution.next_event(), ("refused", "unavailable"))
        self.store.claim.assert_not_called()
