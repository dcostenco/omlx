# SPDX-License-Identifier: Apache-2.0
"""Prepared waiting vision only: real lifecycle methods, stub device/embeddings."""

import asyncio
from contextlib import nullcontext
from functools import partial
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import test_provider_waiting_cleanup as waiting
from omlx.owned_admission import OwnedAdmission, WaitingCleaned, StreamSynchronized
from omlx.owned_vision_preparation import prepare_owned_vision
from omlx.request import RequestStatus


class Tensor:
    def __bool__(self):
        raise AssertionError("array truth must not be evaluated")


class VisionWaitingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = waiting.WaitingCleanupTests()
        await self.fixture.asyncSetUp()
        self.core = self.fixture.core
        self.scheduler = self.fixture.scheduler
        self.sync = mock.Mock()
        self.patch = mock.patch.dict(
            waiting.namespace, mx=SimpleNamespace(synchronize=self.sync)
        )
        self.patch.start()

    async def asyncTearDown(self):
        self.patch.stop()
        await self.fixture.asyncTearDown()

    async def admitted(self, rid="synthetic-vision", prepared=True, pending_sync=False):
        handle = OwnedAdmission(self.core, rid, time.monotonic() + 4)
        if prepared or pending_sync:
            # Actual worker boundary; device work is explicitly a stub.
            synchronization = mock.Mock(
                side_effect=(
                    RuntimeError("synthetic pending preparation")
                    if pending_sync
                    else None
                )
            )
            with self.assertRaises(RuntimeError) if pending_sync else nullcontext():
                await asyncio.get_running_loop().run_in_executor(
                    self.core._mlx_executor,
                    partial(
                        prepare_owned_vision,
                        handle,
                        self.core,
                        lambda: None,
                        dispatch_executor=self.core._mlx_executor,
                        stream_context=lambda stream: nullcontext(),
                        synchronize=synchronization,
                    ),
                )
        await self.core.add_request(
            "synthetic",
            request_id=rid,
            _owned_admission=handle,
            vlm_inputs_embeds=Tensor(),
            vlm_extra_kwargs={"position_ids": Tensor()},
            vlm_image_hash="synthetic-image-hash",
            vlm_cache_key_start=2,
            vlm_cache_key_ranges=[(2, "synthetic-image-hash")],
            skip_cache_store=True,
        )
        return handle

    def assert_payload(self, request):
        self.assertIsInstance(request.vlm_inputs_embeds, Tensor)
        self.assertIsInstance(request.vlm_extra_kwargs["position_ids"], Tensor)
        self.assertEqual(request.vlm_image_hash, "synthetic-image-hash")

    async def test_success_syncs_before_retirement_and_preserves_other_requests(self):
        handle = await self.admitted()
        other = await self.admitted("synthetic-other")
        rid, request = handle.binding.request_id, handle.request
        observed = []

        def sync(stream):
            self.assertIs(stream, handle.binding.stream)
            self.assertIs(self.scheduler.requests[rid], request)
            self.assertIn(request, self.scheduler.waiting)
            self.assert_payload(request)
            observed.append(threading.get_ident())

        self.sync.side_effect = sync
        fact = await self.fixture.cleanup(handle)
        self.assertIs(type(fact), WaitingCleaned)
        self.assertNotEqual(observed, [])
        self.assertTrue(all(t != threading.get_ident() for t in observed))
        self.assertEqual(self.sync.call_count, 1)
        self.assertNotIn(rid, self.scheduler.requests)
        self.assertNotIn(request, self.scheduler.waiting)
        self.assertEqual(request.status, RequestStatus.FINISHED_ABORTED)
        for name in (
            "vlm_inputs_embeds",
            "vlm_extra_kwargs",
            "vlm_image_hash",
            "vlm_cache_key_ranges",
        ):
            self.assertIsNone(getattr(request, name), name)
        self.assertEqual(request.vlm_cache_key_start, 0)
        self.assertIs(self.scheduler.requests[other.binding.request_id], other.request)
        self.assert_payload(other.request)
        self.assertFalse(self.scheduler._owned_waiting_cleanup_proofs)
        self.assertIs(handle._cleanup_fact, fact)
        self.assertIs(
            type(await self.core._owned_stream_barrier(handle, time.monotonic() + 3)),
            StreamSynchronized,
        )

    async def test_sync_failure_keeps_payload_and_queue_without_cleanup_fact(self):
        handle = await self.admitted()
        self.sync.side_effect = RuntimeError("synthetic synchronization failure")
        self.assertIsNone(await self.fixture.cleanup(handle))
        self.assertEqual(handle._cleanup_outcome, "failed")
        self.sync.assert_called_once_with(handle.binding.stream)
        self.assertIs(
            self.scheduler.requests[handle.binding.request_id], handle.request
        )
        self.assert_payload(handle.request)
        self.assertIsNone(handle._cleanup_fact)
        self.assertIsNone(
            await self.core._owned_stream_barrier(handle, time.monotonic() + 3)
        )

    async def test_expiry_or_identity_drift_during_sync_keeps_payload(self):
        for kind in ("deadline", "identity", "scheduler_stream"):
            with self.subTest(kind=kind):
                handle = await self.admitted("synthetic-" + kind)
                original_generation = self.core._prism_control_generation_id
                original_stream = self.scheduler._stream

                def sync(stream):
                    if kind == "deadline":
                        time.sleep(0.05)
                    elif kind == "identity":
                        self.core._prism_control_generation_id = "synthetic-changed"
                    else:
                        self.scheduler._stream = object()

                self.sync.side_effect = sync
                self.sync.reset_mock()
                try:
                    deadline = time.monotonic() + (0.02 if kind == "deadline" else 3)
                    self.assertIsNone(
                        await self.core._owned_cleanup_waiting(handle, deadline)
                    )
                    # Caller timeout is not worker completion. Inspect state
                    # only after the retained executor job actually settles.
                    await asyncio.wrap_future(handle._cleanup_future)
                    self.sync.assert_called_once_with(handle.binding.stream)
                    self.assertIn(handle.binding.request_id, self.scheduler.requests)
                    self.assertIs(
                        self.scheduler.requests[handle.binding.request_id],
                        handle.request,
                    )
                    self.assert_payload(handle.request)
                    self.assertIsNone(handle._cleanup_fact)
                finally:
                    self.core._prism_control_generation_id = original_generation
                    self.scheduler._stream = original_stream

    async def test_unqualified_vision_is_refused_without_sync_or_payload_changes(self):
        for kind in (
            "unprepared",
            "pending_preparation",
            "started",
            "non_pristine",
            "rope",
            "images",
            "videos",
            "missing_embeds",
        ):
            with self.subTest(kind=kind):
                handle = await self.admitted(
                    "synthetic-refuse-" + kind,
                    prepared=kind not in ("unprepared", "pending_preparation"),
                    pending_sync=kind == "pending_preparation",
                )
                request = handle.request
                if kind == "started":
                    handle._started = True
                if kind == "non_pristine":
                    handle._waiting_pristine = False
                if kind == "rope":
                    request.rope_deltas = 1.0
                if kind == "images":
                    request.images = [Tensor()]
                if kind == "videos":
                    request.videos = [Tensor()]
                if kind == "missing_embeds":
                    request.vlm_inputs_embeds = None
                self.sync.reset_mock()
                self.assertIsNone(await self.fixture.cleanup(handle))
                self.sync.assert_not_called()
                self.assertIs(
                    self.scheduler.requests[handle.binding.request_id], request
                )
                self.assertIsNotNone(request.vlm_extra_kwargs)

    async def test_original_text_admission_cannot_be_changed_into_prepared_vision(self):
        handle = OwnedAdmission(
            self.core, "synthetic-altered-text", time.monotonic() + 3
        )
        await asyncio.get_running_loop().run_in_executor(
            self.core._mlx_executor,
            partial(
                prepare_owned_vision,
                handle,
                self.core,
                lambda: None,
                dispatch_executor=self.core._mlx_executor,
                stream_context=lambda stream: nullcontext(),
                synchronize=mock.Mock(),
            ),
        )
        await self.core.add_request(
            "text", request_id=handle.binding.request_id, _owned_admission=handle
        )
        self.assertTrue(handle._text_only)
        handle.request.vlm_inputs_embeds = Tensor()
        self.assertIsNone(await self.fixture.cleanup(handle))
        self.sync.assert_not_called()
        self.assertIsInstance(handle.request.vlm_inputs_embeds, Tensor)

    async def test_unqualified_running_vision_refusal_remains(self):
        handle = await self.admitted()
        self.assertTrue(handle._prepared_vision)
        self.scheduler.model.model_type = "synthetic-unqualified"
        self.assertEqual(
            self.scheduler._owned_running_refusal(
                handle, handle.request, handle.binding.request_id
            ),
            "vision_architecture_unqualified",
        )
