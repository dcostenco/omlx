# SPDX-License-Identifier: Apache-2.0
"""Running cleanup must fence again after a retained synchronization call."""

import time
import unittest

import test_provider_running_cleanup as running
from omlx.owned_admission import RunningCleaned


class RunningCleanupFenceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = running.RunningCleanupTests()
        await self.fixture.asyncSetUp()

    async def asyncTearDown(self):
        await self.fixture.asyncTearDown()

    async def scenario(self, state, change):
        fixture = self.fixture
        core, scheduler = fixture.core, fixture.s
        handle = await fixture.admitted("synthetic-fenced-" + state + "-" + change)
        request, rid = handle.request, handle.binding.request_id
        if state == "decoding":
            fixture.to_decoding(handle, uid=3)
        else:
            fixture.to_chunked(handle)
        before_refs = fixture.refs()
        before_request = dict(scheduler.requests)
        original_generation = core._prism_control_generation_id
        original_stream = scheduler._stream
        observed = []

        def mutate():
            if change == "deadline":
                # This public internal pin only tightens an existing deadline.
                # It models another cleanup caller narrowing it during a wait.
                handle.pin_stream_deadline(core, time.monotonic() - 1)
            elif change == "identity":
                core._prism_control_generation_id = "synthetic-drifted"
            else:
                scheduler._stream = object()

        def sync(stream):
            observed.append(stream)
            self.assertIs(stream, handle.binding.stream)
            if change != "before_stream":
                mutate()

        running.fake_mx.synchronize.side_effect = sync
        if change == "before_stream":
            scheduler._boundary_snapshot_store.cleanup_request.side_effect = (
                lambda *a, **k: mutate()
            )
        try:
            fact = await fixture.cleanup(handle)
            # Timeout/refusal is not worker completion. Inspect after the actual
            # retained job has settled, including its post-call core checks.
            if handle._cleanup_future is not None:
                try:
                    handle._cleanup_future.result(timeout=3)
                except ValueError:
                    pass  # the old core wrapper can observe drift too late
            self.assertIsNone(fact)
            self.assertEqual(scheduler.requests, before_request)
            self.assertEqual(fixture.refs(), before_refs)
            self.assertIsNone(handle._cleanup_fact)
            self.assertEqual(handle._cleanup_outcome, "failed")
            if state == "decoding":
                self.assertIs(scheduler.running[rid], request)
                self.assertEqual(scheduler.request_id_to_uid[rid], 3)
                self.assertIn(3, scheduler.batch_generator._find_uids([3]))
            else:
                self.assertIn(rid, scheduler._prefill_states)
                self.assertIsNotNone(scheduler._prefill_states[rid].cache)
                self.assertIn(request, scheduler.prefilling)
            self.assertEqual(len(observed), 0 if change == "before_stream" else 1)
        finally:
            core._prism_control_generation_id = original_generation
            scheduler._stream = original_stream

    async def test_decoding_deadline_tightened_during_sync_keeps_request(self):
        await self.scenario("decoding", "deadline")

    async def test_chunked_deadline_tightened_during_sync_keeps_request(self):
        await self.scenario("chunked", "deadline")

    async def test_decoding_identity_drift_during_sync_keeps_request(self):
        await self.scenario("decoding", "identity")

    async def test_chunked_identity_drift_during_sync_keeps_request(self):
        await self.scenario("chunked", "identity")

    async def test_decoding_stream_drift_during_sync_keeps_request(self):
        await self.scenario("decoding", "stream")

    async def test_chunked_stream_drift_during_sync_keeps_request(self):
        await self.scenario("chunked", "stream")

    async def test_stream_drift_before_sync_is_refused_without_device_call(self):
        await self.scenario("decoding", "before_stream")

    async def test_unchanged_binding_still_retires_normally(self):
        handle = await self.fixture.admitted("synthetic-normal-fence")
        self.fixture.to_decoding(handle, uid=3)
        fact = await self.fixture.cleanup(handle)
        self.assertIs(type(fact), RunningCleaned)
        self.fixture.assert_retired(handle, uid=3)

    async def test_deadline_narrowed_but_still_future_permits_cleanup(self):
        handle = await self.fixture.admitted("synthetic-future-deadline")
        self.fixture.to_decoding(handle, uid=3)
        narrowed = []

        def sync(stream):
            self.assertIs(stream, handle.binding.stream)
            before = handle._stream_effective_deadline
            after = handle.pin_stream_deadline(
                handle.binding.core, time.monotonic() + 1.5
            )
            self.assertLess(after, before)
            self.assertGreater(after, time.monotonic())
            narrowed.append(after)

        running.fake_mx.synchronize.side_effect = sync
        fact = await self.fixture.cleanup(handle)
        self.assertIs(type(fact), RunningCleaned)
        self.assertEqual(len(narrowed), 1)
        self.fixture.assert_retired(handle, uid=3)
