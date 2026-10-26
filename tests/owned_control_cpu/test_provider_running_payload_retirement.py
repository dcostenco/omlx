# SPDX-License-Identifier: Apache-2.0
"""Request-reference retirement seam only, not running-image admission proof.

The real running-cleanup method executes on its real retained executor. Device,
batch rows and encoder payloads are stubs. Non-text admissions remain refused
at the ordinary cleanup entry; direct tests reach the internal retirement seam
to specify its payload behavior before that guard can ever be relaxed.
"""

import asyncio
import gc
import time
import unittest
import weakref
from types import SimpleNamespace

import test_provider_running_cleanup as fixture
from omlx.owned_admission import RunningCleaned


class Payload:
    pass


def attach(request):
    embeddings, positions = Payload(), Payload()
    request.vlm_inputs_embeds = embeddings
    request.vlm_extra_kwargs = {"position_ids": positions}
    request.vlm_image_hash = "synthetic-image-hash"
    request.vlm_cache_key_start = 7
    request.vlm_cache_key_ranges = [(7, "synthetic-image-hash")]
    return weakref.ref(embeddings), weakref.ref(positions)


class RunningPayloadRetirementTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.base = fixture.RunningCleanupTests()
        await self.base.asyncSetUp()
        self.s, self.core = self.base.s, self.base.core

    async def asyncTearDown(self):
        await self.base.asyncTearDown()

    async def prepared(self, state, rid="synthetic-payload"):
        owned = await self.base.admitted(rid)
        if state == "decoding":
            self.base.to_decoding(owned, uid=3 if rid == "synthetic-payload" else 7)
        else:
            self.base.to_chunked(owned)
        owned._text_only = False
        refs = attach(owned.request)
        return owned, refs

    async def retire(self, owned, state):
        owned._stream_effective_deadline = time.monotonic() + 3
        return await asyncio.get_running_loop().run_in_executor(
            self.core._mlx_executor,
            self.s._cleanup_owned_running,
            owned,
            owned.request,
            owned.binding.request_id,
            state,
        )

    def assert_retained(self, owned, refs):
        self.assertIs(owned.request.vlm_inputs_embeds, refs[0]())
        self.assertIsNotNone(refs[0]())
        self.assertIs(owned.request.vlm_extra_kwargs["position_ids"], refs[1]())
        self.assertEqual(owned.request.vlm_image_hash, "synthetic-image-hash")
        self.assertEqual(owned.request.vlm_cache_key_start, 7)
        self.assertEqual(
            owned.request.vlm_cache_key_ranges, [(7, "synthetic-image-hash")]
        )

    async def success(self, state):
        owned, refs = await self.prepared(state)
        other, other_refs = await self.prepared("decoding", "synthetic-other")
        marker = object()
        self.s.model = SimpleNamespace(
            _pending_embeds=marker,
            _uid_rope_deltas={3: 2, 7: 4},
            _language_model=SimpleNamespace(_position_ids=marker, _rope_deltas=marker),
        )
        fixture.fake_mx.synchronize.side_effect = lambda _stream: self.assert_retained(
            owned, refs
        )
        fact = await self.retire(owned, state)
        self.assertIs(type(fact), RunningCleaned)
        gc.collect()
        self.assertIsNone(owned.request.vlm_inputs_embeds)
        self.assertIsNone(owned.request.vlm_extra_kwargs)
        self.assertIsNone(owned.request.vlm_image_hash)
        self.assertEqual(owned.request.vlm_cache_key_start, 0)
        self.assertIsNone(owned.request.vlm_cache_key_ranges)
        self.assertTrue(all(ref() is None for ref in refs))
        self.assert_retained(other, other_refs)
        self.assertIs(self.s.model._pending_embeds, marker)
        self.assertEqual(self.s.model._uid_rope_deltas, {3: 2, 7: 4})
        self.assertIs(self.s.model._language_model._position_ids, marker)
        self.assertIs(self.s.model._language_model._rope_deltas, marker)
        # An idempotent read must not retire payloads added after its first fact.
        later = attach(owned.request)
        self.assertIs(await self.retire(owned, state), fact)
        self.assert_retained(owned, later)

    async def test_decode_success_releases_only_retired_request_payloads(self):
        await self.success("decoding")

    async def test_chunked_success_releases_only_retired_request_payloads(self):
        await self.success("chunked_prefill")

    async def test_stream_failure_keeps_payloads(self):
        owned, refs = await self.prepared("decoding")
        fixture.fake_mx.synchronize.side_effect = RuntimeError(
            "synthetic device failure"
        )
        self.assertIsNone(await self.retire(owned, "decoding"))
        self.assert_retained(owned, refs)
        self.assertIs(self.s.requests[owned.binding.request_id], owned.request)

    async def test_snapshot_failure_keeps_chunked_payloads(self):
        owned, refs = await self.prepared("chunked_prefill")
        self.s._boundary_snapshot_store.cleanup_request.side_effect = RuntimeError(
            "synthetic snapshot failure"
        )
        self.assertIsNone(await self.retire(owned, "chunked_prefill"))
        self.assert_retained(owned, refs)

    async def test_failed_row_removal_keeps_payloads_and_no_fact(self):
        owned, refs = await self.prepared("decoding")
        self.s.batch_generator.ignore_remove = True
        self.assertIsNone(await self.retire(owned, "decoding"))
        self.assert_retained(owned, refs)

    async def test_late_retirement_failure_keeps_chunked_payloads_and_no_fact(self):
        owned, refs = await self.prepared("chunked_prefill")
        self.s._schedule_deferred_metal_clear.side_effect = RuntimeError(
            "synthetic retirement failure"
        )
        self.assertIsNone(await self.retire(owned, "chunked_prefill"))
        self.assert_retained(owned, refs)

    async def test_changed_stream_after_barrier_keeps_payloads(self):
        owned, refs = await self.prepared("decoding")
        fixture.fake_mx.synchronize.side_effect = lambda _stream: setattr(
            self.s, "_stream", object()
        )
        self.assertIsNone(await self.retire(owned, "decoding"))
        self.assert_retained(owned, refs)

    async def test_normal_entry_still_refuses_nontext_work_without_detaching_payload(
        self,
    ):
        owned, refs = await self.prepared("decoding")
        self.assertIsNone(await self.base.cleanup(owned))
        self.assertEqual(owned._cleanup_outcome, "unsupported")
        self.assertEqual(owned._classified_state, "not_text_only")
        self.assert_retained(owned, refs)
        fixture.fake_mx.synchronize.assert_not_called()
