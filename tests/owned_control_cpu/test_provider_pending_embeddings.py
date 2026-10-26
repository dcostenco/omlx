# SPDX-License-Identifier: Apache-2.0
"""Real owned pool/cleanup methods with synthetic payloads and a stub device."""

from types import MethodType, SimpleNamespace
import unittest

import test_provider_owned_preparation as preparation
import test_provider_running_cleanup as running


class PendingEmbeddingsTests(unittest.IsolatedAsyncioTestCase):
    async def test_unavailable_pending_state_retains_running_request(self):
        class Unobservable:
            @property
            def has_pending_embeddings(self):
                raise AttributeError("synthetic unavailable pending state")

        for mode in ("decoding", "chunked_prefill"):
            with self.subTest(mode=mode):
                fixture = running.RunningCleanupTests()
                await fixture.asyncSetUp()
                try:
                    owned = await fixture.admitted("synthetic-unobservable")
                    if mode == "decoding":
                        fixture.to_decoding(owned, uid=7)
                    else:
                        fixture.to_chunked(owned)
                    fixture.s.model = Unobservable()
                    refs = fixture.refs()
                    assert await fixture.cleanup(owned) is None
                    assert owned._classified_state == "classification_failed"
                    assert owned._cleanup_outcome == "failed"
                    assert fixture.s.requests[owned.binding.request_id] is owned.request
                    assert not fixture.s._owned_waiting_cleanup_proofs
                    assert fixture.refs() == refs
                    running.fake_mx.synchronize.assert_not_called()
                    fixture.s._boundary_snapshot_store.cleanup_request.assert_not_called()
                finally:
                    await fixture.asyncTearDown()

    async def test_running_cleanup_refuses_before_mutation(self):
        for mode in ("decoding", "chunked_prefill"):
            with self.subTest(mode=mode):
                fixture = running.RunningCleanupTests()
                await fixture.asyncSetUp()
                try:
                    owned = await fixture.admitted("synthetic-pending")
                    if mode == "decoding":
                        fixture.to_decoding(owned, uid=7)
                    else:
                        fixture.to_chunked(owned)
                    payload, extra = object(), {"position_ids": object()}
                    model = SimpleNamespace(
                        has_pending_embeddings=True,
                        _pending_embeds=payload,
                        _pending_kwargs=extra,
                        _embed_offset=3,
                    )
                    fixture.s.model = model
                    refs = fixture.refs()
                    assert await fixture.cleanup(owned) is None
                    assert owned._classified_state == "pending_embeddings"
                    assert owned._cleanup_outcome == "unsupported"
                    assert fixture.s.requests[owned.binding.request_id] is owned.request
                    assert not fixture.s._owned_waiting_cleanup_proofs
                    assert fixture.refs() == refs
                    assert model._pending_embeds is payload
                    assert model._pending_kwargs is extra
                    assert model._embed_offset == 3
                    running.fake_mx.synchronize.assert_not_called()
                    fixture.s._boundary_snapshot_store.cleanup_request.assert_not_called()
                finally:
                    await fixture.asyncTearDown()

    async def test_pool_refusal_releases_only_its_acquisition(self):
        pool = preparation.Pool()
        entry = pool.add()
        engine = pool.preload()
        payload = object()
        engine.scheduler.model = SimpleNamespace(
            has_pending_embeddings=True, _pending_embeds=payload
        )
        engine.scheduler._owned_model_refusal = MethodType(
            running.RunningScheduler._owned_model_refusal, engine.scheduler
        )
        with self.assertRaises(preparation.prep.OwnedPreparationRefused) as caught:
            await pool.acquire_owned("m", fence=preparation.Fence(pool))
        assert caught.exception.reason == "model_entangled"
        assert (pool.releases, entry.in_use) == (["m"], 0)
        assert entry.engine is engine
        assert engine.scheduler.model._pending_embeds is payload
        assert pool.unloads == []
