# SPDX-License-Identifier: Apache-2.0
"""Ordinary owned cleanup of one prepared image request; CPU/device stubs only."""

import __future__
import ast
import asyncio
from functools import partial
import gc
from pathlib import Path
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import weakref

import test_provider_running_cleanup as running
from omlx.owned_admission import OwnedAdmission, RunningCleaned
from omlx.owned_vision_preparation import prepare_owned_vision

ADAPTER_METHODS = {
    "clear_pending_embeddings",
    "clear_vlm_position_state",
    "register_rope_delta",
    "unregister_rope_delta",
    "set_step_rope_deltas",
    "has_pending_embeddings",
    "model_type",
    "__call__",
}
tree = ast.parse((running.ROOT / "omlx/models/vlm.py").read_text())
adapter = next(
    n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "VLMModelAdapter"
)
methods = [
    n
    for n in adapter.body
    if isinstance(n, ast.FunctionDef) and n.name in ADAPTER_METHODS
]
assert {n.name for n in methods} == ADAPTER_METHODS
namespace = {}
exec(
    compile(
        ast.fix_missing_locations(
            ast.Module(
                body=[
                    ast.ClassDef(
                        name="Adapter",
                        bases=[],
                        keywords=[],
                        body=methods,
                        decorator_list=[],
                        type_params=[],
                    ),
                ],
                type_ignores=[],
            )
        ),
        str(running.ROOT / "omlx/models/vlm.py"),
        "exec",
        flags=__future__.annotations.compiler_flag,
    ),
    namespace,
)
Adapter = namespace["Adapter"]


class Payload:
    def __bool__(self):
        raise AssertionError("array truth must not be coerced")


class VisionBatch(running.FakeBatchGenerator):
    @property
    def _generation_batch(self):
        return SimpleNamespace(uids=self.stages[2])

    @property
    def _prompt_batch(self):
        return SimpleNamespace(uids=self.stages[1])

    @property
    def _unprocessed_sequences(self):
        return [(uid, None) for uid in self.stages[0]]


class RunningVisionCleanupTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.base = running.RunningCleanupTests()
        await self.base.asyncSetUp()
        self.core, self.s = self.base.core, self.base.s
        self.model = self.s.model = Adapter()
        self.model._vlm_model = SimpleNamespace(
            config=SimpleNamespace(model_type="qwen3_5")
        )
        self.s.batch_generator = VisionBatch()
        self.model._language_model = SimpleNamespace(
            _position_ids=Payload(), _rope_deltas=Payload()
        )
        self.model._batch_rope_deltas = Payload()
        self.model._pending_embeds = None
        self.model._pending_kwargs = {}
        self.model._embed_offset = 0
        self.model._uid_rope_deltas = {}
        self.model._uid_text_positions = set()
        self.model._qwen4_text_prefill_positions = False
        self.model._qwen4_step_text_positions = False

    async def asyncTearDown(self):
        await self.base.asyncTearDown()

    async def prepared(self, state="decoding", *, synchronized=True, **overrides):
        owned = OwnedAdmission(self.core, "synthetic-vision", time.monotonic() + 3)
        embeddings, positions = Payload(), Payload()
        fields = dict(
            vlm_inputs_embeds=embeddings,
            vlm_extra_kwargs={"position_ids": positions},
            vlm_image_hash="synthetic-hash",
            vlm_cache_key_start=4,
            vlm_cache_key_ranges=[(4, "synthetic-hash")],
        )
        fields.update(overrides)
        if synchronized:
            await asyncio.get_running_loop().run_in_executor(
                self.core._mlx_executor,
                partial(
                    prepare_owned_vision,
                    owned,
                    self.core,
                    lambda: None,
                    dispatch_executor=self.core._mlx_executor,
                    stream_context=running.fake_mx.stream,
                    synchronize=running.fake_mx.synchronize,
                    language_model=self.model._language_model,
                ),
            )
        await self.core.add_request(
            "synthetic",
            request_id=owned.binding.request_id,
            _owned_admission=owned,
            **fields,
        )
        owned.request.rope_deltas = -4.0
        if state == "decoding":
            self.base.to_decoding(owned, uid=3)
            self.model.register_rope_delta(3, -4.0)
        else:
            self.base.to_chunked(owned)
            owned.request._prefill_saved_rope_deltas = (
                self.model._language_model,
                Payload(),
            )
        running.fake_mx.synchronize = Mock()
        return owned, (weakref.ref(embeddings), weakref.ref(positions))

    def assert_retained(self, owned, refs):
        self.assertIs(owned.request.vlm_inputs_embeds, refs[0]())
        self.assertIsNotNone(refs[0]())
        self.assertIs(owned.request.vlm_extra_kwargs["position_ids"], refs[1]())
        self.assertIsNone(owned._cleanup_fact)

    async def success(self, state):
        owned, refs = await self.prepared(state)
        self.assertFalse(owned._text_only)
        checked = []
        original_clear = self.model.clear_vlm_position_state

        def clear():
            self.assertNotIn(owned.binding.request_id, self.s.requests)
            self.assertNotIn(owned.binding.request_id, self.s.running)
            self.assertNotIn(owned.binding.request_id, self.s._prefill_states)
            self.assertTrue(all(ref() is not None for ref in refs))
            checked.append("detached-before-clear")
            original_clear()

        self.model.clear_vlm_position_state = clear
        fact = await self.base.cleanup(owned)
        self.assertIs(type(fact), RunningCleaned)
        self.assertTrue(getattr(owned, "_prepared_vision", False))
        self.base.assert_retired(owned, uid=3 if state == "decoding" else None)
        self.assertEqual(checked, ["detached-before-clear"])
        self.assertGreaterEqual(running.fake_mx.synchronize.call_count, 2)
        self.assertEqual(self.model._uid_rope_deltas, {})
        self.assertIsNone(self.model._batch_rope_deltas)
        self.assertIsNone(self.model._language_model._position_ids)
        self.assertIsNone(self.model._language_model._rope_deltas)
        self.assertIsNone(self.model._pending_embeds)
        self.assertEqual(
            (self.model._pending_kwargs, self.model._embed_offset), ({}, 0)
        )
        self.assertIsNone(getattr(owned.request, "_prefill_saved_rope_deltas", None))
        self.assertIsNone(owned.request.vlm_inputs_embeds)
        self.assertIsNone(owned.request.vlm_extra_kwargs)
        self.assertIsNone(owned.request.vlm_image_hash)
        self.assertIsNone(owned.request.vlm_cache_key_ranges)
        self.assertEqual(owned.request.vlm_cache_key_start, 0)
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))

    async def test_prepared_image_decoding_uses_ordinary_cleanup(self):
        await self.success("decoding")

    async def test_prepared_image_chunked_prefill_uses_ordinary_cleanup(self):
        await self.success("chunked_prefill")

    async def test_missing_preparation_is_not_a_vision_capability(self):
        owned, refs = await self.prepared(synchronized=False)
        self.assertFalse(getattr(owned, "_prepared_vision", False))
        self.assertIsNone(await self.base.cleanup(owned))
        self.assert_retained(owned, refs)
        running.fake_mx.synchronize.assert_not_called()

    async def raw_media_refused(self, field):
        owned, refs = await self.prepared(**{field: ["synthetic-raw"]})
        self.assertFalse(getattr(owned, "_prepared_vision", False))
        self.assertIsNone(await self.base.cleanup(owned))
        self.assert_retained(owned, refs)

    async def test_raw_images_still_refused(self):
        await self.raw_media_refused("images")

    async def test_raw_videos_still_refused(self):
        await self.raw_media_refused("videos")

    async def test_foreign_uid_state_refuses_before_barrier(self):
        owned, refs = await self.prepared()
        self.model.register_rope_delta(7, 2.0)
        before = dict(self.model._uid_rope_deltas)
        self.assertIsNone(await self.base.cleanup(owned))
        self.assert_retained(owned, refs)
        self.assertEqual(self.model._uid_rope_deltas, before)
        running.fake_mx.synchronize.assert_not_called()

    async def test_other_request_refuses_without_mutation(self):
        owned, refs = await self.prepared()
        other = await self.base.admitted("synthetic-other")
        self.base.to_chunked(other)
        before = dict(self.s.requests)
        self.assertIsNone(await self.base.cleanup(owned))
        self.assertEqual(self.s.requests, before)
        self.assert_retained(owned, refs)
        running.fake_mx.synchronize.assert_not_called()

    async def test_pending_state_refuses_without_detaching(self):
        owned, refs = await self.prepared()
        self.model._pending_embeds = Payload()
        self.assertIsNone(await self.base.cleanup(owned))
        self.assert_retained(owned, refs)
        self.assertIn(owned.binding.request_id, self.s.requests)
        running.fake_mx.synchronize.assert_not_called()

    async def test_pending_state_appearing_during_barrier_refuses(self):
        owned, refs = await self.prepared()
        running.fake_mx.synchronize.side_effect = lambda _: setattr(
            self.model, "_pending_embeds", Payload()
        )
        self.assertIsNone(await self.base.cleanup(owned))
        self.assert_retained(owned, refs)
        self.assertIn(owned.binding.request_id, self.s.requests)

    async def test_failed_initial_sync_keeps_scheduler_and_payloads(self):
        owned, refs = await self.prepared()
        running.fake_mx.synchronize.side_effect = RuntimeError("synthetic sync error")
        self.assertIsNone(await self.base.cleanup(owned))
        self.assert_retained(owned, refs)
        self.assertIn(owned.binding.request_id, self.s.requests)

    async def test_partial_global_clear_failure_cannot_leave_target_live(self):
        owned, refs = await self.prepared("chunked_prefill")
        self.model.clear_pending_embeddings = Mock(
            side_effect=RuntimeError("synthetic clear")
        )
        self.assertIsNone(await self.base.cleanup(owned))
        self.assert_retained(owned, refs)
        self.assertNotIn(owned.binding.request_id, self.s.requests)
        self.assertNotIn(owned.binding.request_id, self.s._prefill_states)
        self.assertNotIn(owned.request, self.s.prefilling)
        self.assertIsNotNone(owned.request._prefill_saved_rope_deltas)

    async def test_failed_second_sync_keeps_media_without_a_live_target(self):
        owned, refs = await self.prepared()
        running.fake_mx.synchronize.side_effect = [
            None,
            RuntimeError("synthetic second sync"),
        ]
        self.assertIsNone(await self.base.cleanup(owned))
        self.assert_retained(owned, refs)
        self.assertNotIn(owned.binding.request_id, self.s.requests)
        self.assertIsNotNone(self.model._language_model._position_ids)

    async def test_orphan_generator_row_refuses_without_snapshot_mutation(self):
        owned, refs = await self.prepared()
        self.s.batch_generator.stages[0].append(7)
        self.assertIsNone(await self.base.cleanup(owned))
        self.assert_retained(owned, refs)
        self.s._boundary_snapshot_store.cleanup_request.assert_not_called()
        running.fake_mx.synchronize.assert_not_called()

    async def test_duplicate_generator_row_refuses(self):
        owned, refs = await self.prepared()
        self.s.batch_generator.stages[1].append(3)
        self.assertIsNone(await self.base.cleanup(owned))
        self.assert_retained(owned, refs)
        running.fake_mx.synchronize.assert_not_called()

    async def test_foreign_text_uid_proof_refuses(self):
        owned, refs = await self.prepared()
        self.model._uid_text_positions.add(7)
        self.assertIsNone(await self.base.cleanup(owned))
        self.assert_retained(owned, refs)
        running.fake_mx.synchronize.assert_not_called()

    async def test_unknown_architecture_remains_unqualified(self):
        owned, refs = await self.prepared()
        self.model._vlm_model.config.model_type = "synthetic-unqualified"
        self.assertIsNone(await self.base.cleanup(owned))
        self.assert_retained(owned, refs)
        running.fake_mx.synchronize.assert_not_called()

    async def test_missing_cleanup_method_refuses_before_barrier(self):
        owned, refs = await self.prepared()
        self.model.clear_pending_embeddings = None
        self.assertIsNone(await self.base.cleanup(owned))
        self.assert_retained(owned, refs)
        running.fake_mx.synchronize.assert_not_called()

    async def test_pending_kwargs_without_embeddings_refuses(self):
        owned, refs = await self.prepared()
        self.model._pending_kwargs = {"position_ids": Payload()}
        self.assertIsNone(await self.base.cleanup(owned))
        self.assert_retained(owned, refs)
        running.fake_mx.synchronize.assert_not_called()

    async def test_foreign_saved_snapshot_owner_refuses(self):
        owned, refs = await self.prepared("chunked_prefill")
        owned.request._prefill_saved_rope_deltas = (object(), Payload())
        self.assertIsNone(await self.base.cleanup(owned))
        self.assert_retained(owned, refs)
        running.fake_mx.synchronize.assert_not_called()

    async def test_foreign_row_during_barrier_keeps_live_snapshot(self):
        owned, refs = await self.prepared()
        running.fake_mx.synchronize.side_effect = (
            lambda _: self.s.batch_generator.stages[0].append(7)
        )
        self.assertIsNone(await self.base.cleanup(owned))
        self.assert_retained(owned, refs)
        self.assertIn(owned.binding.request_id, self.s.requests)
        self.s._boundary_snapshot_store.cleanup_request.assert_not_called()

    async def test_foreign_state_during_retirement_barrier_prevents_clear(self):
        owned, refs = await self.prepared()

        def sync(_):
            if running.fake_mx.synchronize.call_count == 2:
                self.model.register_rope_delta(7, 1)

        running.fake_mx.synchronize.side_effect = sync
        self.assertIsNone(await self.base.cleanup(owned))
        self.assert_retained(owned, refs)
        self.assertNotIn(owned.binding.request_id, self.s.requests)
        self.assertIsNotNone(self.model._language_model._position_ids)
        self.s._boundary_snapshot_store.cleanup_request.assert_not_called()

    async def test_noop_adapter_clear_is_not_a_cleanup_fact(self):
        owned, refs = await self.prepared("chunked_prefill")
        self.model.clear_vlm_position_state = Mock()
        self.assertIsNone(await self.base.cleanup(owned))
        self.assert_retained(owned, refs)
        self.assertNotIn(owned.binding.request_id, self.s.requests)
        self.assertIsNotNone(owned.request._prefill_saved_rope_deltas)

    async def test_snapshot_store_failure_occurs_after_row_retirement(self):
        owned, refs = await self.prepared()

        def fail(rid, strict):
            self.assertNotIn(rid, self.s.requests)
            raise RuntimeError("synthetic snapshot failure")

        self.s._boundary_snapshot_store.cleanup_request.side_effect = fail
        self.assertIsNone(await self.base.cleanup(owned))
        self.assert_retained(owned, refs)
        self.assertIsNotNone(self.model._language_model._position_ids)

    async def test_prepared_not_admitted_request_keeps_explicit_positions(self):
        owned, _ = await self.prepared()
        positions, delta, embeds = Payload(), Payload(), Payload()
        other_kwargs = {"position_ids": positions, "_captured_rope_deltas": delta}
        # B owns its preparation results but is not visible to scheduler maps.
        # Cleanup A must not edit them; the real adapter forwards B's kwargs.
        fact = await self.base.cleanup(owned)
        self.assertIs(type(fact), RunningCleaned)
        self.model._language_model = Mock(
            return_value=SimpleNamespace(logits="synthetic")
        )
        result = self.model(
            Payload(), inputs_embeds=embeds, vlm_extra_kwargs=dict(other_kwargs)
        )
        self.assertEqual(result, "synthetic")
        forwarded = self.model._language_model.call_args.kwargs
        self.assertIs(forwarded["position_ids"], positions)
        self.assertIs(forwarded["inputs_embeds"], embeds)
        self.assertIs(other_kwargs["_captured_rope_deltas"], delta)


if __name__ == "__main__":
    unittest.main()
