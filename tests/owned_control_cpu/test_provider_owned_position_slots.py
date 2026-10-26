# SPDX-License-Identifier: Apache-2.0
"""Owned preparation slot bindings: real executor, stub encoder and device."""

from types import SimpleNamespace
import time
import unittest

import test_provider_owned_vision_preparation as vision
from omlx.owned_admission import OwnedAdmission
from omlx import owned_preparation as prep


class OwnedPositionSlotTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = vision.VisionPreparationTests()
        await self.fixture.asyncSetUp()
        self.old_position, self.old_delta = object(), object()
        self.new_position, self.new_delta = object(), object()
        self.owner = SimpleNamespace(
            _position_ids=self.old_position, _rope_deltas=self.old_delta
        )
        self.fixture.engine._vlm_model = SimpleNamespace(language_model=self.owner)
        self.original_process = self.fixture.process
        self.process_error = None
        self.observed = []
        self.fixture.engine._process_chat_messages = self.process

    async def asyncTearDown(self):
        await self.fixture.asyncTearDown()

    def process(self, *args):
        self.owner._position_ids = self.new_position
        self.owner._rope_deltas = self.new_delta
        self.observed.append("encode")
        if self.process_error is not None:
            raise self.process_error
        return (
            "synthetic-prompt",
            object(),
            {
                "position_ids": self.new_position,
                "_captured_rope_deltas": self.new_delta,
            },
            "synthetic-image-hash",
            0,
            [],
        )

    def handle(self):
        return OwnedAdmission(
            self.fixture.core, "synthetic-position-owner", time.monotonic() + 3
        )

    async def prepare(self, handle):
        return await self.fixture.engine._stream_chat_prompt(
            [], None, {}, admission=handle
        )

    def assert_restored(self):
        self.assertIs(self.owner._position_ids, self.old_position)
        self.assertIs(self.owner._rope_deltas, self.old_delta)

    async def test_success_restores_previous_bindings_and_preserves_request_result(
        self,
    ):
        handle = self.handle()

        def sync(stream):
            self.assertIs(stream, handle.binding.stream)
            self.assertIs(self.owner._position_ids, self.new_position)
            self.assertIs(self.owner._rope_deltas, self.new_delta)
            self.observed.append("sync")

        self.fixture.synchronize = sync
        self.fixture.namespace["mx"].synchronize = sync
        _, result = await self.prepare(handle)
        self.assert_restored()
        self.assertIs(result["vlm_extra_kwargs"]["position_ids"], self.new_position)
        self.assertIs(
            result["vlm_extra_kwargs"]["_captured_rope_deltas"], self.new_delta
        )
        self.assertEqual(self.observed, ["encode", "sync"])
        self.assertFalse(handle.preparation_unresolved)

    async def test_encoder_exception_restores_bindings_and_keeps_original_exception(
        self,
    ):
        handle = self.handle()
        self.process_error = ValueError("synthetic encode error with position state")
        with self.assertRaises(ValueError) as caught:
            await self.prepare(handle)
        self.assertIs(caught.exception, self.process_error)
        self.assert_restored()
        self.assertFalse(handle.preparation_unresolved)

    async def test_absent_slots_are_absent_again_but_unrelated_state_is_preserved(self):
        self.owner = SimpleNamespace(unrelated=object())
        self.fixture.engine._vlm_model.language_model = self.owner
        unrelated = self.owner.unrelated
        await self.prepare(self.handle())
        self.assertFalse(hasattr(self.owner, "_position_ids"))
        self.assertFalse(hasattr(self.owner, "_rope_deltas"))
        self.assertIs(self.owner.unrelated, unrelated)

    async def test_existing_none_is_preserved_as_existing_not_absent(self):
        self.owner._position_ids = self.old_position = None
        self.owner._rope_deltas = self.old_delta = None
        await self.prepare(self.handle())
        self.assertEqual(
            vars(self.owner), {"_position_ids": None, "_rope_deltas": None}
        )

    async def test_unchanged_model_state_is_preserved(self):
        self.fixture.engine._process_chat_messages = lambda *args: (
            "synthetic-text",
            None,
            {},
            None,
            0,
            [],
        )
        await self.prepare(self.handle())
        self.assert_restored()

    async def test_failed_sync_keeps_boundary_unresolved_and_retains_lease(self):
        self.fixture.sync_error = RuntimeError("synthetic device wait failed")
        self.process_error = ValueError("synthetic encoder failure")
        with self.assertRaises(prep.ExecutionUnresolved):
            await self.fixture.launch()
        self.assertIs(self.owner._position_ids, self.new_position)
        self.assertIs(self.owner._rope_deltas, self.new_delta)
        self.assertEqual(self.fixture.pool.releases, [])
        self.assertTrue(
            self.fixture.execution.preparation_handle.preparation_unresolved
        )

    async def test_failed_restoration_retains_lease_and_cannot_complete_preparation(
        self,
    ):
        old_position = self.old_position

        class RefusesRestore(SimpleNamespace):
            def __setattr__(self, name, value):
                if name == "_position_ids" and value is old_position:
                    raise RuntimeError("synthetic slot restoration failed")
                super().__setattr__(name, value)

        self.owner = RefusesRestore(
            _position_ids=self.old_position, _rope_deltas=self.old_delta
        )
        self.fixture.engine._vlm_model.language_model = self.owner
        self.process_error = ValueError("synthetic encoder failure")
        with self.assertRaises(
            (
                prep.ExecutionUnresolved,
                vision.stage_fixture.coord.EngineStageUnavailable,
            )
        ) as caught:
            await self.fixture.launch()
        self.assertIsInstance(caught.exception, prep.ExecutionUnresolved)
        self.assertEqual(self.fixture.pool.releases, [])
        self.assertTrue(
            self.fixture.execution.preparation_handle.preparation_unresolved
        )

    async def test_successful_restore_precedes_failure_release(self):
        self.process_error = ValueError("synthetic encode failure")
        original_release = self.fixture.pool.release_engine

        def release(*args, **kwargs):
            self.assert_restored()
            return original_release(*args, **kwargs)

        self.fixture.pool.release_engine = release
        with self.assertRaises(vision.stage_fixture.coord.EngineStageUnavailable):
            await self.fixture.launch()
        self.assertEqual(self.fixture.pool.releases, ["m"])
        self.assert_restored()

    async def test_ordinary_preparation_keeps_existing_slot_behavior(self):
        await self.fixture.engine._stream_chat_prompt([], None, {})
        self.assertIs(self.owner._position_ids, self.new_position)
        self.assertIs(self.owner._rope_deltas, self.new_delta)


if __name__ == "__main__":
    unittest.main()
