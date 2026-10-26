import time
import unittest
import asyncio
from functools import partial
from types import MethodType, SimpleNamespace

import test_provider_owned_image_admission as admission
from omlx import owned_preparation as prep
from omlx.owned_image_policy import capture_image_policy
from omlx.api.responses_utils import convert_responses_input_to_messages


class ImagePolicyStageTests(unittest.IsolatedAsyncioTestCase):
    async def test_authorized_fresh_policy_uses_exclusive_image_acquisition(self):
        fixture = admission.ImageAdmissionTests(
            "test_qualified_image_reaches_real_stage_and_trusted_admission"
        )
        await fixture.asyncSetUp()
        try:

            async def prepare(engine, req, model_id):
                return SimpleNamespace(
                    messages=convert_responses_input_to_messages(
                        req.input, preserve_images=True
                    ),
                    chat_kwargs={"max_tokens": 5},
                )

            fixture.stage = prep.make_admission_stage(
                lambda: fixture.pool, prepare, prep.owned_request_refusal
            )
            calls = []

            async def exclusive(pool, model_id, *, fence):
                calls.append("image")
                return await pool.acquire_owned(model_id, fence=fence)

            fixture.pool.acquire_owned_image = MethodType(exclusive, fixture.pool)
            execution = fixture.execution()
            execution.settings_sha256 = "digest"
            execution.image_policy = capture_image_policy(
                "m",
                {
                    "owned_image_input": {
                        "schema": "owned-png-v1",
                        "model_id": "m",
                        "enabled": True,
                    }
                },
                "digest",
                lambda _: "digest",
            )
            task = asyncio.create_task(fixture.stage(execution))
            fixture.tasks.append(task)
            await asyncio.wait_for(fixture.until(lambda: execution.admitted), timeout=2)
            self.assertTrue(execution.admitted)
            req = fixture.s.requests.pop(execution.claim.request_id)
            fixture.s.waiting.remove(req)
            fixture.core._output_collectors[execution.claim.request_id].put(
                admission.stage.out("ok", finished=True)
            )
            await asyncio.wait_for(task, timeout=2)
            self.assertEqual(calls, ["image"])
        finally:
            await fixture.asyncTearDown()

    async def test_stale_policy_refuses_before_preparation_and_text_remains_admissible(
        self,
    ):
        fixture = admission.ImageAdmissionTests(
            "test_unknown_architecture_never_reaches_preparation"
        )
        await fixture.asyncSetUp()
        try:

            async def prepare(engine, req, model_id):
                return SimpleNamespace(
                    messages=convert_responses_input_to_messages(
                        req.input, preserve_images=True
                    ),
                    chat_kwargs={"max_tokens": 5},
                )

            fixture.stage = prep.make_admission_stage(
                lambda: fixture.pool, prepare, prep.owned_request_refusal
            )
            execution = fixture.execution()
            execution.settings_sha256 = "old"
            execution.image_policy = capture_image_policy(
                "m",
                {
                    "owned_image_input": {
                        "schema": "owned-png-v1",
                        "model_id": "synthetic-image",
                        "enabled": True,
                    }
                },
                "old",
                lambda _: "new",
            )
            with self.assertRaises(admission.stage.coord.EngineStageUnavailable):
                await asyncio.wait_for(fixture.stage(execution), timeout=2)
            self.assertEqual(fixture.prepared, [])
        finally:
            await fixture.asyncTearDown()


if __name__ == "__main__":
    unittest.main()
