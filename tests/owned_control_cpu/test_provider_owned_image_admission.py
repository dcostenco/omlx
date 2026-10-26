# SPDX-License-Identifier: Apache-2.0
"""Internal image admission; actual stage/executor, synthetic device/encoder."""

import asyncio
import __future__
import ast
import functools
import json
from copy import deepcopy
from functools import partial
import threading
import time
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock

import test_provider_owned_admission_stage as stage
import test_provider_owned_images as images
import test_provider_running_vision_cleanup as vision
from omlx import owned_preparation as prep
from omlx.owned_images import validate_owned_images
from omlx.owned_vision_preparation import prepare_owned_vision
from omlx.output_collector import RequestOutputCollector
from omlx.api.responses_utils import convert_responses_input_to_messages
from omlx.owned_admission import OwnedAdmission
from omlx.owned_image_admission import owned_image_sources, prepared_image_sources
from omlx.utils.image import extract_images_from_messages


def image_request():
    return images.parse_control_request(
        images.request_with(images.uri(images.encoded_image()))
    )[0]


class ImagePolicyTests(unittest.TestCase):
    def test_default_guard_keeps_its_existing_unrecognized_part_behavior(self):
        for kind in ["image", "image_file", "input_video", "video_url", "audio"]:
            req = stage.request(
                input=[
                    {"role": "user", "content": [{"type": kind, "text": "synthetic"}]}
                ]
            )
            self.assertIsNone(prep.owned_request_refusal(req))
            self.assertEqual(
                prep.owned_request_refusal(req, allow_images=True), "media_input"
            )

    def test_default_serialized_tool_output_remains_text(self):
        uri = images.uri(images.encoded_image())
        req = stage.request(
            input=[
                {
                    "type": "function_call_output",
                    "call_id": "synthetic",
                    "output": json.dumps([{"type": "input_image", "image_url": uri}]),
                }
            ]
        )
        self.assertIsInstance(req.input[0].output, str)
        converted = convert_responses_input_to_messages(req.input, preserve_images=True)
        self.assertTrue(
            all(isinstance(message["content"], str) for message in converted)
        )
        self.assertIsNone(prep.owned_request_refusal(req))
        self.assertEqual(
            prep.owned_request_refusal(req, allow_images=True), "media_input"
        )

    def test_tool_output_images_and_nonuser_images_are_refused(self):
        uri = images.uri(images.encoded_image())
        req = stage.request(
            input=[
                {
                    "type": "function_call_output",
                    "call_id": "synthetic",
                    "output": [{"type": "input_image", "image_url": uri}],
                }
            ]
        )
        self.assertEqual(
            prep.owned_request_refusal(req, allow_images=True), "media_input"
        )
        for role in ["assistant", "system"]:
            req = stage.request(
                input=[
                    {
                        "role": role,
                        "content": [{"type": "input_image", "image_url": uri}],
                    }
                ]
            )
            self.assertEqual(
                prep.owned_request_refusal(req, allow_images=True), "media_input"
            )

    def test_detail_omitted_and_explicit_normalize_to_valid_images(self):
        uri = images.uri(images.encoded_image())
        for detail in [None, "auto", "high"]:
            part = {"type": "input_image", "image_url": uri}
            if detail is not None:
                part["detail"] = detail
            req = stage.request(input=[{"role": "user", "content": [part]}])
            self.assertIsNone(prep.owned_request_refusal(req, allow_images=True))

    def test_opt_in_is_internal_and_default_still_refuses_valid_png(self):
        req = image_request()
        before = req.model_dump(mode="json")
        self.assertEqual(prep.owned_request_refusal(req), "media_input")
        self.assertIsNone(prep.owned_request_refusal(req, allow_images=True))
        self.assertEqual(req.model_dump(mode="json"), before)

    def test_validator_returns_exact_count_without_changing_input(self):
        value = image_request().model_dump(mode="json")["input"]
        before = deepcopy(value)
        self.assertEqual(validate_owned_images(value), 1)
        self.assertEqual(validate_owned_images("text"), 0)
        self.assertEqual(value, before)

    def test_nonboolean_truthy_opt_in_does_not_allow_images(self):
        for value in [1, "true", {}, object()]:
            with self.subTest(value=type(value).__name__):
                self.assertEqual(
                    prep.owned_request_refusal(image_request(), allow_images=value),
                    "media_input",
                )

    def test_remote_jpeg_malformed_and_other_media_refuse(self):
        for part in [
            {"type": "input_image", "image_url": "https://example.invalid/image.png"},
            {
                "type": "input_image",
                "image_url": images.uri(images.encoded_image("JPEG"), "jpeg"),
            },
            {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
            {"type": "image_url", "image_url": "https://example.invalid/image.png"},
            {"type": "input_file", "file_url": "file:///private"},
            {"type": "input_audio", "data": "AAAA"},
        ]:
            value = [{"role": "user", "content": [part]}]
            req = SimpleNamespace(
                tools=None, text=None, model_dump=lambda **_: {"input": value}
            )
            with self.subTest(part_type=part["type"]):
                self.assertEqual(
                    prep.owned_request_refusal(req, allow_images=True), "media_input"
                )

    def test_existing_refusal_priority_is_preserved(self):
        self.assertEqual(
            prep.owned_request_refusal(
                image_request(), allow_images=True, quantizing=True
            ),
            "quantizing",
        )
        self.assertEqual(
            prep.owned_request_refusal(
                stage.request(
                    tools=[
                        {
                            "type": "function",
                            "name": "f",
                            "parameters": {"type": "object"},
                        }
                    ]
                ),
                allow_images=True,
            ),
            "tools",
        )
        self.assertEqual(
            prep.owned_request_refusal(
                stage.request(text={"format": {"type": "json_object"}}),
                allow_images=True,
            ),
            "text_format",
        )


class ImageEngine(stage.StubEngine):
    def __init__(self, core):
        super().__init__(core)
        self.encodes = []

    async def admit_owned_chat(self, messages, *, admission, request_id, **kwargs):
        self.admits.append(request_id)
        self.last_admission = admission
        await asyncio.get_running_loop().run_in_executor(
            self.core._mlx_executor,
            partial(
                prepare_owned_vision,
                admission,
                self.core,
                lambda: self.encodes.append(request_id),
                dispatch_executor=self.core._mlx_executor,
                stream_context=vision.running.fake_mx.stream,
                synchronize=vision.running.fake_mx.synchronize,
                language_model=self.core.scheduler.model._language_model,
            ),
        )
        await self.core.add_request(
            "synthetic-image",
            request_id=request_id,
            _owned_admission=admission,
            vlm_inputs_embeds=vision.Payload(),
            vlm_extra_kwargs={"position_ids": vision.Payload()},
        )


class ImageAdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = vision.RunningVisionCleanupTests()
        await self.fixture.asyncSetUp()
        self.core, self.s = self.fixture.core, self.fixture.s
        for name in ("probe_owned_absence", "release_owned_failure"):
            setattr(
                self.core, name, MethodType(stage.plumbing.core_ns[name], self.core)
            )
        self.s._probe_owned_presence = MethodType(
            stage.plumbing.sched_ns["_probe_owned_presence"], self.s
        )
        self.saved_collector = stage.admission_ns["RequestOutputCollector"]
        stage.admission_ns["RequestOutputCollector"] = RequestOutputCollector
        self.engine = ImageEngine(self.core)
        self.pool = stage.StubPool(self.engine)
        self.prepared = []
        self.tasks = []

        async def prepare(engine, req, model_id):
            self.prepared.append(req)
            return SimpleNamespace(
                messages=convert_responses_input_to_messages(
                    req.input, preserve_images=True
                ),
                chat_kwargs={"max_tokens": 5},
            )

        self.stage = prep.make_admission_stage(
            lambda: self.pool,
            prepare,
            partial(prep.owned_request_refusal, allow_images=True),
        )

    async def asyncTearDown(self):
        for task in self.tasks:
            if not task.done():
                task.cancel()
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        stage.admission_ns["RequestOutputCollector"] = self.saved_collector
        await self.fixture.asyncTearDown()

    def execution(self):
        return stage.FakeExecution(
            image_request(), "synthetic-image", time.monotonic() + 3
        )

    async def until(self, condition):
        deadline = time.monotonic() + 3
        while not condition():
            if time.monotonic() >= deadline:
                raise AssertionError("condition not reached")
            await asyncio.sleep(0.005)

    async def assert_refused_before_preparation(self):
        with self.assertRaises(stage.coord.EngineStageUnavailable):
            await self.stage(self.execution())
        self.assertEqual(self.prepared, [])
        self.assertEqual(self.engine.admits, [])
        self.assertEqual(self.pool.releases, ["m"])

    async def test_qualified_image_reaches_real_stage_and_trusted_admission(self):
        observations = []
        original = self.s._owned_vision_refusal

        def observe(*args, **kwargs):
            observations.append(threading.get_ident())
            return original(*args, **kwargs)

        self.s._owned_vision_refusal = observe
        execution = self.execution()
        task = asyncio.create_task(self.stage(execution))
        self.tasks.append(task)
        await self.until(lambda: execution.admitted)
        self.assertTrue(execution.preparation_handle._prepared_vision)
        self.assertEqual(len(observations), 3)
        self.assertTrue(all(t != threading.get_ident() for t in observations))
        req = self.s.requests.pop(execution.claim.request_id)
        self.s.waiting.remove(req)
        self.core._output_collectors[execution.claim.request_id].put(
            stage.out("synthetic image", finished=True)
        )
        self.assertEqual(await task, stage.COMPLETED)
        self.assertEqual(self.pool.releases, ["m"])
        self.assertEqual(self.core._output_collectors, {})

    async def test_unknown_architecture_never_reaches_preparation(self):
        self.s.model._vlm_model.config.model_type = "synthetic-unqualified"
        await self.assert_refused_before_preparation()

    async def test_noncanonical_architecture_never_reaches_preparation(self):
        self.s.model._vlm_model.config.model_type = "Qwen3-5"
        await self.assert_refused_before_preparation()

    async def test_gemma_architecture_never_reaches_preparation(self):
        self.s.model._vlm_model.config.model_type = "gemma4"
        await self.assert_refused_before_preparation()

    async def test_qwen_vl_architecture_never_reaches_preparation(self):
        self.s.model._vlm_model.config.model_type = "qwen3_vl"
        await self.assert_refused_before_preparation()

    async def test_foreign_collector_never_reaches_preparation(self):
        self.core._output_collectors["foreign"] = object()
        try:
            await self.assert_refused_before_preparation()
        finally:
            self.core._output_collectors.pop("foreign")

    async def test_orphan_generator_row_never_reaches_preparation(self):
        self.s.batch_generator.stages[0].append(7)
        await self.assert_refused_before_preparation()

    async def test_pending_adapter_state_never_reaches_preparation(self):
        self.s.model._pending_embeds = vision.Payload()
        await self.assert_refused_before_preparation()

    async def test_missing_observation_never_reaches_preparation(self):
        self.s._owned_vision_refusal = None
        await self.assert_refused_before_preparation()

    async def test_state_change_after_preflight_refuses_before_encoder(self):
        self.engine.on_preflight = lambda: self.s.batch_generator.stages[0].append(7)
        with self.assertRaises(stage.coord.EngineStageUnavailable):
            await self.stage(self.execution())
        self.assertEqual(len(self.prepared), 1)
        self.assertEqual(len(self.engine.preflights), 1)
        self.assertEqual(self.engine.admits, [])
        self.assertEqual(self.pool.releases, ["m"])

    async def test_preflight_position_mutation_refuses_before_encoder(self):
        self.engine.on_preflight = lambda: setattr(
            self.s.model._language_model, "_position_ids", vision.Payload()
        )
        with self.assertRaises(stage.coord.EngineStageUnavailable):
            await self.stage(self.execution())
        self.assertEqual(self.engine.encodes, [])
        self.assertEqual(self.pool.releases, ["m"])

    async def test_interposed_executor_job_is_caught_inside_encoder_job(self):
        original = self.engine.admit_owned_chat

        async def interposed(*args, **kwargs):
            await asyncio.get_running_loop().run_in_executor(
                self.core._mlx_executor,
                lambda: self.s.batch_generator.stages[0].append(7),
            )
            await original(*args, **kwargs)

        self.engine.admit_owned_chat = interposed
        with self.assertRaises(stage.coord.EngineStageUnavailable):
            await self.stage(self.execution())
        self.assertEqual(self.engine.encodes, [])
        self.assertEqual(self.s.requests, {})
        self.assertEqual(self.pool.releases, ["m"])

    async def test_prepared_messages_must_preserve_image_source(self):
        async def dropped(*_):
            return SimpleNamespace(
                messages=[{"role": "user", "content": "dropped image"}], chat_kwargs={}
            )

        self.stage = prep.make_admission_stage(
            lambda: self.pool,
            dropped,
            partial(prep.owned_request_refusal, allow_images=True),
        )
        with self.assertRaises(stage.coord.EngineStageUnavailable):
            await self.stage(self.execution())
        self.assertEqual(self.engine.preflights, [])
        self.assertEqual(self.engine.encodes, [])
        self.assertEqual(self.pool.releases, ["m"])

    async def test_text_request_cannot_gain_an_image_in_prepared_messages(self):
        req = image_request()

        async def injected(*_):
            return SimpleNamespace(
                messages=convert_responses_input_to_messages(
                    req.input, preserve_images=True
                ),
                chat_kwargs={},
            )

        self.stage = prep.make_admission_stage(
            lambda: self.pool,
            injected,
            partial(prep.owned_request_refusal, allow_images=True),
        )
        execution = stage.FakeExecution(
            stage.request(), "synthetic-injected", time.monotonic() + 3
        )
        with self.assertRaises(stage.coord.EngineStageUnavailable):
            await self.stage(execution)
        self.assertEqual(self.engine.preflights, [])
        self.assertEqual(self.engine.encodes, [])
        self.assertEqual(self.pool.releases, ["m"])

    async def test_validation_change_during_acquisition_is_stage_refusal(self):
        original = self.pool.acquire_owned
        previous = images.ImageFile.LOAD_TRUNCATED_IMAGES

        async def changed(*args, **kwargs):
            lease = await original(*args, **kwargs)
            images.ImageFile.LOAD_TRUNCATED_IMAGES = True
            return lease

        self.pool.acquire_owned = changed
        try:
            await self.assert_refused_before_preparation()
        finally:
            images.ImageFile.LOAD_TRUNCATED_IMAGES = previous

    async def actual_vlm_admission(self, *, mismatched=False):
        plumbing = stage.plumbing
        wrapper = SimpleNamespace(
            engine=self.core,
            _mlx_executor=self.core._mlx_executor,
            add_request=self.core.add_request,
        )
        host = plumbing.VLMHost(wrapper)
        cls = type("ImageVLM", (plumbing.VLMNew, plumbing.VLMHost), {})
        engine = cls.__new__(cls)
        engine.__dict__.update(host.__dict__)
        self.s.model._vlm_model.language_model = self.s.model._language_model
        engine._vlm_model = (
            SimpleNamespace(language_model=object())
            if mismatched
            else self.s.model._vlm_model
        )
        engine.core = self.core
        engine.preflight_chat = self.engine.preflight_chat
        encodes = []

        def encode(*_):
            encodes.append(threading.get_ident())
            return (
                "synthetic actual VLM",
                vision.Payload(),
                {"position_ids": vision.Payload()},
                "synthetic-hash",
                0,
                None,
            )

        engine._process_chat_messages = encode
        self.pool.engine = engine
        namespace = plumbing.VLMNew._stream_chat_prompt.__globals__
        keys = ("mx", "functools", "prepare_owned_vision")
        saved = {key: namespace.get(key) for key in keys}
        namespace.update(
            mx=vision.running.fake_mx,
            functools=functools,
            prepare_owned_vision=prepare_owned_vision,
        )
        execution = self.execution()
        try:
            if mismatched:
                with self.assertRaises(stage.coord.EngineStageUnavailable):
                    await self.stage(execution)
                self.assertEqual(encodes, [])
                self.assertFalse(execution.preparation_handle._preparation_started)
                self.assertEqual(self.pool.releases, ["m"])
                return
            observations = []
            original = self.s._owned_vision_refusal

            def observe(*args, **kwargs):
                observations.append(threading.get_ident())
                return original(*args, **kwargs)

            self.s._owned_vision_refusal = observe
            task = asyncio.create_task(self.stage(execution))
            self.tasks.append(task)
            await self.until(lambda: execution.admitted or task.done())
            if task.done():
                await task
                self.fail("stage ended before admission")
            self.assertTrue(execution.preparation_handle._prepared_vision)
            self.assertEqual(len(encodes), 1)
            self.assertEqual(len(observations), 3)
            self.assertTrue(all(t == encodes[0] for t in observations))
            req = self.s.requests.pop(execution.claim.request_id)
            self.s.waiting.remove(req)
            self.core._output_collectors[execution.claim.request_id].put(
                stage.out("synthetic actual image", finished=True)
            )
            self.assertEqual(await task, stage.COMPLETED)
            self.assertEqual(self.pool.releases, ["m"])
        finally:
            namespace.update(saved)

    async def test_actual_vlm_prompt_and_admit_honor_image_flag(self):
        await self.actual_vlm_admission()

    async def test_actual_vlm_prompt_refuses_replaced_language_model(self):
        await self.actual_vlm_admission(mismatched=True)

    async def test_encoder_guard_rejects_different_language_model_before_begin(self):
        handle = OwnedAdmission(self.core, "synthetic-mismatch", time.monotonic() + 3)
        handle._image_admission = True
        encode = Mock()
        with self.assertRaisesRegex(ValueError, "language_model_mismatch"):
            await asyncio.get_running_loop().run_in_executor(
                self.core._mlx_executor,
                partial(
                    prepare_owned_vision,
                    handle,
                    self.core,
                    encode,
                    dispatch_executor=self.core._mlx_executor,
                    stream_context=vision.running.fake_mx.stream,
                    synchronize=vision.running.fake_mx.synchronize,
                    language_model=object(),
                ),
            )
        encode.assert_not_called()
        self.assertFalse(handle._preparation_started)

    async def test_encoder_guard_rejects_none_language_models(self):
        self.s.model._language_model = None
        handle = OwnedAdmission(self.core, "synthetic-none", time.monotonic() + 3)
        handle._image_admission = True
        encode = Mock()
        with self.assertRaisesRegex(ValueError, "language_model_mismatch"):
            await asyncio.get_running_loop().run_in_executor(
                self.core._mlx_executor,
                partial(
                    prepare_owned_vision,
                    handle,
                    self.core,
                    encode,
                    dispatch_executor=self.core._mlx_executor,
                    stream_context=vision.running.fake_mx.stream,
                    synchronize=vision.running.fake_mx.synchronize,
                    language_model=None,
                ),
            )
        encode.assert_not_called()
        self.assertFalse(handle._preparation_started)

    async def test_cancellation_retains_lease_until_guard_worker_settles(self):
        entered, release = threading.Event(), threading.Event()
        original = self.s._owned_vision_refusal

        def held(*args, **kwargs):
            entered.set()
            if not release.wait(3):
                raise TimeoutError("synthetic guard not released")
            return original(*args, **kwargs)

        self.s._owned_vision_refusal = held
        task = asyncio.create_task(self.stage(self.execution()))
        self.tasks.append(task)
        try:
            await self.until(entered.is_set)
            task.cancel()
            await asyncio.wait((task,), timeout=0.05)
            self.assertFalse(task.done())
            self.assertEqual(self.pool.releases, [])
            self.assertEqual(self.engine.encodes, [])
        finally:
            release.set()
        with self.assertRaises(stage.coord.EngineStageUnavailable):
            await task
        self.assertEqual(self.pool.releases, ["m"])
        self.assertEqual(self.engine.encodes, [])


class ActualPreflightTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_preflight_extracts_exact_images_without_encoder(self):
        path = vision.running.ROOT / "omlx/engine/vlm.py"
        tree = ast.parse(path.read_text())
        engine = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "VLMBatchedEngine"
        )
        wanted = {
            "preflight_chat",
            "_extract_request_media",
            "_count_image_tokens_real",
        }
        nodes = [
            n
            for n in [*tree.body, *engine.body]
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name in wanted
        ]
        self.assertEqual({n.name for n in nodes}, wanted)
        namespace = {
            "extract_images_from_messages": extract_images_from_messages,
            "_IMAGE_TOKEN_UPPER_BOUND_FALLBACK": 1024,
            "_derive_image_token_upper_bound": lambda _: 1024,
        }
        exec(
            compile(
                ast.Module(body=nodes, type_ignores=[]),
                str(path),
                "exec",
                flags=__future__.annotations.compiler_flag,
            ),
            namespace,
        )
        req = image_request()
        messages = convert_responses_input_to_messages(req.input, preserve_images=True)
        sources = owned_image_sources(req)
        self.assertEqual(prepared_image_sources(messages), sources)
        observations = []

        async def memory_check(*args, **kwargs):
            observations.append(kwargs["num_prompt_tokens"])

        host = SimpleNamespace(
            _loaded=True,
            is_diffusion_model=False,
            model_type="qwen3_5",
            _native_video=False,
            _processor=None,
            _apply_chat_template=Mock(return_value="synthetic text"),
            _tokenizer=SimpleNamespace(encode=lambda _: [1, 2]),
            _engine=SimpleNamespace(engine=SimpleNamespace(scheduler=object())),
            _preflight_or_raise_with_eviction=memory_check,
            _process_chat_messages=Mock(
                side_effect=AssertionError("encoder in preflight")
            ),
        )
        original = namespace["_extract_request_media"]
        extracted = []

        def extract(value):
            result = original(host, value)
            extracted.append(len(result[2]))
            self.assertEqual(len(result[2]), len(sources))
            self.assertEqual(prepared_image_sources(value), sources)
            return result

        host._extract_request_media = extract
        await namespace["preflight_chat"](host, messages)
        self.assertEqual(extracted, [1])
        self.assertEqual(observations, [1026])
        host._process_chat_messages.assert_not_called()


if __name__ == "__main__":
    unittest.main()
