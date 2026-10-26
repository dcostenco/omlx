# SPDX-License-Identifier: Apache-2.0
"""Request-local cache bypass: CPU images, actual methods, stubbed encoder."""

import asyncio
import ast
import __future__
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import importlib
import logging
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import test_provider_control_registration as registration
import test_provider_owned_admission_stage as stage_fixture
import test_provider_owned_vision_preparation as vision_fixture
import test_provider_owned_images as image_fixture
import numpy as np
from PIL import Image
from omlx.owned_image_cache import image_cache_allowed, without_image_cache
from omlx import owned_preparation as prep

ROOT = registration.support.ROOT
images = importlib.import_module("omlx.utils.image")


class DecodedCacheTests(unittest.TestCase):
    def setUp(self):
        self.cache = OrderedDict()
        for name, value in [
            ("_image_decode_cache", self.cache),
            ("_image_decode_cache_bytes", 0),
        ]:
            patch = mock.patch.object(images, name, value)
            patch.start()
            self.addCleanup(patch.stop)
        self.a = image_fixture.uri(image_fixture.encoded_image())
        self.b = image_fixture.uri(image_fixture.encoded_image(size=(9, 9)))

    def test_bypass_ignores_hits_and_preserves_cache_entries_accounting_and_lru(self):
        cached = images.load_image(self.a)
        images.load_image(self.b)
        before = list(self.cache.items()), images._image_decode_cache_bytes
        with without_image_cache():
            result = images.load_image(self.a)
            self.assertIsNot(result, cached)
            self.assertEqual(result.tobytes(), cached.tobytes())
            self.assertEqual(
                (list(self.cache.items()), images._image_decode_cache_bytes), before
            )
        self.assertIs(images.load_image(self.a), cached)

    def test_bypass_miss_extract_and_failure_do_not_populate_but_ordinary_calls_do(
        self,
    ):
        with without_image_cache():
            _, extracted, _ = images.extract_images_from_messages(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_image", "image_url": self.a},
                            {"type": "input_image", "image_url": self.b},
                        ],
                    }
                ]
            )
            self.assertEqual([im.size for im in extracted], [(8, 8), (9, 9)])
            with self.assertRaises(Exception):
                images.load_image("data:image/png;base64,AAAA")
        self.assertEqual(list(self.cache), [])
        self.assertEqual(images._image_decode_cache_bytes, 0)
        images.load_image(self.a)
        self.assertEqual(len(self.cache), 1)

    def test_extraction_bypasses_populated_cache_hits_and_lru_but_ordinary_extract_reuses(
        self,
    ):
        cached = images.load_image(self.a)
        images.load_image(self.b)
        before = list(self.cache.items()), images._image_decode_cache_bytes
        messages = [
            {"role": "user", "content": [{"type": "input_image", "image_url": self.a}]}
        ]
        with without_image_cache():
            _, extracted, _ = images.extract_images_from_messages(messages)
            self.assertIsNot(extracted[0], cached)
            self.assertEqual(extracted[0].tobytes(), cached.tobytes())
            self.assertEqual(
                (list(self.cache.items()), images._image_decode_cache_bytes), before
            )
        _, extracted, _ = images.extract_images_from_messages(messages)
        self.assertIs(extracted[0], cached)


def vlm_methods():
    tree = ast.parse((ROOT / "omlx/engine/vlm.py").read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "VLMBatchedEngine"
    )
    names = {
        "_prepare_vision_inputs",
        "_try_build_cached_vision_inputs",
        "_encode_missing_vision_features",
        "_cached_video_features",
    }
    methods = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    assert {n.name for n in methods} == names
    wrapper = ast.ClassDef(
        name="VLM",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
        type_params=[],
    )
    scope = dict(
        image_cache_allowed=image_cache_allowed,
        mx=SimpleNamespace(
            eval=lambda value: None,
            concatenate=lambda values, axis: np.concatenate(values, axis),
        ),
        logger=logging.getLogger("cache-cpu"),
        _GRID_VISION_MODELS=set(),
        _QWEN_VISION_MODELS=set(),
        _CACHED_INPUT_FAST_PREPARE_MODEL_TYPES=set(),
        SINGLE_IMAGE_ONLY_MODELS=set(),
        COHERE2_MOE_MODEL_TYPE="cohere2_moe",
        QWEN4_EXP_MODEL_TYPE="qwen4_exp",
        compute_image_hash=images.compute_image_hash,
        compute_per_image_hashes=images.compute_per_image_hashes,
        detect_and_strip_partial=lambda messages: False,
        _apply_minimax_m3_thinking_mode=lambda *args: None,
        apply_chat_template_with_reasoning_effort_fallback=lambda *args, **kwargs: "<image>text",
        _capture_vlm_position_state=lambda *args: None,
        _grid_row=lambda *args: None,
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])),
            str(ROOT / "omlx/engine/vlm.py"),
            "exec",
            flags=__future__.annotations.compiler_flag,
            dont_inherit=True,
        ),
        scope,
    )
    return scope["VLM"]


class VisionFeatureCacheTests(unittest.TestCase):
    def test_real_engine_module_imports_the_policy_used_by_extracted_methods(self):
        tree = ast.parse((ROOT / "omlx/engine/vlm.py").read_text())
        self.assertTrue(
            any(
                isinstance(node, ast.ImportFrom)
                and node.module == "owned_image_cache"
                and node.level == 2
                and any(alias.name == "image_cache_allowed" for alias in node.names)
                for node in tree.body
            )
        )

    def setUp(self):
        self.cls = vlm_methods()
        self.engine = self.cls()
        self.engine.model_type = "synthetic-model"
        self.engine._model_name = "synthetic-model"
        self.engine._enable_thinking = None
        self.engine._processor = SimpleNamespace(image_token="<image>")
        self.cache = mock.Mock()
        self.feature = np.ones((1, 2))
        self.cache.get.return_value = self.feature
        self.engine._vision_cache = self.cache
        self.engine._vision_cache_enabled = True
        self.embeds = np.ones((1, 2, 2))
        self.model = SimpleNamespace(
            config=SimpleNamespace(),
            get_input_embeddings=mock.Mock(
                return_value=SimpleNamespace(inputs_embeds=self.embeds)
            ),
        )
        self.engine._vlm_model = self.model
        self.engine._format_messages_for_vlm_template = lambda *args, **kwargs: ([], [])
        self.engine._chat_template_target = lambda *args: None
        self.engine._image_token_count = lambda *args: 1
        self.engine._vision_features_match_image_tokens = lambda *args: True
        self.engine._language_prompt_kwargs = lambda *args: {}
        self.engine._compute_vision_features = mock.Mock(return_value=self.feature)
        self.engine._split_vision_features = lambda *args: [self.feature]
        utils = ModuleType("mlx_vlm.utils")
        utils.load_audio = mock.Mock(side_effect=AssertionError("audio not expected"))
        utils.prepare_inputs = lambda *args, **kwargs: {
            "input_ids": np.array([[99, 1]]),
            "pixel_values": np.zeros((1, 1)),
        }
        prompts = ModuleType("mlx_vlm.prompt_utils")
        prompts.apply_chat_template = mock.Mock(side_effect=AssertionError("fallback"))
        prompts.get_chat_template = mock.Mock(side_effect=AssertionError("fallback"))
        package = ModuleType("mlx_vlm")
        patch = mock.patch.dict(
            sys.modules,
            {
                "mlx_vlm": package,
                "mlx_vlm.utils": utils,
                "mlx_vlm.prompt_utils": prompts,
            },
        )
        patch.start()
        self.addCleanup(patch.stop)
        self.picture = Image.new("RGB", (8, 8), "green")
        self.addCleanup(self.picture.close)

    def prepare(self):
        return self.engine._prepare_vision_inputs([], [self.picture])

    def test_owned_full_preparation_ignores_feature_hits_and_leaves_settings_unchanged(
        self,
    ):
        with without_image_cache():
            result = self.prepare()
        self.assertIs(result[1], self.embeds)
        self.assertEqual(result[0], [99, 1])
        self.assertEqual(self.cache.mock_calls, [])
        self.assertNotIn(
            "cached_image_features", self.model.get_input_embeddings.call_args.kwargs
        )
        self.assertIs(self.engine._vision_cache_enabled, True)

    def test_ordinary_full_preparation_still_uses_existing_feature_hits(self):
        self.prepare()
        self.cache.get.assert_called_once()
        np.testing.assert_array_equal(
            self.model.get_input_embeddings.call_args.kwargs["cached_image_features"],
            self.feature,
        )

    def test_ordinary_miss_still_stores_features(self):
        self.cache.get.return_value = None
        self.prepare()
        self.cache.put.assert_called_once()

    def test_fast_path_ordinary_hit_reaches_cache_and_owned_scope_bypasses_it(self):
        scope = self.engine._try_build_cached_vision_inputs.__globals__
        scope["_CACHED_INPUT_FAST_PREPARE_MODEL_TYPES"] = {"synthetic-model"}
        scope["mx"] = SimpleNamespace(
            array=np.array,
            ones=np.ones,
            zeros=np.zeros,
            concatenate=np.concatenate,
            int32=np.int32,
            float32=np.float32,
        )
        tokenizer = mock.Mock(return_value={"input_ids": [7]})
        tokens = {1: "<V>", 99: "<I>", 2: "<E>"}
        tokenizer.convert_ids_to_tokens.side_effect = tokens.__getitem__
        self.engine._processor.tokenizer = tokenizer
        self.engine._processor.image_processor = SimpleNamespace(merge_size=1)
        self.model.config = SimpleNamespace(
            vision_start_token_id=1, image_token_id=99, vision_end_token_id=2
        )
        self.cache.get_grid.return_value = [1, 1, 1]
        prompt = "<V><I><E>"
        result = self.engine._try_build_cached_vision_inputs(prompt, [self.picture])
        self.assertIsNotNone(result)
        self.cache.get.assert_called_once()
        self.cache.get_grid.assert_called_once()
        before = list(self.cache.mock_calls)
        with without_image_cache():
            self.assertIsNone(
                self.engine._try_build_cached_vision_inputs(prompt, [self.picture])
            )
        self.assertEqual(self.cache.mock_calls, before)

    def test_partial_path_ordinary_miss_stores_and_owned_scope_bypasses_it(self):
        scope = self.engine._encode_missing_vision_features.__globals__
        scope["_QWEN_VISION_MODELS"] = {"synthetic-model"}
        scope["_grid_row"] = lambda *args: [1, 1, 1]
        scope["mx"] = SimpleNamespace(
            array=np.array, concatenate=np.concatenate, eval=lambda value: None
        )
        self.model.vision_tower = SimpleNamespace(spatial_merge_size=1)
        args = (
            np.zeros((2, 1)),
            {"image_grid_thw": np.array([[1, 1, 1], [1, 1, 1]])},
            [self.feature, None],
            ["synthetic-first", "synthetic-second"],
            2,
        )
        result = self.engine._encode_missing_vision_features(*args)
        self.assertIsNotNone(result)
        self.cache.put.assert_called_once()
        before = list(self.cache.mock_calls)
        with without_image_cache():
            self.assertIsNone(self.engine._encode_missing_vision_features(*args))
        self.assertEqual(self.cache.mock_calls, before)

    def test_video_path_ordinary_hit_reaches_cache_and_owned_scope_bypasses_it(self):
        result = self.engine._cached_video_features("synthetic-video", None, {})
        self.assertIs(result, self.feature)
        self.cache.get.assert_called_once()
        before = list(self.cache.mock_calls)
        with without_image_cache():
            self.assertIsNone(
                self.engine._cached_video_features("synthetic-video", None, {})
            )
        self.assertEqual(self.cache.mock_calls, before)

    def test_all_specialized_feature_cache_paths_exit_before_access_when_bypassed(self):
        with without_image_cache():
            self.assertIsNone(
                self.engine._try_build_cached_vision_inputs("<image>", [self.picture])
            )
            self.assertIsNone(
                self.engine._encode_missing_vision_features(None, {}, [], [], None)
            )
            self.assertIsNone(self.engine._cached_video_features("synthetic", None, {}))
        self.assertEqual(self.cache.mock_calls, [])


class OwnedCacheScopeTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_preflight_scope_restores_after_failure(self):
        fixture = stage_fixture.AdmissionStageTests()
        await fixture.asyncSetUp()
        try:
            observed = []
            fixture.engine.on_preflight = lambda: observed.append(image_cache_allowed())
            fixture.engine.preflight_error = ValueError("synthetic preflight refusal")
            with self.assertRaises(stage_fixture.coord.EngineStageUnavailable):
                await fixture.stage(fixture.execution())
            self.assertEqual(observed, [False])
            self.assertTrue(image_cache_allowed())
            self.assertEqual(fixture.pool.releases, ["m"])
        finally:
            await fixture.asyncTearDown()

    async def test_actual_owned_worker_sets_scope_and_restores_thread_after_failure(
        self,
    ):
        fixture = vision_fixture.VisionPreparationTests()
        await fixture.asyncSetUp()
        try:
            observed = []
            process = fixture.engine._process_chat_messages

            def capture(*args):
                observed.append(image_cache_allowed())
                return process(*args)

            fixture.engine._process_chat_messages = capture
            with self.assertRaises(stage_fixture.coord.EngineStageUnavailable):
                await fixture.stage(fixture.fixture.execution())
            self.assertEqual(observed, [False])
            self.assertTrue(
                await asyncio.get_running_loop().run_in_executor(
                    fixture.core._mlx_executor, image_cache_allowed
                )
            )
        finally:
            await fixture.asyncTearDown()


class CachePolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_nested_scope_restores_policy_even_after_an_exception(self):
        self.assertTrue(image_cache_allowed())
        with without_image_cache():
            self.assertFalse(image_cache_allowed())
            with self.assertRaises(ValueError), without_image_cache():
                self.assertFalse(image_cache_allowed())
                raise ValueError("synthetic failure")
            self.assertFalse(image_cache_allowed())
        self.assertTrue(image_cache_allowed())

    async def test_ordinary_task_keeps_cache_while_owned_task_is_suspended(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def owned():
            with without_image_cache():
                entered.set()
                await release.wait()
                self.assertFalse(image_cache_allowed())

        task = asyncio.create_task(owned())
        await entered.wait()
        self.assertTrue(image_cache_allowed())
        release.set()
        await task
        self.assertTrue(image_cache_allowed())

    async def test_cancelled_task_restores_its_policy(self):
        entered = asyncio.Event()
        restored = []

        async def owned():
            try:
                with without_image_cache():
                    entered.set()
                    await asyncio.Event().wait()
            finally:
                restored.append(image_cache_allowed())

        task = asyncio.create_task(owned())
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(restored, [True])
        self.assertTrue(image_cache_allowed())

    async def test_executor_requires_its_own_scope_and_restores_before_reuse(self):
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(1) as executor:

            def scoped():
                with without_image_cache():
                    return image_cache_allowed()

            with without_image_cache():
                self.assertTrue(
                    await loop.run_in_executor(executor, image_cache_allowed)
                )
            self.assertFalse(await loop.run_in_executor(executor, scoped))
            self.assertTrue(await loop.run_in_executor(executor, image_cache_allowed))
