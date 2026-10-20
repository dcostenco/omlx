# SPDX-License-Identifier: Apache-2.0
"""CPU tests of exact local generation wrappers through real core methods."""

import ast
import asyncio
import logging
import os
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union
import unittest
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

ROOT = Path(os.environ["OMLX_CONTROL_ROOT"])
sys.path.insert(0, str(ROOT))
from omlx.request import RequestOutput, SamplingParams


def method(path, class_name, name):
    source = ROOT / path
    tree = ast.parse(source.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    node = next(
        n
        for n in cls.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
    )
    namespace = dict(
        __name__=__name__,
        __package__="omlx.engine",
        asyncio=asyncio,
        Any=Any,
        Dict=Dict,
        List=List,
        Optional=Optional,
        Tuple=Tuple,
        Union=Union,
        SamplingParams=SamplingParams,
        RequestOutput=RequestOutput,
        GenerationOutput=SimpleNamespace,
        clean_special_tokens=lambda text: text,
        logger=logging.getLogger("cpu-identity"),
        _raise_request_output_error=Mock(),
    )
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace[name]


WRAPPERS = {
    name: method(f"omlx/engine/{name}.py", class_name, "generate")
    for name, class_name in [("batched", "BatchedEngine"), ("vlm", "VLMBatchedEngine")]
}
SPEC = {
    name: method(f"omlx/engine/{name}.py", class_name, "_pop_specprefill_kwargs")
    for name, class_name in [("batched", "BatchedEngine"), ("vlm", "VLMBatchedEngine")]
}
CORE = method("omlx/engine_core.py", "EngineCore", "generate")
ASYNC_CORE = method("omlx/engine_core.py", "AsyncEngineCore", "generate")


class RequestIdentityTests(unittest.IsolatedAsyncioTestCase):
    def fixture(self, wrapper):
        core = SimpleNamespace(
            _finished_events={},
            _output_collectors={},
            _cleanup_request=Mock(),
            abort_request=AsyncMock(),
        )

        async def admission(**kwargs):
            rid = (
                kwargs["request_id"]
                if kwargs["request_id"] is not None
                else str(uuid4())
            )
            event = asyncio.Event()
            event.set()
            core._finished_events[rid] = event
            output = SimpleNamespace(
                output_text="synthetic code",
                prompt_tokens=4,
                completion_tokens=3,
                finish_reason="stop",
                tool_calls=None,
                cached_tokens=0,
                generated_at=1.0,
                error=None,
            )
            core._output_collectors[rid] = SimpleNamespace(
                get_nowait=Mock(side_effect=[output, None])
            )
            return rid

        core.add_request = AsyncMock(side_effect=admission)
        core.generate = MethodType(CORE, core)
        async_core = SimpleNamespace(engine=core)
        async_core.generate = MethodType(ASYNC_CORE, async_core)
        engine = SimpleNamespace(
            _loaded=True,
            start=AsyncMock(),
            _engine=async_core,
            _prepare_k2_tool_grammar=Mock(),
            _pop_specprefill_kwargs=SPEC[wrapper],
            is_diffusion_model=False,
            is_ocr_model=False,
        )
        engine.generate = MethodType(WRAPPERS[wrapper], engine)
        return engine, core

    async def test_supplied_id_reaches_actual_core_admission(self):
        for name in WRAPPERS:
            with self.subTest(wrapper=name):
                engine, core = self.fixture(name)
                result = await engine.generate(
                    "synthetic prompt", _request_id="synthetic-owned-id"
                )
                self.assertEqual(
                    core.add_request.call_args.kwargs["request_id"],
                    "synthetic-owned-id",
                )
                self.assertEqual(result.text, "synthetic code")
                core._cleanup_request.assert_called_once_with("synthetic-owned-id")
                engine.start.assert_not_awaited()

    async def test_absence_or_none_keeps_default_allocation(self):
        for name in WRAPPERS:
            for kwargs in [{}, {"_request_id": None}]:
                with self.subTest(wrapper=name, kwargs=kwargs):
                    engine, core = self.fixture(name)
                    await engine.generate("synthetic prompt", **kwargs)
                    self.assertIsNone(core.add_request.call_args.kwargs["request_id"])
                    self.assertNotIn("_request_id", core.add_request.call_args.kwargs)
                    self.assertEqual(core._cleanup_request.call_count, 1)

    async def test_sampling_tools_reasoning_and_prefill_preserved(self):
        for name in WRAPPERS:
            with self.subTest(wrapper=name):
                engine, core = self.fixture(name)
                tools = [{"type": "function", "name": "synthetic_tool"}]
                await engine.generate(
                    "synthetic prompt",
                    _request_id="synthetic-owned-id",
                    max_tokens=91,
                    temperature=0.4,
                    top_p=0.8,
                    top_k=7,
                    thinking_budget=256,
                    stop=["synthetic_stop"],
                    seed=17,
                    tools=tools,
                    preserve_reasoning=True,
                    specprefill=False,
                    specprefill_keep_pct=0.6,
                    generation_prompt_text="marker",
                )
                args = core.add_request.call_args.kwargs
                self.assertIs(args["tools"], tools)
                self.assertIs(args["preserve_reasoning"], True)
                self.assertIs(args["specprefill"], False)
                self.assertEqual(args["specprefill_keep_pct"], 0.6)
                self.assertEqual(args["generation_prompt_text"], "marker")
                sampling = args["sampling_params"]
                self.assertEqual(
                    (
                        sampling.max_tokens,
                        sampling.temperature,
                        sampling.top_p,
                        sampling.top_k,
                    ),
                    (91, 0.4, 0.8, 7),
                )
                self.assertEqual(
                    (sampling.thinking_budget, sampling.seed, sampling.stop),
                    (256, 17, ["synthetic_stop"]),
                )

    async def test_vlm_inputs_and_ocr_stop_ids_preserved(self):
        engine, core = self.fixture("vlm")
        embeds, extra = object(), {"synthetic": True}
        engine.is_ocr_model = True
        engine._resolve_ocr_stop_token_ids = Mock(return_value=[73])
        await engine.generate(
            [1, 2],
            _request_id="synthetic-vision-id",
            vlm_inputs_embeds=embeds,
            vlm_extra_kwargs=extra,
            vlm_image_hash="synthetic-image",
            vlm_cache_key_start=2,
            vlm_cache_key_ranges=[(2, "synthetic-range")],
        )
        args = core.add_request.call_args.kwargs
        self.assertIs(args["vlm_inputs_embeds"], embeds)
        self.assertIs(args["vlm_extra_kwargs"], extra)
        self.assertEqual(args["vlm_cache_key_ranges"], [(2, "synthetic-range")])
        self.assertEqual(args["sampling_params"].stop_token_ids, [73])
        self.assertEqual(args["request_id"], "synthetic-vision-id")

    async def test_diffusion_keeps_existing_stream_delegate(self):
        engine, core = self.fixture("vlm")
        engine.is_diffusion_model = True
        captured = []

        async def stream(prompt, **kwargs):
            captured.append(kwargs)
            yield SimpleNamespace(
                new_text="synthetic",
                prompt_tokens=2,
                completion_tokens=1,
                finish_reason="stop",
            )

        engine.stream_generate = stream
        result = await engine.generate(
            "synthetic", _request_id="synthetic-diffusion-id"
        )
        self.assertEqual(captured[0]["_request_id"], "synthetic-diffusion-id")
        self.assertEqual(result.text, "synthetic")
        core.add_request.assert_not_awaited()

    async def test_separate_requests_keep_their_ids(self):
        for name in WRAPPERS:
            with self.subTest(wrapper=name):
                engine, core = self.fixture(name)
                await asyncio.gather(
                    engine.generate("one", _request_id="synthetic-one"),
                    engine.generate("two", _request_id="synthetic-two"),
                )
                self.assertEqual(
                    {
                        call.kwargs["request_id"]
                        for call in core.add_request.call_args_list
                    },
                    {"synthetic-one", "synthetic-two"},
                )
                self.assertEqual(
                    {call.args[0] for call in core._cleanup_request.call_args_list},
                    {"synthetic-one", "synthetic-two"},
                )

    async def test_admission_error_propagates_with_original_id(self):
        for name in WRAPPERS:
            with self.subTest(wrapper=name):
                engine, core = self.fixture(name)
                core.add_request.side_effect = RuntimeError(
                    "synthetic_admission_failure"
                )
                with self.assertRaisesRegex(
                    RuntimeError, "synthetic_admission_failure"
                ):
                    await engine.generate(
                        "synthetic", _request_id="synthetic-failed-id"
                    )
                self.assertEqual(
                    core.add_request.call_args.kwargs["request_id"],
                    "synthetic-failed-id",
                )
                core._cleanup_request.assert_not_called()


    async def test_cancelled_owned_generate_is_left_to_its_coordinator(self):
        # 25b.4: an owned request is cleaned up with proof by its coordinator,
        # so caller cancellation must not issue the ordinary abort.
        for owned in (True, False):
            with self.subTest(owned=owned):
                core = SimpleNamespace(
                    _finished_events={},
                    _output_collectors={},
                    _cleanup_request=Mock(),
                    abort_request=AsyncMock(),
                )

                async def admission(**kwargs):
                    rid = kwargs["request_id"]
                    core._finished_events[rid] = asyncio.Event()  # never set
                    core._output_collectors[rid] = SimpleNamespace(
                        get_nowait=Mock(return_value=None)
                    )
                    return rid

                core.add_request = AsyncMock(side_effect=admission)
                core.generate = MethodType(CORE, core)
                extra = {"_owned_admission": object()} if owned else {}
                task = asyncio.create_task(
                    core.generate("synthetic prompt", request_id="synthetic-cancelled", **extra)
                )
                for _ in range(5):
                    await asyncio.sleep(0)
                self.assertIn("synthetic-cancelled", core._finished_events)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                if owned:
                    core.abort_request.assert_not_awaited()
                    core._cleanup_request.assert_not_called()
                    self.assertIn("synthetic-cancelled", core._output_collectors)
                else:
                    core.abort_request.assert_awaited_once_with("synthetic-cancelled")
                    core._cleanup_request.assert_called_once_with("synthetic-cancelled")

if __name__ == "__main__":
    unittest.main()
