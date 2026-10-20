# SPDX-License-Identifier: Apache-2.0
"""Slice 27.4a CPU proof: owned admission plumbing.

Part 1 runs each engine's real stream_chat/stream_generate (and their new
shared helpers and admit_owned_chat) against the verbatim oMLX 757b68e
methods, over a fake core that records add_request, stream and abort calls.
Part 2 runs the real classification-only probe and failure release on the
25b scheduler/core harness with a real single-worker executor. Prompt
templates, tokenizers and VLM encoders are stubs.
"""

import __future__
import ast
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import math
import threading
import time
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock

import test_provider_waiting_cleanup as waiting_fixture
import omlx_engine_stream_757b68e as baseline
from omlx.owned_admission import OwnedAdmission, StreamSynchronized
from omlx.request import Request, SamplingParams

ROOT = waiting_fixture.ROOT
FUTURE = __future__.annotations.compiler_flag


def class_node(name, body):
    return ast.ClassDef(name=name, bases=[], keywords=[], body=body, decorator_list=[], type_params=[])


def methods_of(tree, class_name, names):
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    found = [n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
    assert {n.name for n in found} == set(names), set(names) - {n.name for n in found}
    return found


class GenerationOutput(SimpleNamespace):
    pass


ENGINE_NS = dict(
    __package__="omlx.engine",
    __name__="omlx.engine._owned_admission_plumbing_test",
    asyncio=asyncio,
    logger=logging.getLogger("owned-admission-plumbing"),
    convert_tools_for_template=lambda tools: [("template", tool) for tool in tools],
    clean_special_tokens=lambda text: text,
    GenerationOutput=GenerationOutput,
    InvalidRequestError=ValueError,
)


def build_engine_classes(path, class_name):
    tree = ast.parse((ROOT / path).read_text())
    new = methods_of(tree, class_name, {
        "stream_generate", "stream_chat", "_stream_chat_prompt",
        "_stream_request_arguments", "admit_owned_chat", "_pop_specprefill_kwargs",
    })
    pop = [m for m in new if m.name == "_pop_specprefill_kwargs"]
    base_tree = ast.parse(open(baseline.__file__).read())
    old = methods_of(base_tree, f"Baseline{class_name}", {"stream_generate", "stream_chat"})
    module = ast.Module(body=[class_node("New", new), class_node("Old", [*old, *pop])], type_ignores=[])
    namespace = dict(ENGINE_NS)
    exec(compile(ast.fix_missing_locations(module), str(ROOT / path), "exec",
                 flags=FUTURE, dont_inherit=True), namespace)
    return namespace["New"], namespace["Old"]


BatchedNew, BatchedOld = build_engine_classes("omlx/engine/batched.py", "BatchedEngine")
VLMNew, VLMOld = build_engine_classes("omlx/engine/vlm.py", "VLMBatchedEngine")


class FakeCore:
    def __init__(self, outputs):
        self.outputs = outputs
        self.added = []
        self.aborted = []
        self.streamed = []
        self._mlx_executor = ThreadPoolExecutor(1, thread_name_prefix="fake-mlx")

    async def add_request(self, **kwargs):
        self.added.append(kwargs)
        return kwargs.get("request_id") or "generated-rid"

    async def stream_outputs(self, request_id):
        self.streamed.append(request_id)
        for output in self.outputs:
            yield output

    async def abort_request(self, request_id):
        self.aborted.append(request_id)


class BatchedHost:
    def __init__(self, core):
        self._loaded = True
        self._engine = core
        self.started = 0

    async def start(self):
        self.started += 1

    def _preprocess_messages(self, messages):
        return [dict(m, preprocessed=True) for m in messages]

    def _apply_chat_template(self, messages, tools, chat_template_kwargs=None, is_partial=None):
        return f"T({messages!r},{tools!r},{chat_template_kwargs!r},{is_partial!r})"

    def _inject_specprefill_system_end(self, messages, prompt, template_tools, ct_kwargs,
                                       kwargs, full_prompt_token_count=None):
        if ct_kwargs and ct_kwargs.get("mark"):
            # The raw count, so a dropped argument is visible.
            kwargs["specprefill_system_end"] = (
                full_prompt_token_count if full_prompt_token_count is not None else -1
            )

    def _generation_prompt_text(self, ct_kwargs, partial):
        return ("<gen>", bool(partial)) if ct_kwargs else (None, False)

    def _prepare_k2_tool_grammar(self, tools, kwargs):
        if tools:
            kwargs["compiled_grammar"] = "k2-grammar"


class VLMHost:
    def __init__(self, core, ocr=False):
        self._loaded = True
        self._engine = core
        self.is_diffusion_model = False
        self.is_ocr_model = ocr
        self.threads = []

    async def start(self):
        pass

    def _resolve_ocr_stop_token_ids(self):
        return [7, 8]

    def _process_chat_messages(self, messages, tools, kwargs):
        self.threads.append(threading.current_thread().name)
        ct = kwargs.pop("chat_template_kwargs", None)
        partial = kwargs.pop("is_partial", None)
        return (f"V({messages!r},{tools!r},{ct!r},{partial!r})", "embeds", {"k": 1}, "hash", 2, [(0, "range")])

    def _inject_specprefill_system_end(self, messages, prompt, kwargs):
        kwargs.setdefault("specprefill_system_end", 1)

    def _generation_prompt_text(self, ct_kwargs, partial):
        return ("<gen>", bool(partial)) if ct_kwargs else (None, False)


def output(text="hi", finished=True):
    return SimpleNamespace(
        output_text=text, new_text=text, prompt_tokens=2, completion_tokens=1,
        finished=finished, finish_reason="stop" if finished else None, tool_calls=None,
        cached_tokens=0,
    )


SCENARIOS = {
    "template kwargs, sampling and stop": dict(
        messages=[{"role": "user", "content": "hi"}], max_tokens=7, temperature=0.1, top_p=0.8,
        chat_template_kwargs={"enable_thinking": False, "mark": True}, is_partial=False, seed=3,
        stop=["x"], preserve_reasoning=True, repetition_context_size=64, thinking_budget=5,
        frequency_penalty=0.2, xtc_probability=0.1,
    ),
    "prepared prompt": dict(
        messages=[{"role": "user", "content": "hi"}], _prepared_prompt=("TEXT", [1, 2, 3, 4, 5]),
        chat_template_kwargs={"mark": True},
    ),
    "tools and SpecPrefill overrides": dict(
        messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        tools=[{"type": "function", "function": {"name": "f"}}], specprefill=True,
        specprefill_keep_pct=0.5, skip_cache_store=True,
    ),
    "partial assistant prefix": dict(
        messages=[{"role": "user", "content": "q"}], chat_template_kwargs={"x": 1}, is_partial=True,
    ),
}


def normalized(added):
    out = []
    for call in added:
        call = dict(call)
        call["sampling_params"] = vars(call["sampling_params"])
        out.append(call)
    return out


async def drive_stream(engine_class, host_class, scenario, outputs, *, stop_after=None, ocr=False):
    core = FakeCore(outputs)
    host = host_class(core, ocr=ocr) if host_class is VLMHost else host_class(core)
    engine = type("Engine", (engine_class, host_class), {})
    instance = engine.__new__(engine)
    instance.__dict__.update(host.__dict__)
    spec = dict(scenario)
    messages = spec.pop("messages")
    yielded = []
    stream = instance.stream_chat(messages, **spec)
    async for item in stream:
        yielded.append(vars(item))
        if stop_after is not None and len(yielded) >= stop_after:
            break
    await stream.aclose()
    return core, instance, yielded


async def drive_owned(engine_class, host_class, scenario, *, request_id="claim-rid", ocr=False, loaded=True):
    core = FakeCore([])
    host = host_class(core, ocr=ocr) if host_class is VLMHost else host_class(core)
    engine = type("Engine", (engine_class, host_class), {})
    instance = engine.__new__(engine)
    instance.__dict__.update(host.__dict__)
    instance._loaded = loaded
    spec = dict(scenario)
    messages = spec.pop("messages")
    handle = object()
    await instance.admit_owned_chat(messages, admission=handle, request_id=request_id, **spec)
    return core, instance, handle


def run(coroutine):
    return asyncio.run(asyncio.wait_for(coroutine, 20))


class EngineDifferentialTests(unittest.TestCase):
    ENGINES = (
        ("batched", BatchedNew, BatchedOld, BatchedHost, False),
        ("vlm", VLMNew, VLMOld, VLMHost, False),
        ("vlm ocr", VLMNew, VLMOld, VLMHost, True),
    )

    def test_stream_chat_is_unchanged_for_ordinary_callers(self):
        for engine_name, new, old, host, ocr in self.ENGINES:
            for name, scenario in SCENARIOS.items():
                for stop_after, outputs in ((None, [output("a", False), output("ab")]),
                                            (1, [output("a", False), output("ab")])):
                    with self.subTest(engine=engine_name, scenario=name, stop_after=stop_after):
                        got = run(drive_stream(new, host, scenario, outputs, stop_after=stop_after, ocr=ocr))
                        want = run(drive_stream(old, host, scenario, outputs, stop_after=stop_after, ocr=ocr))
                        self.assertEqual(normalized(got[0].added), normalized(want[0].added))
                        self.assertEqual(got[0].aborted, want[0].aborted)
                        self.assertEqual(got[0].streamed, want[0].streamed)
                        self.assertEqual(got[2], want[2])
                        if stop_after is not None:
                            # The early exit still aborts on the ordinary path.
                            self.assertEqual(got[0].aborted, ["generated-rid"])

    def test_owned_admission_sends_the_ordinary_request(self):
        for engine_name, new, old, host, ocr in self.ENGINES:
            for name, scenario in SCENARIOS.items():
                with self.subTest(engine=engine_name, scenario=name):
                    ordinary = run(drive_stream(new, host, scenario, [output()], ocr=ocr))[0]
                    core, instance, handle = run(drive_owned(new, host, scenario, ocr=ocr))
                    want = normalized(ordinary.added)[0]
                    got = normalized(core.added)[0]
                    self.assertIs(got.pop("_owned_admission"), handle)
                    self.assertEqual(got.pop("request_id"), "claim-rid")
                    self.assertIsNone(want.pop("request_id"))
                    self.assertEqual(got, want)
                    # No reader, no abort, ever.
                    self.assertEqual((core.aborted, core.streamed), ([], []))

    def test_vlm_prompt_preparation_runs_on_the_mlx_executor(self):
        core, instance, _ = run(drive_owned(VLMNew, VLMHost, SCENARIOS["template kwargs, sampling and stop"]))
        self.assertEqual(len(instance.threads), 1)
        self.assertTrue(instance.threads[0].startswith("fake-mlx"), instance.threads)

    def test_owned_admission_refuses_a_conflicting_id_and_an_unloaded_engine(self):
        for engine_name, new, _old, host, _ocr in self.ENGINES[:2]:
            with self.subTest(engine=engine_name):
                with self.assertRaises(ValueError):
                    run(drive_owned(new, host, dict(SCENARIOS["prepared prompt"], _request_id="other")))
                with self.assertRaises(RuntimeError):
                    run(drive_owned(new, host, SCENARIOS["prepared prompt"], loaded=False))

    def test_owned_admission_rejects_what_stream_chat_would_reject(self):
        with self.assertRaises(TypeError):
            run(drive_owned(BatchedNew, BatchedHost, dict(SCENARIOS["prepared prompt"], prompt="x")))
        with self.assertRaises(TypeError):
            run(drive_owned(VLMNew, VLMHost, dict(SCENARIOS["prepared prompt"], vlm_image_hash="h")))

    def test_vlm_owned_admission_refuses_a_diffusion_engine(self):
        async def scenario():
            core = FakeCore([])
            host = VLMHost(core)
            engine = type("Engine", (VLMNew, VLMHost), {})
            instance = engine.__new__(engine)
            instance.__dict__.update(host.__dict__)
            instance.is_diffusion_model = True
            await instance.admit_owned_chat(
                [{"role": "user", "content": "hi"}], admission=object(), request_id="claim")
            return core

        with self.assertRaises(RuntimeError):
            run(scenario())

    def test_owned_admission_never_starts_an_engine(self):
        core, instance, _ = run(drive_owned(BatchedNew, BatchedHost, SCENARIOS["prepared prompt"]))
        self.assertEqual(instance.started, 0)


# Part 2: the probe and the failure release on the real 25b harness.
core_tree = ast.parse((ROOT / "omlx/engine_core.py").read_text())
core_methods = methods_of(core_tree, "EngineCore", {"probe_owned_absence", "release_owned_failure"})
core_ns = dict(waiting_fixture.core_ns)
exec(compile(ast.Module(body=core_methods, type_ignores=[]), str(ROOT / "omlx/engine_core.py"), "exec"), core_ns)
sched_tree = ast.parse((ROOT / "omlx/scheduler.py").read_text())
probe_method = methods_of(sched_tree, "Scheduler", {"_probe_owned_presence"})
sched_ns = dict(waiting_fixture.namespace)
exec(compile(ast.Module(body=probe_method, type_ignores=[]), str(ROOT / "omlx/scheduler.py"), "exec"), sched_ns)


class ProbeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = waiting_fixture.WaitingCleanupTests()
        await self.fixture.asyncSetUp()
        self.core = self.fixture.core
        self.s = self.fixture.scheduler
        for name in ("probe_owned_absence", "release_owned_failure"):
            setattr(self.core, name, MethodType(core_ns[name], self.core))
        self.s._probe_owned_presence = MethodType(sched_ns["_probe_owned_presence"], self.s)

    async def asyncTearDown(self):
        await self.fixture.asyncTearDown()

    def handle(self, rid="synthetic-probe"):
        return OwnedAdmission(self.core, rid, time.monotonic() + 3)

    async def probe(self, owned, seconds=3):
        return await self.core.probe_owned_absence(owned, time.monotonic() + seconds)

    async def test_an_admitted_request_is_present_and_nothing_is_touched(self):
        owned = await self.fixture.admitted("synthetic-probe")
        self.assertEqual(await self.probe(owned), "present")
        self.assertEqual(owned._probe_result, "present")
        self.assertIsNone(owned._classified_state)
        self.assertIsNone(owned._cleanup_outcome)
        self.assertFalse(owned._cleanup_started)
        self.assertIn("synthetic-probe", self.s.requests)

    async def test_a_handle_that_never_submitted_is_absent(self):
        owned = self.handle()
        self.assertIsNone(owned.future)
        self.assertEqual(await self.probe(owned), "absent")

    async def test_a_removed_request_is_absent(self):
        owned = await self.fixture.admitted("synthetic-probe")
        del self.s.requests["synthetic-probe"]
        self.assertEqual(await self.probe(owned), "absent")

    async def test_a_store_future_means_finished_before_cleanup(self):
        owned = await self.fixture.admitted("synthetic-probe")
        self.s._inflight_store_futures["synthetic-probe"] = object()
        self.assertEqual(await self.probe(owned), "finished_before_cleanup")

    async def test_an_identity_mismatch_or_closed_core_is_unknown(self):
        owned = await self.fixture.admitted("synthetic-probe")
        self.s.requests["synthetic-probe"] = Request(
            request_id="synthetic-probe", prompt="other", sampling_params=SamplingParams())
        self.assertEqual(await self.probe(owned), "unknown")
        self.assertEqual(owned._probe_result, "unknown")
        owned2 = await self.fixture.admitted("synthetic-probe-2")
        self.core._closed = True
        try:
            self.assertEqual(await self.probe(owned2), "unknown")
        finally:
            self.core._closed = False

    async def test_invalid_or_past_deadlines_are_unknown(self):
        owned = await self.fixture.admitted("synthetic-probe")
        for deadline in (math.nan, math.inf, True, "soon"):
            with self.subTest(deadline=deadline):
                self.assertEqual(await self.core.probe_owned_absence(owned, deadline), "unknown")
        self.assertEqual(await self.core.probe_owned_absence(owned, time.monotonic() - 1), "unknown")

    async def test_the_probe_reruns_and_keeps_its_latest_run(self):
        owned = await self.fixture.admitted("synthetic-probe")
        await self.probe(owned)
        first = owned._probe_future
        del self.s.requests["synthetic-probe"]
        self.assertEqual(await self.probe(owned), "absent")
        self.assertIsNot(owned._probe_future, first)
        self.assertEqual(owned._probe_result, "absent")

    async def test_a_failed_admission_is_absent_released_and_barrier_ready(self):
        self.s.preflight_or_raise = Mock(side_effect=RuntimeError("synthetic preflight"))
        owned = self.handle("synthetic-refused")
        with self.assertRaises(RuntimeError):
            await self.core.add_request("synthetic", request_id="synthetic-refused", _owned_admission=owned)
        joined = await owned.join(self.core, time.monotonic() + 3)
        self.assertEqual((joined.insertion_attempted, joined.admission_failed), (True, True))
        self.assertEqual(await self.probe(owned), "absent")
        self.assertTrue(self.core.release_owned_failure(owned))
        self.core._cleanup_request.assert_called_once_with("synthetic-refused")
        # The probe never touched the cleanup fields, so the barrier runs.
        self.assertFalse(owned._cleanup_started)
        fact = await self.core._owned_stream_barrier(owned, time.monotonic() + 3)
        self.assertIsInstance(fact, StreamSynchronized)

    async def test_release_needs_a_proved_absence_and_a_finished_dispatch(self):
        owned = await self.fixture.admitted("synthetic-probe")
        self.assertFalse(self.core.release_owned_failure(owned))  # never probed
        self.assertEqual(await self.probe(owned), "present")
        self.assertFalse(self.core.release_owned_failure(owned))  # present
        never = self.handle("synthetic-never")
        self.assertEqual(await self.probe(never), "absent")
        self.assertTrue(self.core.release_owned_failure(never))  # nothing submitted
        self.core._cleanup_request.assert_called_once_with("synthetic-never")

    async def test_an_absence_taken_before_the_dispatch_never_releases_it(self):
        owned = self.handle("synthetic-admitted-later")
        self.assertEqual(await self.probe(owned), "absent")  # nothing submitted yet
        await self.core.add_request(
            "synthetic", request_id="synthetic-admitted-later", _owned_admission=owned)
        self.assertIn("synthetic-admitted-later", self.s.requests)
        # The admission returned: the stale result must never release it.
        self.assertFalse(self.core.release_owned_failure(owned))
        self.core._cleanup_request.assert_not_called()

    async def test_a_failed_admission_needs_a_probe_of_its_own_dispatch(self):
        owned = self.handle("synthetic-rebind")
        self.assertEqual(await self.probe(owned), "absent")  # before any dispatch
        self.s.preflight_or_raise = Mock(side_effect=RuntimeError("synthetic preflight"))
        with self.assertRaises(RuntimeError):
            await self.core.add_request("synthetic", request_id="synthetic-rebind", _owned_admission=owned)
        # The earlier result was taken before this dispatch existed.
        self.assertFalse(self.core.release_owned_failure(owned))
        self.assertEqual(await self.probe(owned), "absent")
        self.assertTrue(self.core.release_owned_failure(owned))
        self.core._cleanup_request.assert_called_once_with("synthetic-rebind")

    async def test_a_present_or_unknown_result_never_releases(self):
        self.s.preflight_or_raise = Mock(side_effect=RuntimeError("synthetic preflight"))
        owned = self.handle("synthetic-late-insert")
        with self.assertRaises(RuntimeError):
            await self.core.add_request("synthetic", request_id="synthetic-late-insert", _owned_admission=owned)
        # A late insertion the joined facts did not see: the probe of this very
        # dispatch says present, and only the result itself refuses release.
        self.s.requests["synthetic-late-insert"] = owned.request
        self.assertEqual(await self.probe(owned), "present")
        self.assertFalse(self.core.release_owned_failure(owned))
        self.s.requests["synthetic-late-insert"] = Request(
            request_id="synthetic-late-insert", prompt="other", sampling_params=SamplingParams())
        self.assertEqual(await self.probe(owned), "unknown")
        self.assertFalse(self.core.release_owned_failure(owned))
        self.core._cleanup_request.assert_not_called()

    async def test_an_admitted_request_that_finished_is_never_released(self):
        owned = await self.fixture.admitted("synthetic-finished")
        del self.s.requests["synthetic-finished"]  # finished and gone
        self.assertEqual(await self.probe(owned), "absent")
        # Its collector holds the final output for the reader.
        self.assertFalse(self.core.release_owned_failure(owned))
        self.core._cleanup_request.assert_not_called()

    async def test_overlapping_probes_keep_only_the_latest(self):
        owned = await self.fixture.admitted("synthetic-probe")
        self.assertEqual(await self.probe(owned), "present")
        gate = threading.Event()
        blocker = self.core._mlx_executor.submit(gate.wait, 5)
        submitted = owned._probe_future
        first = asyncio.create_task(self.probe(owned))
        while owned._probe_future is submitted:
            await asyncio.sleep(0.002)
        submitted = owned._probe_future
        second = asyncio.create_task(self.probe(owned))
        while owned._probe_future is submitted:
            await asyncio.sleep(0.002)
        # No stale result is readable while a run is in flight.
        self.assertIsNone(owned._probe_result)
        gate.set()
        blocker.result()
        self.assertEqual((await first, await second), ("unknown", "present"))
        self.assertEqual(owned._probe_result, "present")

    async def test_a_cancelled_probe_reports_unknown(self):
        owned = await self.fixture.admitted("synthetic-probe")
        gate = threading.Event()
        blocker = self.core._mlx_executor.submit(gate.wait, 5)
        submitted = owned._probe_future
        task = asyncio.create_task(self.probe(owned))
        while owned._probe_future is submitted:
            await asyncio.sleep(0.002)
        task.cancel()
        self.assertEqual(await task, "unknown")
        gate.set()
        blocker.result()

    async def test_a_probe_on_a_parallel_executor_is_unknown(self):
        # Ordering after the dispatch holds only on a serial executor.
        serial = self.core._mlx_executor
        parallel = ThreadPoolExecutor(2)
        self.core._mlx_executor = parallel
        try:
            owned = OwnedAdmission(self.core, "synthetic-parallel", time.monotonic() + 3)
            self.assertEqual(await self.probe(owned), "unknown")
            self.assertIsNone(owned._probe_future)
        finally:
            self.core._mlx_executor = serial
            parallel.shutdown(wait=False)

    async def test_release_refuses_on_a_closed_core(self):
        never = self.handle("synthetic-closed")
        self.assertEqual(await self.probe(never), "absent")
        self.core._closed = True
        try:
            self.assertFalse(self.core.release_owned_failure(never))
        finally:
            self.core._closed = False
        self.core._cleanup_request.assert_not_called()

    async def test_a_stale_absence_never_releases_a_dispatch_still_running(self):
        owned = self.handle("synthetic-late")
        self.assertEqual(await self.probe(owned), "absent")  # nothing submitted yet
        gate = threading.Event()
        blocker = self.core._mlx_executor.submit(gate.wait, 5)
        admit = asyncio.create_task(
            self.core.add_request("synthetic", request_id="synthetic-late", _owned_admission=owned))
        while owned.future is None:
            await asyncio.sleep(0.005)
        # The dispatch is queued behind the blocker: the stale result must
        # not release what this admission is about to use.
        self.assertFalse(self.core.release_owned_failure(owned))
        gate.set()
        await admit
        blocker.result()
        self.core._cleanup_request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
