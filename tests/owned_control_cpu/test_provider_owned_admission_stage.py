# SPDX-License-Identifier: Apache-2.0
"""Slice 27.4b CPU proof: the shared Responses preparation and the admission stage.

Part 1 proves create_response's preparation block moved verbatim into
_prepare_responses_chat (AST against the vendored upstream oMLX 0032173 route,
the base the fork is merged onto).
Part 2 runs the real admission stage on the 25b harness: the real
EngineCore.add_request owned branch, the real scheduler add_request, a real
single-worker executor, the real probe, release, barrier and owned cleanup,
and real output collectors. The engine's prompt and admission call are stubs
that drive the real core; the pool is a stub that leases one engine.
"""

import ast
import asyncio
import copy
import threading
import time
from types import MethodType, SimpleNamespace
import unittest
from unittest import mock
from unittest.mock import Mock

import test_provider_waiting_cleanup as waiting_fixture
import test_provider_owned_admission_plumbing as plumbing
from omlx import owned_preparation as prep
from omlx.api import prism_control_coordinator as coord
from omlx.api import prism_control_routes as routes
from omlx.api.responses_models import ResponsesRequest
from omlx.output_collector import RequestOutputCollector

ROOT = waiting_fixture.ROOT
admission_ns = waiting_fixture.stream_fixture.admission_fixture.ns
BASELINE = (plumbing.Path if hasattr(plumbing, "Path") else __import__("pathlib").Path)(
    __file__).with_name("omlx_create_response_0032173.py.txt")
RETURNED = (
    "messages", "chat_kwargs", "current_input_messages", "namespace_aliases",
    "response_format", "response_format_warning", "native_response_format",
    "native_reasoning", "tools_for_template", "temperature", "top_p",
)


def request(**fields):
    return ResponsesRequest.model_validate({"model": "m", "input": "hi", "max_output_tokens": 5} | fields)


class PreparationMoveTests(unittest.TestCase):
    def setUp(self):
        self.new = ast.parse((ROOT / "omlx" / ("ser" + "ver.py")).read_text())
        old_module = ast.parse(BASELINE.read_text())
        self.old_route = next(n for n in old_module.body if isinstance(n, ast.AsyncFunctionDef))
        self.new_route = next(n for n in self.new.body
                              if isinstance(n, ast.AsyncFunctionDef) and n.name == "create_response")

    @staticmethod
    def split(route):
        body = next(n for n in route.body if isinstance(n, ast.Try)).body
        resolved = next(i for i, s in enumerate(body)
                        if isinstance(s, ast.Assign) and ast.unparse(s.targets[0]) == "resolved_model")
        return body, resolved

    def test_the_block_moved_verbatim_with_one_substitution(self):
        old_body, resolved = self.split(self.old_route)
        last = next(i for i, s in enumerate(old_body)
                    if isinstance(s, ast.Assign)
                    and ast.unparse(s.targets[0]) == "chat_kwargs['preserve_reasoning']")
        old_block = old_body[resolved + 1:last + 1]
        helper = next(n for n in self.new.body
                      if isinstance(n, ast.AsyncFunctionDef) and n.name == "_prepare_responses_chat")
        self.assertEqual([a.arg for a in helper.args.args], ["engine", "request", "resolved_model"])
        self.assertEqual([a.arg for a in helper.args.kwonlyargs], ["mcp_manager"])
        body = helper.body[1:]  # drop the docstring
        returned = body[-1]
        self.assertIsInstance(returned, ast.Return)
        self.assertEqual(
            {k.arg: ast.unparse(k.value) for k in returned.value.keywords},
            {name: name for name in RETURNED},
        )

        class Restore(ast.NodeTransformer):
            def visit_Name(self, node):
                if node.id == "mcp_manager":
                    return ast.copy_location(ast.Attribute(
                        value=ast.Name(id="_server_state", ctx=ast.Load()),
                        attr="mcp_manager", ctx=node.ctx), node)
                return node

        restored = [Restore().visit(copy.deepcopy(s)) for s in body[:-1]]
        self.assertEqual(
            [ast.dump(s, include_attributes=False) for s in restored],
            [ast.dump(s, include_attributes=False) for s in old_block],
        )
        uses = sum(isinstance(n, ast.Name) and n.id == "mcp_manager"
                   for s in body for n in ast.walk(s))
        self.assertEqual(uses, 2)

    def test_the_route_is_unchanged_around_the_call(self):
        old_body, old_resolved = self.split(self.old_route)
        new_body, new_resolved = self.split(self.new_route)
        last = next(i for i, s in enumerate(old_body)
                    if isinstance(s, ast.Assign)
                    and ast.unparse(s.targets[0]) == "chat_kwargs['preserve_reasoning']")
        dump = lambda s: ast.dump(s, include_attributes=False)
        self.assertEqual([dump(s) for s in new_body[:new_resolved + 1]],
                         [dump(s) for s in old_body[:old_resolved + 1]])
        call = new_body[new_resolved + 1]
        self.assertEqual(
            ast.unparse(call),
            "prepared_chat = await _prepare_responses_chat(engine, request, resolved_model, "
            "mcp_manager=_server_state.mcp_manager)")
        unpack = new_body[new_resolved + 2:new_resolved + 2 + len(RETURNED)]
        self.assertEqual([ast.unparse(s) for s in unpack],
                         [f"{n} = prepared_chat.{n}" for n in RETURNED])
        self.assertEqual([dump(s) for s in new_body[new_resolved + 2 + len(RETURNED):]],
                         [dump(s) for s in old_body[last + 1:]])
        old_rest = [dump(s) for s in self.old_route.body if not isinstance(s, ast.Try)]
        new_rest = [dump(s) for s in self.new_route.body if not isinstance(s, ast.Try)]
        self.assertEqual(new_rest, old_rest)

    def test_the_result_class_needs_no_import(self):
        cls = next(n for n in self.new.body if isinstance(n, ast.ClassDef) and n.name == "_ResponsesChat")
        fields = [(ast.unparse(a.target), ast.unparse(a.annotation)) for a in cls.body if isinstance(a, ast.AnnAssign)]
        self.assertEqual(fields, [(n, "object") for n in RETURNED])


class RefusalTests(unittest.TestCase):
    def test_owned_refusals(self):
        self.assertIsNone(prep.owned_request_refusal(request()))
        self.assertIsNone(prep.owned_request_refusal(request(instructions="be brief")))
        self.assertEqual(prep.owned_request_refusal(request(), quantizing=True), "quantizing")
        self.assertEqual(prep.owned_request_refusal(request(tools=[
            {"type": "function", "name": "f", "parameters": {"type": "object"}}])), "tools")
        self.assertEqual(prep.owned_request_refusal(request(text={"format": {"type": "json_object"}})),
                         "text_format")
        self.assertEqual(prep.owned_request_refusal(request(input=[{"role": "user", "content": [
            {"type": "input_text", "text": "see"},
            {"type": "input_image", "image_url": "data:image/png;base64,AAAA"}]}])), "media_input")

    def test_a_lone_surrogate_is_refused_at_parse(self):
        with self.assertRaises(ValueError):
            routes.parse_control_request({"model": "m", "input": "\ud800", "max_output_tokens": 5})
        parsed, digest = routes.parse_control_request({"model": "m", "input": "ok", "max_output_tokens": 5})
        self.assertEqual(len(digest), 64)


class StubEngine:
    def __init__(self, core):
        self.core = core
        self.preflights = []
        self.admits = []
        self.admit_kwargs = []
        self.preflight_error = None
        self.before_admit = None
        self.after_admit_error = None
        self.on_preflight = None

    async def preflight_chat(self, messages, request_id=None, **kwargs):
        self.preflights.append(kwargs)
        if self.on_preflight:
            self.on_preflight()
        if self.preflight_error:
            raise self.preflight_error

    async def admit_owned_chat(self, messages, *, admission, request_id, **kwargs):
        self.admits.append(request_id)
        self.admit_kwargs.append(kwargs)
        self.last_admission = admission
        if self.before_admit:
            self.before_admit(admission)
        await self.core.add_request("synthetic", request_id=request_id, _owned_admission=admission)
        if self.after_admit_error:
            raise self.after_admit_error


class StubPool:
    def __init__(self, engine):
        self.engine = engine
        self.entry = SimpleNamespace(pending_unload_reason=None, abort_requested=False)
        self.acquired = 0
        self.releases = []

    async def acquire_owned(self, model_id, *, fence):
        await fence()
        self.acquired += 1
        return prep.OwnedLease(model_id, self.engine)

    async def release_engine(self, model_id):
        self.releases.append(model_id)

    def get_entry(self, model_id):
        return self.entry

    @staticmethod
    def _resolve_engine_core_from_engine(engine):
        return engine.core


class FakeExecution:
    def __init__(self, req, rid, deadline):
        self.model_id = "m"
        self.request = req
        self.claim = SimpleNamespace(request_id=rid)
        self.deadline = deadline
        self.cancellations = []
        self.lease = None
        self.drain_deadline = None
        self.cleanup_task = None
        self.final_output = None
        self.admitted = False
        self.cancel_requested = False
        self.stop_requested = False
        self.events = []
        self.fences = []
        self.verifies = 0
        self.verify_error_at = None
        self.on_verify = None

    def emit(self, event, payload=None):
        self.events.append((event, payload))

    def publish_fence(self, fence):
        self.fences.append(fence)

    async def verify(self):
        self.verifies += 1
        if self.on_verify:
            self.on_verify(self)
        if self.verify_error_at == self.verifies:
            raise RuntimeError("synthetic verify refusal")


async def prepare_chat(engine, req, model_id):
    return SimpleNamespace(messages=[{"role": "user", "content": "hi"}], chat_kwargs={"max_tokens": 5})


def out(text, finished=False):
    # The accumulated text differs from the delta, so emitting the wrong one shows.
    return SimpleNamespace(new_text=text, prompt_tokens=2, completion_tokens=1, finished=finished,
                           output_text="ACCUMULATED:" + text, error=None,
                           finish_reason="stop" if finished else None)


def err(finish_reason, error, text=""):
    return SimpleNamespace(new_text=text, prompt_tokens=2, completion_tokens=1, finished=True,
                           output_text=text, error=error, finish_reason=finish_reason)


COMPLETED = ("completed", ("finished_output", "absent_proved", "stream_synchronized"))
CANCELLED = ("cancelled", ("waiting_cleaned", "stream_synchronized"))


class AdmissionStageTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = waiting_fixture.WaitingCleanupTests()
        await self.fixture.asyncSetUp()
        self.core = self.fixture.core
        self.s = self.fixture.scheduler
        for name in ("probe_owned_absence", "release_owned_failure"):
            setattr(self.core, name, MethodType(plumbing.core_ns[name], self.core))
        self.s._probe_owned_presence = MethodType(plumbing.sched_ns["_probe_owned_presence"], self.s)
        self.saved_collector = admission_ns["RequestOutputCollector"]
        admission_ns["RequestOutputCollector"] = RequestOutputCollector
        self.engine = StubEngine(self.core)
        self.pool = StubPool(self.engine)
        self.stage = prep.make_admission_stage(lambda: self.pool, prepare_chat, prep.owned_request_refusal)

    async def asyncTearDown(self):
        admission_ns["RequestOutputCollector"] = self.saved_collector
        await self.fixture.asyncTearDown()

    def execution(self, req=None, rid="synthetic-owned", seconds=3):
        return FakeExecution(req or request(), rid, time.monotonic() + seconds)

    async def until(self, predicate, seconds=5):
        deadline = time.monotonic() + seconds
        while not predicate():
            if time.monotonic() >= deadline:
                raise AssertionError("condition not reached")
            await asyncio.sleep(0.005)

    def finish_in_scheduler(self, rid):
        # What the scheduler's step does once the request finished.
        request = self.s.requests.pop(rid)
        self.s.waiting.remove(request)

    async def feed(self, collector, *outputs):
        for output in outputs:
            await self.until(lambda: collector.output is None)
            collector.put(output)

    async def test_unsupported_requests_never_reach_the_pool(self):
        for req in (request(tools=[{"type": "function", "name": "f", "parameters": {"type": "object"}}]),
                    request(text={"format": {"type": "json_object"}})):
            with self.assertRaises(coord.EngineStageUnavailable):
                await asyncio.wait_for(self.stage(self.execution(req)), 5)
        self.assertEqual(self.pool.acquired, 0)

    async def test_a_preflight_refusal_is_not_admitted_and_releases_the_lease(self):
        self.engine.preflight_error = RuntimeError("synthetic prefill memory")
        with self.assertRaises(coord.EngineStageUnavailable):
            await self.stage(self.execution())
        self.assertEqual((self.engine.admits, self.pool.releases), ([], ["m"]))
        self.assertEqual(self.core._output_collectors, {})

    async def test_unload_requests_are_checked_before_and_after_the_preflight_and_before_admission(self):
        cases = {
            "before preflight": lambda ex: setattr(self.pool.entry, "pending_unload_reason", "manual unload"),
            "after preflight": None,
            "before admission": None,
        }
        for name in cases:
            with self.subTest(name):
                self.pool.entry.pending_unload_reason = None
                self.pool.entry.abort_requested = False
                self.engine.preflights.clear()
                self.engine.admits.clear()
                self.engine.on_preflight = None
                ex = self.execution(rid=f"synthetic-{name.replace(' ', '-')}")
                if name == "before preflight":
                    self.pool.entry.pending_unload_reason = "manual unload"
                elif name == "after preflight":
                    self.engine.on_preflight = lambda: setattr(self.pool.entry, "abort_requested", True)
                else:
                    def on_verify(execution):
                        if execution.verifies == 2:
                            self.pool.entry.abort_requested = True
                    ex.on_verify = on_verify
                with self.assertRaises(coord.EngineStageUnavailable):
                    await asyncio.wait_for(self.stage(ex), 5)
                self.assertEqual(self.engine.admits, [])
                if name == "before preflight":
                    self.assertEqual(self.engine.preflights, [])
                if name == "after preflight":
                    # Refused before the final verify: only preparation's fence ran.
                    self.assertEqual(ex.verifies, 1)

    async def test_the_fence_is_published_before_a_refused_final_verify(self):
        ex = self.execution()
        ex.verify_error_at = 2  # the first verify is preparation's fence
        seen = []
        ex.on_verify = lambda e: seen.append(len(e.fences))
        with self.assertRaises(coord.EngineStageUnavailable):
            await self.stage(ex)
        self.assertEqual(seen, [0, 1])
        self.assertEqual((self.engine.admits, self.pool.releases), ([], ["m"]))

    async def test_an_admitted_request_streams_then_completes_once_absent(self):
        ex = self.execution()
        task = asyncio.create_task(self.stage(ex))
        await self.until(lambda: ex.admitted)
        collector = self.core._output_collectors["synthetic-owned"]
        await self.feed(collector, out("Hel"), out("lo", finished=True))
        await self.until(lambda: ex.final_output is not None)
        await asyncio.sleep(0.1)  # still present: termination waits
        self.assertFalse(task.done())
        self.finish_in_scheduler("synthetic-owned")
        self.assertEqual(await asyncio.wait_for(task, 5), COMPLETED)
        self.assertEqual(ex.events, [
            ("delta", {"text": "Hel", "prompt_tokens": 2, "completion_tokens": 1}),
            ("delta", {"text": "lo", "prompt_tokens": 2, "completion_tokens": 1}),
            ("finished", {"finish_reason": "stop", "prompt_tokens": 2, "completion_tokens": 1}),
        ])
        self.assertTrue(self.engine.last_admission._stream_started)
        self.assertEqual(self.pool.releases, ["m"])
        # The owned consumer has drained the final output and all stop proofs.
        # Retirement makes this classifier evictable before the next load.
        self.core._cleanup_request.assert_called_once_with("synthetic-owned")
        self.assertNotIn("synthetic-owned", self.core._output_collectors)

    async def test_only_a_none_retention_skips_the_cache_store(self):
        # A classifier request is unique and never reused; storing its cache
        # wrote about 60 MB to SSD per call. Only "none" opts out.
        for n, (fields, skipped) in enumerate((({}, None), ({"prompt_cache_retention": "24h"}, None),
                                               ({"prompt_cache_retention": "none"}, True))):
            rid = f"synthetic-retention-{n}"
            ex = self.execution(request(**fields), rid=rid)
            task = asyncio.create_task(self.stage(ex))
            await self.until(lambda: ex.admitted)
            await self.feed(self.core._output_collectors[rid], out("ok", finished=True))
            await self.until(lambda: ex.final_output is not None)
            self.finish_in_scheduler(rid)
            self.assertEqual(await asyncio.wait_for(task, 5), COMPLETED)
            self.assertEqual(self.engine.admit_kwargs[-1].get("skip_cache_store"), skipped, fields)
            # Preflight sees the same arguments the admission does.
            self.assertEqual(self.engine.preflights[-1].get("skip_cache_store"), skipped, fields)

    async def test_a_scheduler_refusal_is_admission_refused_with_a_barrier(self):
        self.s.preflight_or_raise = Mock(side_effect=RuntimeError("synthetic scheduler preflight"))
        ex = self.execution(rid="synthetic-refused")
        outcome = await self.stage(ex)
        self.assertEqual(outcome, ("not_started", ("admission_refused", "stream_synchronized")))
        # The barrier really ran on the refused handle.
        self.assertTrue(self.engine.last_admission._stream_started)
        self.core._cleanup_request.assert_called_once_with("synthetic-refused")
        self.assertFalse(ex.admitted)
        self.assertEqual(self.pool.releases, ["m"])

    async def test_a_handle_cancellation_is_not_admitted_and_the_task_lives(self):
        self.engine.before_admit = lambda handle: handle.cancel()
        ex = self.execution(rid="synthetic-cancelled")
        with self.assertRaises(coord.EngineStageUnavailable):
            await self.stage(ex)
        self.assertEqual(ex.cancellations, [])
        self.assertEqual(asyncio.current_task().cancelling(), 0)
        self.assertNotIn("synthetic-cancelled", self.s.requests)
        self.core._cleanup_request.assert_called_once_with("synthetic-cancelled")

    async def test_a_task_cancellation_during_admission_is_counted(self):
        gate = threading.Event()
        blocker = self.core._mlx_executor.submit(gate.wait, 5)
        ex = self.execution(rid="synthetic-task-cancel")
        task = asyncio.create_task(self.stage(ex))
        await self.until(lambda: self.engine.admits)
        await asyncio.sleep(0.02)
        task.cancel()
        await asyncio.sleep(0.02)
        gate.set()
        blocker.result()
        with self.assertRaises(coord.EngineStageUnavailable):
            await task
        self.assertEqual(len(ex.cancellations), 1)
        self.assertNotIn("synthetic-task-cancel", self.s.requests)

    async def test_an_unjoinable_admission_publishes_nothing(self):
        gate = threading.Event()
        blocker = self.core._mlx_executor.submit(gate.wait, 5)
        ex = self.execution(rid="synthetic-unjoinable", seconds=0.3)
        with mock.patch.object(prep, "DRAIN_WINDOW_SECONDS", 0.2):
            with self.assertRaises(prep.ExecutionUnresolved) as caught:
                await self.stage(ex)
        self.assertEqual(str(caught.exception), "admission_unjoinable")
        self.assertEqual(self.pool.releases, ["m"])
        gate.set()
        blocker.result()

    async def test_a_late_admission_is_treated_as_admitted(self):
        self.engine.after_admit_error = RuntimeError("synthetic late failure")
        ex = self.execution(rid="synthetic-late")
        task = asyncio.create_task(self.stage(ex))
        await self.until(lambda: ex.admitted)
        await self.feed(self.core._output_collectors["synthetic-late"], out("x", finished=True))
        await self.until(lambda: ex.final_output is not None)
        self.finish_in_scheduler("synthetic-late")
        self.assertEqual(await asyncio.wait_for(task, 5), COMPLETED)
        self.assertEqual([e[0] for e in ex.events], ["delta", "finished"])

    async def test_a_fence_after_admission_starts_one_cleanup_and_stops_the_reader(self):
        ex = self.execution(rid="synthetic-fenced")
        task = asyncio.create_task(self.stage(ex))
        await self.until(lambda: ex.admitted)
        fence = ex.fences[-1]
        fence()
        first = ex.cleanup_task
        fence()
        self.assertIs(ex.cleanup_task, first)
        self.assertEqual(await asyncio.wait_for(task, 5), CANCELLED)
        self.assertNotIn("synthetic-fenced", self.s.requests)
        # The cleaned request's collector is released by termination.
        self.core._cleanup_request.assert_called_once_with("synthetic-fenced")

    async def test_a_cancel_during_admission_starts_the_cleanup_after_it(self):
        ex = self.execution(rid="synthetic-cancel-mark")
        self.engine.before_admit = lambda handle: setattr(ex, "cancel_requested", True)
        task = asyncio.create_task(self.stage(ex))
        self.assertEqual(await asyncio.wait_for(task, 5), CANCELLED)
        self.assertTrue(ex.admitted)
        self.assertIsNotNone(ex.cleanup_task)
        self.assertNotIn("synthetic-cancel-mark", self.s.requests)

    async def test_the_reader_stops_when_the_core_closes(self):
        ex = self.execution(rid="synthetic-closing")
        task = asyncio.create_task(self.stage(ex))
        await self.until(lambda: ex.admitted)
        self.core._closing = True
        try:
            with self.assertRaises(prep.ExecutionUnresolved):
                await asyncio.wait_for(task, 5)
        finally:
            self.core._closing = False


    async def test_a_fence_after_the_admission_window_still_retires_the_request(self):
        # 27.4b code review MAJOR-1: the cleanup's window starts with the cleanup.
        # The window must also fit the cleanup and barrier under full-suite load:
        # at 0.2 s the terminal barrier intermittently ran out (barrier_refused).
        with mock.patch.object(prep, "DRAIN_WINDOW_SECONDS", 1.0):
            ex = self.execution(rid="synthetic-fenced-late")
            task = asyncio.create_task(self.stage(ex))
            await self.until(lambda: ex.admitted)
            await asyncio.sleep(1.1)  # the admission's own window is over
            ex.fences[-1]()
            fact = await asyncio.wait_for(ex.cleanup_task, 5)
            self.assertEqual(await asyncio.wait_for(task, 5), CANCELLED)
        self.assertIn(type(fact).__name__, ("WaitingCleaned", "RunningCleaned"))
        self.assertNotIn("synthetic-fenced-late", self.s.requests)

    async def test_a_cleanup_that_ends_before_its_job_stops_the_reader(self):
        async def refused(handle, deadline):
            return None
        self.core._owned_cleanup_waiting = refused
        ex = self.execution(rid="synthetic-no-job")
        task = asyncio.create_task(self.stage(ex))
        await self.until(lambda: ex.admitted)
        ex.fences[-1]()
        with self.assertRaises(prep.ExecutionUnresolved):
            await asyncio.wait_for(task, 2)
        self.assertIsNone(self.engine.last_admission._cleanup_future)
        self.assertEqual(ex.events, [])

    async def test_a_shutdown_long_after_admission_reads_for_its_own_window(self):
        with mock.patch.object(prep, "DRAIN_WINDOW_SECONDS", 0.3):
            ex = self.execution(rid="synthetic-shutdown")
            task = asyncio.create_task(self.stage(ex))
            await self.until(lambda: ex.admitted)
            await asyncio.sleep(0.4)  # the admission's own window is over
            ex.stop_requested = True
            self.core._output_collectors["synthetic-shutdown"].put(out("late"))
            await self.until(lambda: ex.events)
            self.assertFalse(task.done())
            # The shutdown window ends the reader without a finished output.
            with self.assertRaises(prep.ExecutionUnresolved) as caught:
                await asyncio.wait_for(task, 2)
        self.assertEqual(str(caught.exception), "no_final_output")
        self.assertEqual(ex.events[0][1]["text"], "late")
        self.assertEqual(self.pool.releases, ["m"])

    async def test_cancellations_before_admission_are_counted_and_not_admitted(self):
        for where in ("preparation", "final verify"):
            with self.subTest(where):
                self.pool.releases.clear()
                self.engine.admits.clear()
                ex = self.execution(rid=f"synthetic-{where.replace(' ', '-')}-cancel")
                if where == "preparation":
                    self.engine.preflight_error = asyncio.CancelledError()
                else:
                    self.engine.preflight_error = None

                    def on_verify(execution):
                        if execution.verifies == 2:
                            raise asyncio.CancelledError()
                    ex.on_verify = on_verify
                with self.assertRaises(coord.EngineStageUnavailable):
                    await asyncio.wait_for(self.stage(ex), 5)
                self.assertEqual(ex.cancellations, [None])
                self.assertEqual((self.engine.admits, self.pool.releases), ([], ["m"]))

    async def test_a_request_id_in_use_is_not_admitted_and_releases_nothing(self):
        # F4: the dispatch refuses before any attempt. The probe cannot prove the
        # id absent (another request holds it), and not_admitted is still true.
        other = SimpleNamespace(request_id="synthetic-in-use")
        self.s.requests["synthetic-in-use"] = other
        ex = self.execution(rid="synthetic-in-use")
        with self.assertRaises(coord.EngineStageUnavailable):
            await asyncio.wait_for(self.stage(ex), 5)
        self.assertEqual(self.engine.last_admission._probe_result, "unknown")
        self.assertFalse(self.engine.last_admission._attempted)
        self.core._cleanup_request.assert_not_called()
        self.assertIs(self.s.requests["synthetic-in-use"], other)
        self.assertEqual(self.pool.releases, ["m"])

    def blocking_insert(self):
        gate = threading.Event()

        def preflight(*args, **kwargs):
            gate.wait(5)
        self.s.preflight_or_raise = preflight
        return gate

    async def test_an_insert_that_outlives_the_admission_deadline_is_admitted(self):
        # admit() raises at its deadline while the dispatch is still inside
        # add_request; the join, not the exception, decides.
        gate = self.blocking_insert()
        ex = self.execution(rid="synthetic-truly-late", seconds=0.2)
        task = asyncio.create_task(self.stage(ex))
        await self.until(lambda: self.engine.admits)
        handle = self.engine.last_admission
        await self.until(lambda: handle._cancelled)  # admit() gave up
        self.assertFalse(handle._returned)
        gate.set()
        # Admitted from the join, and the past admission deadline does not
        # stop it: it streams and completes.
        await self.until(lambda: ex.admitted)
        self.assertTrue(handle._returned)
        self.assertIn("synthetic-truly-late", self.s.requests)
        await self.feed(self.core._output_collectors["synthetic-truly-late"], out("ok", finished=True))
        await self.until(lambda: ex.final_output is not None)
        self.finish_in_scheduler("synthetic-truly-late")
        self.assertEqual(await asyncio.wait_for(task, 5), COMPLETED)
        self.assertIsNone(ex.cleanup_task)

    async def test_a_cancellation_inside_the_join_joins_once_more(self):
        gate = self.blocking_insert()
        joins = []

        def before_admit(handle):
            original = handle.join

            async def join(core, deadline):
                joins.append(deadline)
                return await original(core, deadline)
            handle.join = join
        self.engine.before_admit = before_admit
        ex = self.execution(rid="synthetic-rejoin", seconds=0.2)
        task = asyncio.create_task(self.stage(ex))
        await self.until(lambda: joins)
        await asyncio.sleep(0.02)
        task.cancel()
        await self.until(lambda: len(joins) == 2)
        gate.set()
        await self.until(lambda: ex.admitted)
        self.assertEqual(ex.cancellations, [None])
        await self.feed(self.core._output_collectors["synthetic-rejoin"], out("ok", finished=True))
        await self.until(lambda: ex.final_output is not None)
        self.finish_in_scheduler("synthetic-rejoin")
        self.assertEqual(await asyncio.wait_for(task, 5), COMPLETED)

    async def test_an_unknown_probe_after_a_failed_admission_publishes_nothing(self):
        self.s.preflight_or_raise = Mock(side_effect=RuntimeError("synthetic scheduler preflight"))

        async def unknown(handle, deadline):
            return "unknown"
        self.core.probe_owned_absence = unknown
        ex = self.execution(rid="synthetic-probe-unknown")
        with self.assertRaises(prep.ExecutionUnresolved) as caught:
            await asyncio.wait_for(self.stage(ex), 5)
        self.assertEqual(str(caught.exception), "probe_unknown")
        self.core._cleanup_request.assert_not_called()
        self.assertFalse(self.engine.last_admission._stream_started)
        self.assertFalse(ex.admitted)
        self.assertEqual(self.pool.releases, ["m"])


    # 27.5 termination.
    async def test_error_outputs_publish_failed_and_never_stream(self):
        shapes = {
            "abort_request": err("abort", "Request aborted"),
            "abort_all_requests": err("error", "memory limit", "\n\n[Error: memory limit]"),
            "engine loop": err("error", "loop failed"),
            "error finish without a message": err("error", None),
        }
        for index, (name, output) in enumerate(shapes.items()):
            with self.subTest(name):
                rid = f"synthetic-error-{index}"
                ex = self.execution(rid=rid)
                task = asyncio.create_task(self.stage(ex))
                await self.until(lambda: ex.admitted)
                await self.feed(self.core._output_collectors[rid], out("partial"), output)
                await self.until(lambda: ex.final_output is not None)
                self.finish_in_scheduler(rid)
                self.assertEqual(await asyncio.wait_for(task, 5),
                                 ("failed", ("error_output", "absent_proved", "stream_synchronized")))
                self.assertEqual(ex.events, [
                    ("delta", {"text": "partial", "prompt_tokens": 2, "completion_tokens": 1})])

    async def test_a_cancel_after_the_final_output_starts_no_cleanup(self):
        ex = self.execution(rid="synthetic-cancel-after")
        task = asyncio.create_task(self.stage(ex))
        await self.until(lambda: ex.admitted)
        await self.feed(self.core._output_collectors["synthetic-cancel-after"], out("x", finished=True))
        await self.until(lambda: ex.final_output is not None)
        ex.fences[-1]()
        self.assertIsNone(ex.cleanup_task)
        self.finish_in_scheduler("synthetic-cancel-after")
        self.assertEqual(await asyncio.wait_for(task, 5), COMPLETED)

    async def test_the_admission_deadline_does_not_bound_generation(self):
        # Generation ends at its output cap or when Prism cancels it.
        ex = self.execution(rid="synthetic-deadline", seconds=0.3)
        task = asyncio.create_task(self.stage(ex))
        await self.until(lambda: ex.admitted)
        await asyncio.sleep(0.6)  # well past the admission deadline
        self.assertIsNone(ex.cleanup_task)
        self.assertFalse(task.done())
        await self.feed(self.core._output_collectors["synthetic-deadline"], out("late", finished=True))
        await self.until(lambda: ex.final_output is not None)
        self.finish_in_scheduler("synthetic-deadline")
        self.assertEqual(await asyncio.wait_for(task, 5), COMPLETED)
        self.assertEqual(ex.events[0][1]["text"], "late")

    async def test_a_running_cleaned_fact_is_labelled_running(self):
        from omlx.owned_admission import RunningCleaned as running

        async def cleaned(handle, deadline):
            return running(handle, handle._request, handle.binding.generation_id,
                           handle.binding.request_id, "decoding")
        self.core._owned_cleanup_waiting = cleaned
        ex = self.execution(rid="synthetic-running")
        task = asyncio.create_task(self.stage(ex))
        await self.until(lambda: ex.admitted)
        ex.fences[-1]()
        self.assertEqual(await asyncio.wait_for(task, 5),
                         ("cancelled", ("running_cleaned", "stream_synchronized")))
        self.core._cleanup_request.assert_called_once_with("synthetic-running")

    async def test_absence_never_proved_publishes_nothing(self):
        with mock.patch.object(prep, "DRAIN_WINDOW_SECONDS", 0.2):
            ex = self.execution(rid="synthetic-present")
            task = asyncio.create_task(self.stage(ex))
            await self.until(lambda: ex.admitted)
            await self.feed(self.core._output_collectors["synthetic-present"], out("x", finished=True))
            with self.assertRaises(prep.ExecutionUnresolved) as caught:
                await asyncio.wait_for(task, 5)
        self.assertEqual(str(caught.exception), "absence_unproved")
        self.assertFalse(self.engine.last_admission._stream_started)
        self.assertEqual(self.pool.releases, ["m"])

    async def test_an_unknown_probe_after_the_output_publishes_nothing(self):
        ex = self.execution(rid="synthetic-unknown")
        task = asyncio.create_task(self.stage(ex))
        await self.until(lambda: ex.admitted)

        async def unknown(handle, deadline):
            return "unknown"
        self.core.probe_owned_absence = unknown
        await self.feed(self.core._output_collectors["synthetic-unknown"], out("x", finished=True))
        with self.assertRaises(prep.ExecutionUnresolved) as caught:
            await asyncio.wait_for(task, 5)
        self.assertEqual(str(caught.exception), "probe_unknown")
        self.assertFalse(self.engine.last_admission._stream_started)

    async def test_the_barrier_runs_once_after_absence(self):
        calls = []
        barrier = self.core._owned_stream_barrier

        async def counted(handle, deadline):
            calls.append(handle._probe_result)
            return await barrier(handle, deadline)
        self.core._owned_stream_barrier = counted
        ex = self.execution(rid="synthetic-once")
        task = asyncio.create_task(self.stage(ex))
        await self.until(lambda: ex.admitted)
        await self.feed(self.core._output_collectors["synthetic-once"], out("x", finished=True))
        await self.until(lambda: ex.final_output is not None)
        self.finish_in_scheduler("synthetic-once")
        self.assertEqual(await asyncio.wait_for(task, 5), COMPLETED)
        self.assertEqual(calls, ["absent"])

    async def test_a_cache_store_in_flight_keeps_termination_waiting(self):
        ex = self.execution(rid="synthetic-storing")
        task = asyncio.create_task(self.stage(ex))
        await self.until(lambda: ex.admitted)
        await self.feed(self.core._output_collectors["synthetic-storing"], out("x", finished=True))
        await self.until(lambda: ex.final_output is not None)
        # Finished, but its cache store still runs: the probe reads finished_before_cleanup.
        self.s._inflight_store_futures["synthetic-storing"] = object()
        await asyncio.sleep(0.15)
        self.assertFalse(task.done())
        del self.s._inflight_store_futures["synthetic-storing"]
        self.finish_in_scheduler("synthetic-storing")
        self.assertEqual(await asyncio.wait_for(task, 5), COMPLETED)

    async def test_a_refused_barrier_publishes_nothing(self):
        async def refused(handle, deadline):
            return None
        for path in ("completed", "cancelled"):
            with self.subTest(path):
                rid = f"synthetic-barrier-{path}"
                self.core._owned_stream_barrier = refused
                ex = self.execution(rid=rid)
                task = asyncio.create_task(self.stage(ex))
                await self.until(lambda: ex.admitted)
                if path == "completed":
                    await self.feed(self.core._output_collectors[rid], out("x", finished=True))
                    await self.until(lambda: ex.final_output is not None)
                    self.finish_in_scheduler(rid)
                else:
                    ex.fences[-1]()
                with self.assertRaises(prep.ExecutionUnresolved) as caught:
                    await asyncio.wait_for(task, 5)
                self.assertEqual(str(caught.exception), "barrier_refused")

    async def test_finish_gets_a_window_from_termination(self):
        with mock.patch.object(prep, "DRAIN_WINDOW_SECONDS", 0.3):
            ex = self.execution(rid="synthetic-window")
            task = asyncio.create_task(self.stage(ex))
            await self.until(lambda: ex.admitted)
            await asyncio.sleep(0.4)  # the admission's window has passed
            stopped = time.monotonic()
            await self.feed(self.core._output_collectors["synthetic-window"], out("x", finished=True))
            await self.until(lambda: ex.final_output is not None)
            self.finish_in_scheduler("synthetic-window")
            self.assertEqual(await asyncio.wait_for(task, 5), COMPLETED)
        self.assertGreaterEqual(ex.drain_deadline, stopped + 0.3)


class RouteEventTests(unittest.TestCase):
    def test_the_execute_stream_forwards_output_events(self):
        import test_provider_control_execute as execute_tests

        async def stage(execution):
            execution.emit("delta", {"text": "hi", "prompt_tokens": 1, "completion_tokens": 1})
            raise coord.EngineStageUnavailable()

        case = execute_tests.ExecuteTests("test_execute_ends_not_started_with_a_durable_terminal")
        try:
            h = case.make(stage=stage)

            async def scenario():
                await case.register(h)
                return await execute_tests.asgi(h.app, "POST", execute_tests.path(suffix="/execute"),
                                                execute_tests.execute_body())

            result = case.run_async(scenario())
            self.assertEqual([name for name, _ in result.events()], ["claimed", "delta", "terminal"])
            self.assertEqual(result.events()[1][1]["text"], "hi")
        finally:
            case.doCleanups()

    def test_streaming_continues_through_a_client_disconnect(self):
        import test_provider_control_execute as execute_tests

        gate = asyncio.Event()
        emitted = []

        async def stage(execution):
            execution.emit("delta", {"text": "a", "prompt_tokens": 1, "completion_tokens": 1})
            emitted.append("a")
            await gate.wait()
            execution.emit("delta", {"text": "b", "prompt_tokens": 1, "completion_tokens": 2})
            emitted.append("b")
            raise coord.EngineStageUnavailable()

        case = execute_tests.ExecuteTests("test_execute_ends_not_started_with_a_durable_terminal")
        try:
            h = case.make(stage=stage)

            async def scenario():
                await case.register(h)
                disconnect = asyncio.Event()

                def on_send(message):
                    if message["type"] == "http.response.body" and b"delta" in message.get("body", b""):
                        disconnect.set()

                result = await asyncio.wait_for(execute_tests.asgi(
                    h.app, "POST", execute_tests.path(suffix="/execute"), execute_tests.execute_body(),
                    disconnect=disconnect, on_send=on_send), 5)
                execution = h.coordinator._executions[execute_tests.identity()]
                self.assertFalse(execution.task.done())
                gate.set()
                await asyncio.wait_for(asyncio.shield(execution.task), 5)
                return result, execution

            result, execution = case.run_async(scenario())
            self.assertIn(b"delta", result.body)
            self.assertNotIn(b"terminal", result.body)
            self.assertEqual(emitted, ["a", "b"])
            self.assertFalse(execution.task.cancelled())
            self.assertEqual(case.status(h).state, "not_started")
        finally:
            case.doCleanups()


if __name__ == "__main__":
    unittest.main()
