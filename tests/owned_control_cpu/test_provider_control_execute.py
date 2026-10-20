# SPDX-License-Identifier: Apache-2.0
"""CPU-only slice 27.2 tests: execute route, coordinator, ordered cancel, drain.

Never imports the inference server or mlx. Every request is driven through the
app as a raw ASGI callable on one event loop per test, because TestClient runs
the app to completion before it returns a body and so cannot disconnect a
stream midway.
"""

import ast
import asyncio
from hashlib import sha256
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(os.environ["OMLX_CONTROL_ROOT"])


class ForbidEngines(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mlx" or fullname.startswith("mlx.") or fullname == "omlx.server":
            raise RuntimeError("engine_import_forbidden")


sys.meta_path.insert(0, ForbidEngines())
sys.path.insert(0, str(ROOT))
from fastapi import FastAPI, HTTPException
from omlx.api import prism_control_coordinator as coord
from omlx.api import prism_control_routes as routes
from omlx.api.prism_control_store import (
    ControlConflict,
    ControlStore,
    ControlStoreError,
)

# Keep the CPU no-engine boundary: load the real network utility without
# executing utils/__init__.py, which imports MLX hardware readers.
from types import ModuleType
utils_package = ModuleType("omlx.utils")
utils_package.__path__ = [str(ROOT / "omlx/utils")]
sys.modules.setdefault("omlx.utils", utils_package)
network_spec = importlib.util.spec_from_file_location(
    "omlx.utils.network", ROOT / "omlx/utils/network.py"
)
network_module = importlib.util.module_from_spec(network_spec)
sys.modules[network_spec.name] = network_module
network_spec.loader.exec_module(network_module)

spec = importlib.util.spec_from_file_location("omlx.admin.tested_execute_auth", ROOT / "omlx/admin/auth.py")
auth = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = auth
spec.loader.exec_module(auth)
spec = importlib.util.spec_from_file_location(
    "tested_execute_network", ROOT / "omlx/utils/network.py"
)
network = importlib.util.module_from_spec(spec)
spec.loader.exec_module(network)
is_loopback_bind = network.is_loopback_bind

KEY = "synthetic-main-control-key"
SUB = "synthetic-sub-control-key"
JOB = "11111111-1111-4111-8111-111111111111"
OP = "22222222-2222-4222-8222-222222222222"
OP2 = "33333333-3333-4333-8333-333333333333"
OWNER = sha256(KEY.encode()).hexdigest()
REQUEST = {"model": "synthetic-model", "input": "hello", "max_output_tokens": 16}
SETTINGS = SimpleNamespace(
    auth=SimpleNamespace(
        sub_keys=[SimpleNamespace(key=SUB)],
        skip_api_key_verification=True,
        allow_unauthenticated_inference=True,
    )
)


def path(op=OP, suffix=""):
    return f"/prism-control/v1/jobs/{JOB}/operations/{op}{suffix}"


def identity(op=OP):
    return (OWNER, JOB, op)


class Snapshot:
    """The settings snapshot callback; tests change it to change the digest."""

    def __init__(self):
        self.version = 1
        self.fail = False

    def __call__(self, model_id):
        if self.fail:
            raise HTTPException(status_code=503, detail="Control settings unavailable")
        return {"snapshot_version": 1, "model": model_id, "settings": self.version}


class StoreProxy:
    """Counts store calls; a hook replaces one call, receiving the real method."""

    HOOKED = ("claim", "verify_claim", "finish", "request_cancel", "lookup", "register")

    def __init__(self, store):
        self._store = store
        self.calls = []
        self.hooks = {}

    def __getattr__(self, name):
        target = getattr(self._store, name)
        if name not in self.HOOKED:
            return target

        def call(*args, **kwargs):
            self.calls.append(name)
            hook = self.hooks.get(name)
            if hook is not None:
                return hook(target, *args, **kwargs)
            return target(*args, **kwargs)

        return call


class Result(SimpleNamespace):
    def events(self):
        out = []
        for block in self.body.decode().strip().split("\n\n"):
            fields = dict(line.split(": ", 1) for line in block.split("\n"))
            out.append((fields["event"], json.loads(fields["data"])))
        return out

    def json(self):
        return json.loads(self.body)


async def asgi(app, method, url, payload=None, *, raw=None, key=KEY, disconnect=None,
               on_send=None, body_delay=None, reads=None):
    """One request through the app; `disconnect` gates http.disconnect."""
    body = raw if raw is not None else (b"" if payload is None else json.dumps(payload).encode())
    delivered = False
    sent = []

    async def receive():
        nonlocal delivered
        if reads is not None:
            reads.append(None)
        if not delivered:
            delivered = True
            if body_delay:
                await asyncio.sleep(body_delay)
            return {"type": "http.request", "body": body, "more_body": False}
        await (disconnect.wait() if disconnect is not None else asyncio.Event().wait())
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)
        if on_send is not None:
            on_send(message)

    headers = [(b"host", b"testserver"), (b"content-type", b"application/json")]
    if key is not None:
        headers.append((b"authorization", f"Bearer {key}".encode()))
    scope = {
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1", "method": method, "scheme": "http", "path": url,
        "raw_path": url.encode(), "query_string": b"", "root_path": "",
        "headers": headers, "client": ("127.0.0.1", 50000),
        "server": ("testserver", 80), "state": {},
    }
    await app(scope, receive, send)
    start = next((m for m in sent if m["type"] == "http.response.start"), None)
    chunks = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return Result(status=None if start is None else start["status"], body=chunks)


def execute_body(request=REQUEST, timeout_ms=5000):
    return {"request": request, "admission_timeout_ms": timeout_ms}


async def until(predicate, seconds=5):
    deadline = time.monotonic() + seconds
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.005)


class ExecuteTests(unittest.TestCase):
    def make(self, base=None, stage=None):
        if base is None:
            temp = tempfile.TemporaryDirectory(prefix="prism-control-execute-")
            self.addCleanup(temp.cleanup)
            base = Path(temp.name)
        snapshot = Snapshot()
        state = SimpleNamespace(
            api_key=KEY, bind_host="127.0.0.1", global_settings=SETTINGS,
            prism_control_store=None,
        )
        routes.configure_control_store(state, base, is_loopback_bind)
        proxy = StoreProxy(state.prism_control_store)
        state.prism_control_store = proxy
        routes.configure_control_coordinator(state, snapshot, stage)
        coordinator = state.prism_control_coordinator
        if coordinator is not None:
            self.addCleanup(os.close, coordinator.lock_descriptor)
        app = FastAPI()
        app.include_router(
            routes.create_control_router(
                lambda: state, auth.verify_any_api_key, is_loopback_bind, snapshot
            )
        )
        return SimpleNamespace(
            state=state, store=proxy, coordinator=coordinator, snapshot=snapshot,
            app=app, base=base, directory=base / "prism-control",
        )

    def run_async(self, coroutine):
        # A bound, so a broken ordering fails instead of hanging the suite.
        return asyncio.run(asyncio.wait_for(coroutine, 30))

    async def register(self, h, op=OP, request=REQUEST):
        result = await asyncio.wait_for(asgi(h.app, "PUT", path(op), {"request": request}), 5)
        self.assertEqual(result.status, 200, result.body)
        return result.json()

    def status(self, h, op=OP):
        # A fresh store instance reads the file as a restarted process would.
        return ControlStore(h.directory).lookup_status(OWNER, JOB, op)

    def live_status(self, h, op=OP):
        return h.store.lookup_status(OWNER, JOB, op)

    def assert_unresolved(self, h, op=OP):
        # No terminal: claimed in this process, unknown after a restart.
        self.assertEqual(
            (self.live_status(h, op).claim_state, self.status(h, op).claim_state),
            ("claimed", "unknown"),
        )

    # 1. Execute ends not_started, durably.
    def test_execute_ends_not_started_with_a_durable_terminal(self):
        h = self.make()

        async def scenario():
            await self.register(h)
            return await asgi(h.app, "POST", path(suffix="/execute"), execute_body())

        result = self.run_async(scenario())
        self.assertEqual(result.status, 200)
        events = result.events()
        self.assertEqual([name for name, _ in events], ["claimed", "terminal"])
        self.assertEqual(events[0][1], {"state": "claimed"})
        terminal = events[1][1]
        self.assertEqual(
            (terminal["state"], terminal["evidence"], terminal["backend_stopped"],
             terminal["claim_state"], terminal["cancel_requested"]),
            ("not_started", ["not_admitted"], True, "finished", False),
        )
        self.assertNotIn("owner_id", terminal)
        after = self.status(h)
        self.assertEqual(
            (type(after).__name__, after.state, after.evidence, after.claim_state),
            ("ControlTerminalStatus", "not_started", ("not_admitted",), "finished"),
        )

    # 2. Refusals that write nothing.
    def test_body_digest_and_auth_refusals_write_nothing(self):
        h = self.make()
        cases = [
            ("different request", execute_body({**REQUEST, "input": "other"}), 409),
            ("different model", execute_body({**REQUEST, "model": "other-model"}), 409),
            ("extra key", {**execute_body(), "extra": 1}, 400),
            ("missing timeout", {"request": REQUEST}, 400),
            ("request not object", {"request": [], "admission_timeout_ms": 5}, 400),
            ("zero timeout", execute_body(timeout_ms=0), 400),
            ("negative timeout", execute_body(timeout_ms=-1), 400),
            ("bool timeout", execute_body(timeout_ms=True), 400),
            ("float timeout", execute_body(timeout_ms=1.5), 400),
            ("string timeout", execute_body(timeout_ms="5"), 400),
            ("over maximum", execute_body(timeout_ms=routes.MAX_ADMISSION_TIMEOUT_MS + 1), 400),
            # A caller-supplied digest is an extra key, refused, never used.
            ("caller digest", {**execute_body(), "request_sha256": "c" * 64}, 400),
        ]

        async def scenario():
            await self.register(h)
            out = []
            for name, body, _ in cases:
                out.append((name, (await asgi(h.app, "POST", path(suffix="/execute"), body)).status))
            out.append(("oversized", (await asgi(
                h.app, "POST", path(suffix="/execute"),
                raw=b"{" + b" " * routes.MAX_REGISTRATION_BYTES + b"}")).status))
            with mock.patch.object(routes, "BODY_TIMEOUT_SECONDS", 0.05):
                out.append(("slow body", (await asgi(
                    h.app, "POST", path(suffix="/execute"), execute_body(),
                    body_delay=0.2)).status))
            out.append(("no key", (await asgi(
                h.app, "POST", path(suffix="/execute"), execute_body(), key=None)).status))
            out.append(("wrong key", (await asgi(
                h.app, "POST", path(suffix="/execute"), execute_body(), key="wrong")).status))
            # The sub key is its own owner namespace: unregistered there.
            out.append(("sub key", (await asgi(
                h.app, "POST", path(suffix="/execute"), execute_body(), key=SUB)).status))
            h.snapshot.version = 2
            out.append(("settings changed", (await asgi(
                h.app, "POST", path(suffix="/execute"), execute_body())).status))
            h.snapshot.version = 1
            h.state.bind_host = "0.0.0.0"
            out.append(("public bind", (await asgi(
                h.app, "POST", path(suffix="/execute"), execute_body())).status))
            h.state.bind_host = "127.0.0.1"
            return out

        results = dict(self.run_async(scenario()))
        expected = {name: code for name, _, code in cases}
        expected.update({"oversized": 413, "slow body": 408, "no key": 401,
                         "wrong key": 401, "sub key": 409, "settings changed": 409,
                         "public bind": 403})
        self.assertEqual(results, expected)
        after = self.status(h)
        self.assertEqual((after.state, after.claim_state), ("registered", "none"))
        self.assertIsNone(ControlStore(h.directory).lookup_status(
            sha256(SUB.encode()).hexdigest(), JOB, OP))

    # 3. Claim refusals.
    def test_claim_refusals_map_to_answers_and_write_nothing(self):
        h = self.make()

        async def scenario():
            out = {}
            out["unregistered"] = (await asgi(h.app, "POST", path(OP2, "/execute"), execute_body())).status
            await self.register(h)
            cancel = await asgi(h.app, "POST", path(suffix="/cancel"))
            self.assertEqual(cancel.status, 200)
            out["cancelled first"] = (await asgi(h.app, "POST", path(suffix="/execute"), execute_body())).status
            return out

        self.assertEqual(self.run_async(scenario()), {"unregistered": 409, "cancelled first": 409})
        after = self.status(h)
        self.assertEqual((after.state, after.claim_state), ("cancel_requested", "none"))
        self.assertIsNone(self.status(h, OP2))

    def test_second_execute_after_a_terminal_is_refused_and_changes_nothing(self):
        h = self.make()

        async def scenario():
            await self.register(h)
            first = await asgi(h.app, "POST", path(suffix="/execute"), execute_body())
            second = await asgi(h.app, "POST", path(suffix="/execute"), execute_body())
            return first, second

        first, second = self.run_async(scenario())
        self.assertEqual((first.status, second.status), (200, 409))
        self.assertEqual(second.json(), {"detail": "Control execution refused"})
        self.assertEqual(h.store.calls.count("finish"), 1)
        self.assertEqual(self.status(h).state, "not_started")

    def test_registration_from_an_earlier_store_generation_is_refused(self):
        temp = tempfile.TemporaryDirectory(prefix="prism-control-execute-")
        self.addCleanup(temp.cleanup)
        base = Path(temp.name)
        earlier = self.make(base)

        async def register_only():
            await self.register(earlier)

        self.run_async(register_only())
        # The earlier "process" exits: its lock is released.
        os.close(earlier.coordinator.lock_descriptor)
        self._cleanups = [c for c in self._cleanups if c[1] != (earlier.coordinator.lock_descriptor,)]
        later = self.make(base)
        self.assertIsNotNone(later.coordinator)
        result = self.run_async(asgi(later.app, "POST", path(suffix="/execute"), execute_body()))
        self.assertEqual(result.status, 409)
        self.assertEqual(self.status(later).claim_state, "none")

    def test_a_deadline_that_expires_before_the_claim_answers_408_with_no_claim(self):
        h = self.make()

        def slow_claim(real, *args, **kwargs):
            time.sleep(0.05)
            return real(*args, **kwargs)

        h.store.hooks["claim"] = slow_claim

        async def scenario():
            await self.register(h)
            return await asgi(h.app, "POST", path(suffix="/execute"), execute_body(timeout_ms=10))

        self.assertEqual(self.run_async(scenario()).status, 408)
        self.assertEqual(self.status(h).claim_state, "none")

    # 4. Duplicates.
    def test_concurrent_executes_leave_one_claim_and_one_terminal(self):
        h = self.make()
        release = threading.Event()

        def held_claim(real, *args, **kwargs):
            release.wait(5)
            return real(*args, **kwargs)

        h.store.hooks["claim"] = held_claim

        async def scenario():
            await self.register(h)
            tasks = [asyncio.create_task(asgi(h.app, "POST", path(suffix="/execute"), execute_body()))
                     for _ in range(4)]
            await until(lambda: sum(t.done() for t in tasks) == 3)
            live = h.coordinator._executions[identity()]
            release.set()
            return await asyncio.gather(*tasks), live

        results, live = self.run_async(scenario())
        self.assertEqual(sorted(r.status for r in results), [200, 409, 409, 409])
        self.assertEqual(
            [r.json() for r in results if r.status == 409],
            [{"detail": "Control execution exists"}] * 3,
        )
        self.assertTrue(live.final)
        self.assertEqual(h.store.calls.count("finish"), 1)
        self.assertEqual(self.status(h).state, "not_started")

    def test_a_live_execution_refuses_a_duplicate_without_any_store_call(self):
        gate = asyncio.Event()
        entered = []

        async def stage(execution):
            entered.append(execution)
            await gate.wait()
            raise coord.EngineStageUnavailable()

        h = self.make(stage=stage)

        async def scenario():
            await self.register(h)
            first = asyncio.create_task(asgi(h.app, "POST", path(suffix="/execute"), execute_body()))
            await until(lambda: entered)
            before = list(h.store.calls)
            second = await asgi(h.app, "POST", path(suffix="/execute"), execute_body())
            calls = h.store.calls[len(before):]
            await asyncio.sleep(0)
            # The refused duplicate never disturbs the live execution's entry.
            self.assertIs(h.coordinator._executions[identity()], entered[0])
            gate.set()
            return second, calls, await first

        second, calls, first = self.run_async(scenario())
        self.assertEqual(second.status, 409)
        self.assertEqual(second.json(), {"detail": "Control execution exists"})
        # The identity lookup runs; no claim, verify or finish does.
        self.assertEqual(calls, ["lookup"])
        self.assertEqual([n for n, _ in first.events()], ["claimed", "terminal"])

    def test_a_done_entry_counts_as_absent(self):
        h = self.make()

        async def scenario():
            finished = asyncio.get_running_loop().create_future()
            finished.set_result(None)
            stale = SimpleNamespace(task=finished)
            h.coordinator._executions[identity()] = stale
            await self.register(h)
            return await asgi(h.app, "POST", path(suffix="/execute"), execute_body())

        self.assertEqual(self.run_async(scenario()).status, 200)

    def test_two_starts_in_one_loop_iteration_cannot_both_run(self):
        h = self.make()

        async def scenario():
            await self.register(h)
            fields = dict(request_sha256="c" * 64, model_id="synthetic-model",
                          settings_sha256="d" * 64,
                          deadline=time.monotonic() + 5)
            first = h.coordinator.start(identity(), **fields)
            with self.assertRaises(coord.ExecutionExists):
                h.coordinator.start(identity(), **fields)
            while not first.final:
                await first.next_event()
            await asyncio.sleep(0)

        self.run_async(scenario())
        self.assertEqual(h.store.calls.count("claim"), 1)

    def test_a_finished_execution_never_forgets_its_successor(self):
        h = self.make()

        async def scenario():
            await self.register(h)
            fields = dict(request_sha256="c" * 64, model_id="synthetic-model",
                          settings_sha256="d" * 64,
                          deadline=time.monotonic() + 5)
            first = h.coordinator.start(identity(), **fields)
            while not first.final:
                await first.next_event()
            # The first task is done; its done callback has not run yet.
            self.assertTrue(first.task.done())
            self.assertIs(h.coordinator._executions[identity()], first)
            second = h.coordinator.start(identity(), **fields)
            await asyncio.sleep(0)
            self.assertIs(h.coordinator._executions[identity()], second)
            while not second.final:
                await second.next_event()
            await asyncio.sleep(0)
            return second

        second = self.run_async(scenario())
        # The store is still the gate: the successor's claim is refused.
        self.assertEqual(h.store.calls.count("claim"), 2)
        self.assertIsNone(second.claim)

    def test_an_invalid_identity_is_refused_before_any_body_read(self):
        h = self.make()

        async def scenario():
            out = []
            for method, suffix, body in (("PUT", "", {"request": REQUEST}),
                                         ("POST", "/execute", execute_body())):
                reads = []
                url = f"/prism-control/v1/jobs/not-a-uuid/operations/{OP}{suffix}"
                result = await asgi(h.app, method, url, body, reads=reads)
                out.append((method, result.status, len(reads)))
            return out

        self.assertEqual(self.run_async(scenario()), [("PUT", 400, 0), ("POST", 400, 0)])

    def test_a_deadline_inside_the_registration_write_writes_nothing(self):
        h = self.make()

        def slow_register(real, *args, **kwargs):
            time.sleep(0.15)
            return real(*args, **kwargs)

        h.store.hooks["register"] = slow_register

        async def scenario():
            with mock.patch.object(routes, "BODY_TIMEOUT_SECONDS", 0.1):
                return await asgi(h.app, "PUT", path(), {"request": REQUEST})

        self.assertEqual(self.run_async(scenario()).status, 408)
        self.assertIsNone(self.status(h))

    def test_registration_store_calls_never_block_the_event_loop(self):
        h = self.make()
        for name in ("lookup", "register"):
            h.store.hooks[name] = lambda real, *a, **k: (time.sleep(0.2), real(*a, **k))[1]

        async def scenario():
            ticks = []

            async def ticker():
                while True:
                    ticks.append(time.monotonic())
                    await asyncio.sleep(0.01)

            task = asyncio.create_task(ticker())
            await asyncio.sleep(0.02)
            await self.register(h)
            # A tick after the last store call, so a stall there is measured.
            await asyncio.sleep(0.03)
            task.cancel()
            return max(b - a for a, b in zip(ticks, ticks[1:])), len(ticks)

        # Two 0.2-second store calls; the loop never stalls on either of them.
        gap, count = self.run_async(scenario())
        self.assertLess(gap, 0.1)
        self.assertGreater(count, 20)

    def test_the_admission_budget_counts_the_body_read(self):
        h = self.make()

        async def scenario():
            await self.register(h)
            return await asgi(h.app, "POST", path(suffix="/execute"),
                              execute_body(timeout_ms=100), body_delay=0.2)

        self.assertEqual(self.run_async(scenario()).status, 408)
        self.assertEqual(self.status(h).claim_state, "none")

    # 5. Disconnects never cancel the coordinator.
    def test_a_disconnect_after_claimed_leaves_the_execution_running(self):
        gate = asyncio.Event()
        entered = []

        async def stage(execution):
            entered.append(execution)
            await gate.wait()
            raise coord.EngineStageUnavailable()

        h = self.make(stage=stage)

        async def scenario():
            await self.register(h)
            disconnect = asyncio.Event()

            def on_send(message):
                if message["type"] == "http.response.body" and b"claimed" in message.get("body", b""):
                    disconnect.set()

            result = await asyncio.wait_for(asgi(
                h.app, "POST", path(suffix="/execute"), execute_body(),
                disconnect=disconnect, on_send=on_send), 5)
            await until(lambda: entered)
            execution = entered[0]
            self.assertFalse(execution.task.done())
            gate.set()
            await asyncio.wait_for(asyncio.shield(execution.task), 5)
            return result, execution

        result, execution = self.run_async(scenario())
        self.assertEqual(result.status, 200)
        self.assertNotIn(b"terminal", result.body)
        self.assertFalse(execution.task.cancelled())
        self.assertEqual(self.status(h).state, "not_started")

    def test_a_disconnect_before_the_first_event_leaves_the_execution_running(self):
        release = threading.Event()
        h = self.make()

        def held_claim(real, *args, **kwargs):
            release.wait(5)
            return real(*args, **kwargs)

        h.store.hooks["claim"] = held_claim

        async def scenario():
            await self.register(h)
            disconnect = asyncio.Event()
            disconnect.set()
            request = asyncio.create_task(asgi(
                h.app, "POST", path(suffix="/execute"), execute_body(), disconnect=disconnect))
            await until(lambda: identity() in h.coordinator._executions)
            execution = h.coordinator._executions[identity()]
            await asyncio.sleep(0.05)
            self.assertFalse(execution.task.done())
            release.set()
            await asyncio.wait_for(asyncio.shield(execution.task), 5)
            await asyncio.wait_for(request, 5)
            return execution

        execution = self.run_async(scenario())
        self.assertFalse(execution.task.cancelled())
        self.assertEqual(self.status(h).state, "not_started")

    # 6. Cancel ordering.
    def fence_stage(self, log, *, after_verify=None, admitted_gate=None, final_verify=True):
        class Handle:
            def __init__(self):
                self.lock = threading.Lock()
                self.cancelled = False
                self.admitted = False

            def fence(self):
                with self.lock:
                    self.cancelled = True
                    log.append("fence")
                    if self.admitted:
                        log.append("cleanup started")

            def admit(self):
                with self.lock:
                    if self.cancelled:
                        log.append("admission refused")
                        return False
                    self.admitted = True
                    log.append("admitted")
                    return True

        async def stage(execution):
            handle = Handle()
            # Publish before the final verify [slice 26 F2-1].
            execution.publish_fence(handle.fence)
            if final_verify:
                try:
                    await execution.verify()
                except Exception:
                    log.append("final verify refused")
                    raise coord.EngineStageUnavailable() from None
            if after_verify is not None:
                await after_verify()
            handle.admit()
            if admitted_gate is not None:
                await admitted_gate.wait()
            raise coord.EngineStageUnavailable()

        return stage

    def test_verify_before_the_commit_never_lets_admission_follow_the_ack(self):
        log = []
        verified = threading.Event()
        proceed = threading.Event()

        calls = []

        def verify_then_hold(real, *args, **kwargs):
            real(*args, **kwargs)
            calls.append(None)
            if len(calls) == 2:
                # The stage's final verify read before the cancellation commits.
                verified.set()
                proceed.wait(5)

        h = self.make(stage=self.fence_stage(log))
        h.store.hooks["verify_claim"] = verify_then_hold

        async def scenario():
            await self.register(h)
            execute = asyncio.create_task(asgi(h.app, "POST", path(suffix="/execute"), execute_body()))
            await until(verified.is_set)

            def on_send(message):
                if message["type"] == "http.response.start":
                    log.append("ack")

            cancel = await asgi(h.app, "POST", path(suffix="/cancel"), on_send=on_send)
            proceed.set()
            return cancel, await asyncio.wait_for(execute, 5)

        cancel, execute = self.run_async(scenario())
        self.assertEqual(cancel.status, 200)
        # The fence landed during the final verify's store read; verify
        # re-checks the mark afterwards, so the stage never admits.
        self.assertEqual(log, ["fence", "ack", "final verify refused"])
        terminal = execute.events()[-1][1]
        self.assertEqual((terminal["state"], terminal["cancel_requested"]), ("not_started", True))

    def test_a_published_fence_stops_an_admission_that_follows_the_ack(self):
        # The stage-level contract of slice 26 F2-1, without verify's help:
        # once the fence is published, admission checks its own cancel mark.
        log = []
        published = asyncio.Event()
        hold = asyncio.Event()

        async def after_publish():
            published.set()
            await hold.wait()

        h = self.make(stage=self.fence_stage(log, after_verify=after_publish, final_verify=False))

        async def scenario():
            await self.register(h)
            execute = asyncio.create_task(asgi(h.app, "POST", path(suffix="/execute"), execute_body()))
            await published.wait()

            def on_send(message):
                if message["type"] == "http.response.start":
                    log.append("ack")

            cancel = await asgi(h.app, "POST", path(suffix="/cancel"), on_send=on_send)
            hold.set()
            return cancel, await asyncio.wait_for(execute, 5)

        cancel, execute = self.run_async(scenario())
        self.assertEqual(cancel.status, 200)
        self.assertEqual(log, ["fence", "ack", "admission refused"])

    def test_an_admission_before_the_commit_has_cleanup_started_before_the_ack(self):
        log = []
        commit = threading.Event()
        holding = threading.Event()
        admitted_gate = asyncio.Event()

        def hold_commit(real, *args, **kwargs):
            holding.set()
            commit.wait(5)
            return real(*args, **kwargs)

        h = self.make(stage=self.fence_stage(log, admitted_gate=admitted_gate))
        h.store.hooks["request_cancel"] = hold_commit

        async def scenario():
            await self.register(h)

            def on_send(message):
                if message["type"] == "http.response.start":
                    log.append("ack")

            cancel = asyncio.create_task(asgi(h.app, "POST", path(suffix="/cancel"), on_send=on_send))
            await until(holding.is_set)
            execute = asyncio.create_task(asgi(h.app, "POST", path(suffix="/execute"), execute_body()))
            await until(lambda: "admitted" in log)
            commit.set()
            cancel = await asyncio.wait_for(cancel, 5)
            admitted_gate.set()
            return cancel, await asyncio.wait_for(execute, 5)

        cancel, execute = self.run_async(scenario())
        self.assertEqual(cancel.status, 200)
        self.assertEqual(log, ["admitted", "fence", "cleanup started", "ack"])

    def test_a_fence_that_raises_answers_503_and_a_retry_fences_again(self):
        gate = asyncio.Event()
        fences = []

        async def stage(execution):
            def fence():
                fences.append(None)
                if len(fences) == 1:
                    raise RuntimeError("synthetic fence failure")

            execution.publish_fence(fence)
            await gate.wait()
            raise coord.EngineStageUnavailable()

        h = self.make(stage=stage)

        async def scenario():
            await self.register(h)
            execute = asyncio.create_task(asgi(h.app, "POST", path(suffix="/execute"), execute_body()))
            await until(lambda: identity() in h.coordinator._executions
                        and h.coordinator._executions[identity()]._fence is not None)
            first = await asgi(h.app, "POST", path(suffix="/cancel"))
            second = await asgi(h.app, "POST", path(suffix="/cancel"))
            gate.set()
            await asyncio.wait_for(execute, 5)
            return first, second

        first, second = self.run_async(scenario())
        self.assertEqual((first.status, first.json()), (503, {"detail": "Control cancellation incomplete"}))
        self.assertEqual(second.status, 200)
        self.assertEqual(len(fences), 2)

    def test_a_cancellation_between_claim_and_stage_ends_not_started(self):
        # "store read skipped": verify_claim does not read the tombstone, so
        # only the in-process cancel mark the fence set can refuse the stage.
        for variant in ("store read", "store read skipped"):
            with self.subTest(variant):
                stage_ran = []

                async def stage(execution):
                    stage_ran.append(None)
                    raise coord.EngineStageUnavailable()

                h = self.make(stage=stage)
                claimed = threading.Event()
                proceed = threading.Event()

                def claim_then_hold(real, *args, **kwargs):
                    result = real(*args, **kwargs)
                    claimed.set()
                    proceed.wait(5)
                    return result

                h.store.hooks["claim"] = claim_then_hold
                if variant == "store read skipped":
                    h.store.hooks["verify_claim"] = lambda real, *a, **k: None

                async def scenario():
                    await self.register(h)
                    execute = asyncio.create_task(
                        asgi(h.app, "POST", path(suffix="/execute"), execute_body()))
                    await until(claimed.is_set)
                    cancel = await asgi(h.app, "POST", path(suffix="/cancel"))
                    self.assertTrue(h.coordinator._executions[identity()].cancel_requested)
                    proceed.set()
                    return cancel, await asyncio.wait_for(execute, 5)

                cancel, execute = self.run_async(scenario())
                self.assertEqual(cancel.status, 200)
                self.assertEqual(stage_ran, [])
                terminal = execute.events()[-1][1]
                self.assertEqual(
                    (terminal["state"], terminal["evidence"], terminal["cancel_requested"]),
                    ("not_started", ["not_admitted"], True))

    # 7. The check before the stage.
    def test_settings_changed_or_deadline_passed_after_the_claim_ends_not_started(self):
        for name in ("settings", "deadline", "settings callback raises"):
            with self.subTest(name):
                stage_ran = []

                async def stage(execution):
                    stage_ran.append(None)
                    return ("completed", ("finished_output", "absent_proved", "stream_synchronized"))

                h = self.make(stage=stage)

                def claim_then_change(real, *args, **kwargs):
                    result = real(*args, **kwargs)
                    if name == "settings":
                        h.snapshot.version = 2
                    elif name == "deadline":
                        time.sleep(0.15)
                    else:
                        h.snapshot.fail = True
                    return result

                h.store.hooks["claim"] = claim_then_change

                async def scenario():
                    await self.register(h)
                    return await asgi(h.app, "POST", path(suffix="/execute"),
                                      execute_body(timeout_ms=100 if name == "deadline" else 5000))

                result = self.run_async(scenario())
                self.assertEqual(result.status, 200)
                self.assertEqual(stage_ran, [])
                terminal = result.events()[-1]
                self.assertEqual((terminal[0], terminal[1]["state"], terminal[1]["evidence"]),
                                 ("terminal", "not_started", ["not_admitted"]))

    def test_a_stage_that_fails_unexpectedly_publishes_no_terminal(self):
        async def stage(execution):
            raise RuntimeError("synthetic stage failure")

        h = self.make(stage=stage)

        async def scenario():
            await self.register(h)
            return await asgi(h.app, "POST", path(suffix="/execute"), execute_body())

        result = self.run_async(scenario())
        self.assertEqual([n for n, _ in result.events()], ["claimed", "unresolved"])
        self.assertEqual(h.store.calls.count("finish"), 0)
        self.assert_unresolved(h)

    # 8. Finish retries.
    def test_finish_retries_only_while_the_store_is_unavailable(self):
        def failing(times, error=ControlStoreError("control_store_unavailable"), after_commit=False):
            count = [0]

            def hook(real, *args, **kwargs):
                count[0] += 1
                if count[0] <= times:
                    if after_commit:
                        real(*args, **kwargs)
                    raise error
                return real(*args, **kwargs)

            return hook

        cases = [
            ("two failures then success", failing(2), "terminal", 3),
            ("committed then reported failure", failing(1, after_commit=True), "terminal", 2),
            ("conflict is final", failing(1, ControlConflict("control_terminal_exists")), "unresolved", 1),
            ("unavailable for the whole window", failing(10**6), "unresolved", None),
        ]
        for name, hook, last, attempts in cases:
            with self.subTest(name):
                h = self.make()
                h.store.hooks["finish"] = hook

                async def scenario():
                    await self.register(h)
                    with mock.patch.object(coord, "DRAIN_WINDOW_SECONDS", 0.4):
                        return await asgi(h.app, "POST", path(suffix="/execute"), execute_body())

                started = time.monotonic()
                result = self.run_async(scenario())
                elapsed = time.monotonic() - started
                self.assertEqual(result.events()[-1][0], last)
                if attempts is not None:
                    self.assertEqual(h.store.calls.count("finish"), attempts)
                else:
                    self.assertGreater(h.store.calls.count("finish"), 2)
                    self.assertLess(elapsed, 3)
                if last == "terminal":
                    self.assertEqual(self.status(h).state, "not_started")
                else:
                    self.assert_unresolved(h)

    # 9. The claim shield.
    def test_a_cancelled_task_waits_for_its_claim_and_finishes_it(self):
        release = threading.Event()
        h = self.make()

        def held_claim(real, *args, **kwargs):
            release.wait(5)
            return real(*args, **kwargs)

        h.store.hooks["claim"] = held_claim

        async def scenario():
            await self.register(h)
            request = asyncio.create_task(asgi(h.app, "POST", path(suffix="/execute"), execute_body()))
            await until(lambda: identity() in h.coordinator._executions)
            execution = h.coordinator._executions[identity()]
            await asyncio.sleep(0.02)
            execution.task.cancel()
            await asyncio.sleep(0.02)
            execution.task.cancel()
            await asyncio.sleep(0.02)
            self.assertFalse(execution.task.done())
            release.set()
            try:
                await asyncio.wait_for(asyncio.shield(execution.task), 5)
            except asyncio.CancelledError:
                pass
            request.cancel()
            return execution

        execution = self.run_async(scenario())
        self.assertTrue(execution.task.cancelled())
        self.assertEqual(self.status(h).state, "not_started")

    # 10. Drain.
    def test_drain_refuses_new_work_first_and_waits_for_live_executions(self):
        gate = asyncio.Event()
        entered = []

        async def stage(execution):
            entered.append(execution)
            await gate.wait()
            raise coord.EngineStageUnavailable()

        h = self.make(stage=stage)

        async def scenario():
            await self.register(h)
            await self.register(h, OP2)
            first = asyncio.create_task(asgi(h.app, "POST", path(suffix="/execute"), execute_body()))
            await until(lambda: entered)
            drain = asyncio.create_task(h.coordinator.drain(window=5))
            await asyncio.sleep(0)
            late = await asgi(h.app, "POST", path(OP2, "/execute"), execute_body())
            self.assertFalse(drain.done())
            loop = asyncio.get_running_loop()
            loop.call_later(0.05, gate.set)
            await asyncio.wait_for(drain, 5)
            self.assertTrue(entered[0].task.done())
            return late, await first

        late, first = self.run_async(scenario())
        self.assertEqual((late.status, late.json()), (503, {"detail": "Control execution unavailable"}))
        self.assertEqual(self.status(h, OP2).claim_state, "none")
        self.assertEqual(self.status(h).state, "not_started")

    def test_drain_stops_an_execution_before_its_engine_stage(self):
        stage_ran = []

        async def stage(execution):
            stage_ran.append(None)
            raise coord.EngineStageUnavailable()

        h = self.make(stage=stage)
        release = threading.Event()

        def held_claim(real, *args, **kwargs):
            release.wait(5)
            return real(*args, **kwargs)

        h.store.hooks["claim"] = held_claim

        async def scenario():
            await self.register(h)
            request = asyncio.create_task(asgi(h.app, "POST", path(suffix="/execute"), execute_body()))
            await until(lambda: identity() in h.coordinator._executions)
            drain = asyncio.create_task(h.coordinator.drain(window=5))
            await asyncio.sleep(0)
            release.set()
            await asyncio.wait_for(drain, 5)
            return await asyncio.wait_for(request, 5)

        result = self.run_async(scenario())
        self.assertEqual(stage_ran, [])
        terminal = result.events()[-1][1]
        self.assertEqual((terminal["state"], terminal["evidence"]), ("not_started", ["not_admitted"]))

    def test_a_drain_during_the_verify_read_stops_the_stage(self):
        stage_ran = []

        async def stage(execution):
            stage_ran.append(None)
            raise coord.EngineStageUnavailable()

        h = self.make(stage=stage)
        reading = threading.Event()
        proceed = threading.Event()

        def held_verify(real, *args, **kwargs):
            real(*args, **kwargs)
            reading.set()
            proceed.wait(5)

        h.store.hooks["verify_claim"] = held_verify

        async def scenario():
            await self.register(h)
            request = asyncio.create_task(asgi(h.app, "POST", path(suffix="/execute"), execute_body()))
            await until(reading.is_set)
            drain = asyncio.create_task(h.coordinator.drain(window=5))
            await asyncio.sleep(0)
            proceed.set()
            await asyncio.wait_for(drain, 5)
            return await asyncio.wait_for(request, 5)

        result = self.run_async(scenario())
        self.assertEqual(stage_ran, [])
        terminal = result.events()[-1][1]
        self.assertEqual((terminal["state"], terminal["evidence"]), ("not_started", ["not_admitted"]))

    def test_an_execution_that_dies_still_ends_its_stream(self):
        async def stage(execution):
            # Stands in for a loop-close cancellation during an engine stage.
            asyncio.current_task().cancel()
            await asyncio.sleep(0)

        h = self.make(stage=stage)

        async def scenario():
            await self.register(h)
            return await asyncio.wait_for(
                asgi(h.app, "POST", path(suffix="/execute"), execute_body()), 5)

        result = self.run_async(scenario())
        self.assertEqual([n for n, _ in result.events()], ["claimed", "unresolved"])
        self.assert_unresolved(h)

    def test_drain_never_cancels_and_keeps_the_lock_after_its_window(self):
        gate = asyncio.Event()
        entered = []

        async def stage(execution):
            entered.append(execution)
            await gate.wait()
            raise coord.EngineStageUnavailable()

        h = self.make(stage=stage)

        async def scenario():
            await self.register(h)
            first = asyncio.create_task(asgi(h.app, "POST", path(suffix="/execute"), execute_body()))
            await until(lambda: entered)
            started = time.monotonic()
            await h.coordinator.drain(window=0.1)
            waited = time.monotonic() - started
            # Let any cancellation the drain might have requested land.
            await asyncio.sleep(0.05)
            task = entered[0].task
            state = (task.done(), task.cancelled(), entered[0].stop_requested)
            status = (self.live_status(h).claim_state, self.status(h).claim_state)
            lock = coord.acquire_execution_lock(h.directory)
            gate.set()
            await asyncio.wait_for(first, 5)
            return waited, state, status, lock

        waited, state, status, lock = self.run_async(scenario())
        self.assertGreaterEqual(waited, 0.1)
        self.assertEqual(state, (False, False, True))
        self.assertEqual(status, ("claimed", "unknown"))
        self.assertIsNone(lock)

    # 11. Execution lock.
    def test_only_the_lock_holder_executes(self):
        first = self.make()
        second = self.make(first.base)
        self.assertIsNotNone(first.coordinator)
        self.assertIsNone(second.coordinator)

        async def scenario():
            await self.register(second)
            out = [(await asgi(second.app, "POST", path(suffix="/execute"), execute_body())).status]
            out.append((await asgi(second.app, "GET", path())).status)
            out.append((await asgi(second.app, "POST", path(suffix="/cancel"))).status)
            return out

        self.assertEqual(self.run_async(scenario()), [503, 200, 200])
        # -E -P, not -I: ignore PYTHON* variables and the unsafe path entry,
        # but keep this interpreter's site-packages (user site included), so
        # the child imports with the same packages the parent did.
        child = subprocess.run(
            [sys.executable, "-E", "-P", "-c",
             "import sys; sys.path.insert(0, sys.argv[1]);"
             "from omlx.api.prism_control_coordinator import acquire_execution_lock as a;"
             "print(a(sys.argv[2]))", str(ROOT), str(first.directory)],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(child.stdout.strip(), "None", child.stderr)

    def test_an_unsafe_lock_file_disables_execution(self):
        for name in ("symlink", "mode", "non-empty", "directory"):
            with self.subTest(name):
                temp = tempfile.TemporaryDirectory(prefix="prism-control-execute-")
                self.addCleanup(temp.cleanup)
                directory = Path(temp.name) / "prism-control"
                directory.mkdir(mode=0o700)
                lock = directory / coord.EXECUTION_LOCK
                if name == "symlink":
                    target = Path(temp.name) / "elsewhere"
                    target.touch(mode=0o600)
                    lock.symlink_to(target)
                elif name == "mode":
                    lock.touch(mode=0o600)
                    lock.chmod(0o644)
                elif name == "non-empty":
                    lock.write_bytes(b"x")
                    lock.chmod(0o600)
                else:
                    lock.mkdir(mode=0o700)
                h = self.make(Path(temp.name))
                self.assertIsNotNone(h.state.prism_control_store)
                self.assertIsNone(h.coordinator)

    # 12. Source wiring.
    def test_server_wiring_and_engine_free_coordinator(self):
        server = ast.parse((ROOT / "omlx/server.py").read_text())
        functions = {n.name: n for n in ast.walk(server)
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        setup = ast.unparse(functions["init_server"])
        # 27.5: the coordinator's engine stage is owned admission. The pool
        # and the oQ manager are created later, so both are read per call;
        # MCP exposure is off for owned work.
        self.assertLess(setup.index("configure_control_store(_server_state"),
                        setup.index(
                            "configure_control_coordinator(_server_state, "
                            "get_control_settings_snapshot, make_admission_stage("
                            "lambda: _server_state.engine_pool, "
                            "lambda engine, request, model_id: _prepare_responses_chat("
                            "engine, request, model_id, mcp_manager=None), "
                            "lambda request: owned_request_refusal(request, quantizing=bool("
                            "_server_state.oq_manager and "
                            "_server_state.oq_manager.is_quantizing))))"))
        self.assertNotIn("make_preparation_stage", ast.unparse(server))
        lifespan = functions["lifespan"]
        body = lifespan.body
        index = next(i for i, s in enumerate(body)
                     if isinstance(s, ast.Expr) and isinstance(s.value, ast.Yield))
        after = body[index + 1:]
        first_await = next(s for s in after if any(isinstance(n, ast.Await) for n in ast.walk(s)))
        self.assertIs(first_await, after[0])
        self.assertIn("prism_control_coordinator.drain()", ast.unparse(after[0]))
        rest = ast.unparse(ast.Module(body=after, type_ignores=[]))
        self.assertLess(rest.index("prism_control_coordinator.drain()"),
                        rest.index("engine_pool.shutdown()"))
        module = ast.parse((ROOT / "omlx/api/prism_control_coordinator.py").read_text())
        imported = set()
        for node in ast.walk(module):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(("." * node.level) + (node.module or ""))
        self.assertEqual(imported, {"functools", "pathlib", "asyncio", "fcntl", "os",
                                    "stat", "time", ".prism_control_store"})
        coordinator = next(n for n in module.body
                           if isinstance(n, ast.ClassDef) and n.name == "ControlCoordinator")
        names = {n.attr for n in ast.walk(coordinator) if isinstance(n, ast.Attribute)}
        self.assertNotIn("close", names, "the coordinator never closes its lock")
        cli = ast.parse((ROOT / "omlx/cli.py").read_text())
        configs = [n for n in ast.walk(cli) if isinstance(n, ast.Call)
                   and ast.unparse(n.func) == "uvicorn.Config"]
        self.assertEqual(len(configs), 2)
        for call in configs:
            self.assertNotIn("timeout_graceful_shutdown", {k.arg for k in call.keywords})

    def test_a_claim_store_failure_answers_503_and_leaves_no_entry(self):
        h = self.make()

        def failing_claim(real, *args, **kwargs):
            raise ControlStoreError("control_store_unavailable")

        h.store.hooks["claim"] = failing_claim

        async def scenario():
            await self.register(h)
            result = await asgi(h.app, "POST", path(suffix="/execute"), execute_body())
            await asyncio.sleep(0.01)
            return result, dict(h.coordinator._executions)

        result, entries = self.run_async(scenario())
        self.assertEqual((result.status, result.json()), (503, {"detail": "Control state unavailable"}))
        self.assertEqual(entries, {})


if __name__ == "__main__":
    unittest.main()
