# SPDX-License-Identifier: Apache-2.0
"""Slice 26 durable execution claim: the real store and router, no engines.

Process tests run real child interpreters that load the store from the same
source tree. Power-loss durability is not tested.
"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
import uuid

import test_provider_control_store as support
import test_provider_control_routes as http

api = support.api
route_store = sys.modules["omlx.api.prism_control_store"]
routes = sys.modules["omlx.api.prism_control_routes"]
OWNER, JOB, OP = support.OWNER, support.JOB, support.OP
FIELDS = support.FIELDS
REAL_CONNECT = sqlite3.connect

CHILD = r"""
import importlib.util, json, os, sys, time
from pathlib import Path
spec = importlib.util.spec_from_file_location(
    "claim_child_store",
    Path(os.environ["OMLX_CONTROL_ROOT"]) / "omlx/api/prism_control_store.py",
)
api = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = api
spec.loader.exec_module(api)
args = json.loads(sys.argv[1])
store = api.ControlStore(args["directory"], capacity=args["capacity"])
if args.get("generation"):
    store._generation_id = args["generation"]
fields = args["fields"]
if args.get("exit_inside_claim"):
    calls = []
    real_now = api._now
    def now():
        calls.append(1)
        if len(calls) == 3:  # after the insert, before commit
            os._exit(0)
        return real_now()
    api._now = now
start = args.get("start")
if start:
    while time.time() < start:
        time.sleep(0.0005)
out = {}
try:
    if args.get("register"):
        store.register(**fields)
    if args["action"] == "claim":
        claim = store.claim(**fields, deadline=time.monotonic() + 10)
        out = {"ok": True, "claim": [
            claim.owner_id, claim.job_id, claim.operation_id,
            claim.store_generation_id, claim.worker_id, claim.request_id,
        ]}
    elif args["action"] == "cancel":
        record = store.request_cancel(
            fields["owner_id"], fields["job_id"], fields["operation_id"]
        )
        out = {"ok": True, "claim_state": record.claim_state}
except Exception as error:
    out = {"ok": False, "type": type(error).__name__, "reason": str(error)}
print(json.dumps(out), flush=True)
if args.get("exit_after"):
    os._exit(0)
"""


def later(seconds=10):
    return time.monotonic() + seconds


def spawn(**args):
    return subprocess.Popen(
        [sys.executable, "-I", "-c", CHILD, json.dumps(args)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def result(process):
    out, err = process.communicate(timeout=60)
    lines = out.strip().splitlines()
    return json.loads(lines[-1]) if lines else {"exited": process.returncode, "stderr": err[-300:]}


class Recorder:
    """Proxy for the store's own sqlite connection: records SQL, injects faults."""

    def __init__(self, real, log, fail_commit=False, fail_close_after_commit=False, fail_close=False):
        self._real, self._log = real, log
        self._fail_commit, self._fail_close = fail_commit, fail_close_after_commit
        self._fail_close_always = fail_close
        self._committed = False

    def execute(self, sql, *args):
        self._log.append(sql)
        result = self._real.execute(sql, *args)
        if sql == "PRAGMA fullfsync = ON":
            self._log.append(("effective fullfsync", self._real.execute("PRAGMA fullfsync").fetchone()[0]))
        return result

    def commit(self):
        if self._fail_commit:
            raise sqlite3.OperationalError("synthetic commit failure")
        self._real.commit()
        self._committed = True

    def close(self):
        self._real.close()
        if self._fail_close_always or (self._fail_close and self._committed):
            raise sqlite3.OperationalError("synthetic close failure")

    def __getattr__(self, name):
        return getattr(self._real, name)


class ClaimStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="prism-control-claim-")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / "control"
        self.store = api.ControlStore(self.directory, capacity=8)

    def register(self, store=None, **changes):
        return (store or self.store).register(**(FIELDS | changes))

    def claim(self, store=None, **changes):
        return (store or self.store).claim(**(FIELDS | {"deadline": later()} | changes))

    def rows(self, path=None):
        path = path or self.store.path
        with closing(REAL_CONNECT(f"file:{path}?mode=ro", uri=True)) as connection:
            return connection.execute("SELECT * FROM claims").fetchall()

    def fresh(self, capacity=8):
        return api.ControlStore(self.directory, capacity=capacity)

    @contextmanager
    def connections(self, **faults):
        log, path = [], str(self.store.path)

        def connect(target, *args, **kwargs):
            real = REAL_CONNECT(target, *args, **kwargs)
            return Recorder(real, log, **faults) if str(target) == path else real

        with mock.patch.object(api.sqlite3, "connect", connect):
            yield log

    @contextmanager
    def exact_store_error(self):
        # ControlConflict is a ControlStoreError subclass; insist on the base.
        with self.assertRaises(api.ControlStoreError) as caught:
            yield
        self.assertIs(type(caught.exception), api.ControlStoreError)
        self.assertEqual(str(caught.exception), "control_store_unavailable")

    def assert_refused(self, reason, kind=None, **changes):
        before = self.store.path.read_bytes()
        with self.assertRaises(kind or api.ControlConflict) as caught:
            self.claim(**changes)
        self.assertEqual(str(caught.exception), reason)
        self.assertEqual(self.store.path.read_bytes(), before)
        return caught.exception

    # 1. Success, identity and durability.
    def test_claim_persists_distinct_ids_and_survives_a_fresh_instance(self):
        self.register()
        claim = self.claim()
        self.assertIs(type(claim), api.ControlClaim)
        ids = [claim.worker_id, claim.request_id, claim.store_generation_id]
        self.assertEqual(len(set(ids)), 3)
        for value in ids:
            self.assertEqual(str(uuid.UUID(value)), value)
        self.assertEqual(claim.store_generation_id, self.store.generation_id)
        self.assertNotIn(claim.request_id, {JOB, OP})
        self.assertEqual(
            self.rows(),
            [(OWNER, JOB, OP, claim.store_generation_id, claim.worker_id, claim.request_id)],
        )
        status = self.store.lookup_status(OWNER, JOB, OP)
        self.assertEqual(
            (type(status), status.state, status.claim_state), (api.ControlClaimStatus, "claimed", "claimed")
        )
        restarted = self.fresh().lookup_status(OWNER, JOB, OP)
        self.assertEqual(
            (restarted.state, restarted.claim_state, restarted.claim_store_generation_id),
            ("unknown", "unknown", claim.store_generation_id),
        )
        seen = {claim.worker_id, claim.request_id}
        for _ in range(2):
            op = str(uuid.uuid4())
            self.register(operation_id=op)
            other = self.claim(operation_id=op)
            seen |= {other.worker_id, other.request_id}
        self.assertEqual(len(seen), 6)

    # 2. Every refusal writes nothing and names its reason.
    def test_every_refusal_writes_nothing_and_names_its_reason(self):
        self.register()
        self.assert_refused("control_claim_unregistered", operation_id=str(uuid.uuid4()))
        self.assert_refused("control_claim_unregistered", owner_id=support.OTHER_OWNER)
        for name, changes in [
            ("request", dict(request_sha256="e" * 64)),
            ("model", dict(model_id="synthetic-other-model")),
            ("settings", dict(settings_sha256="f" * 64)),
        ]:
            with self.subTest(mismatch=name):
                self.assert_refused("control_claim_digest_mismatch", **changes)
        self.assertEqual(self.rows(), [])
        cancelled = str(uuid.uuid4())
        self.register(operation_id=cancelled)
        self.store.request_cancel(OWNER, JOB, cancelled)
        self.assert_refused("control_claim_cancel_requested", operation_id=cancelled)
        # A registration from an earlier store generation is stale.
        restarted = self.fresh()
        before = self.store.path.read_bytes()
        with self.assertRaises(api.ControlConflict) as caught:
            self.claim(store=restarted)
        self.assertEqual(str(caught.exception), "control_claim_stale_registration")
        self.assertEqual(self.store.path.read_bytes(), before)
        # An existing claim, from this generation or an earlier one, reports itself.
        self.claim()
        self.assert_refused("control_claim_exists")
        with self.assertRaises(api.ControlConflict) as caught:
            self.claim(store=restarted)
        self.assertEqual(str(caught.exception), "control_claim_exists")
        self.assertEqual(len(self.rows()), 1)

    # 3. Deadlines.
    def test_invalid_deadlines_and_call_time_expiry_take_no_lock(self):
        self.register()
        for value in (True, float("nan"), float("inf"), "1", None):
            with self.subTest(deadline=value):
                with self.assertRaises(ValueError):
                    self.claim(deadline=value)
        with mock.patch.object(self.store, "_exclusion", side_effect=AssertionError("locked")):
            with self.assertRaises(TimeoutError):
                self.claim(deadline=time.monotonic() - 1)
            for field, value in [
                ("owner_id", "x"), ("job_id", "not-a-uuid"), ("operation_id", OP.replace("-", "")),
                ("request_sha256", "g" * 64), ("settings_sha256", "c" * 63), ("model_id", "bad model"),
            ]:
                with self.subTest(field=field), self.assertRaises(ValueError):
                    self.claim(**{field: value})
            # The deadline itself is already expired, not merely close.
            exact = time.monotonic() + 5
            with mock.patch.object(api, "_now", return_value=exact), self.assertRaises(TimeoutError):
                self.claim(deadline=exact)
        self.assertEqual(self.rows(), [])

    def test_expiry_between_insert_and_commit_is_a_timeout_with_nothing_written(self):
        self.register()
        before = self.store.path.read_bytes()
        base = time.monotonic()
        with mock.patch.object(api, "_now", side_effect=[base, base, base + 100]):
            with self.assertRaises(TimeoutError) as caught:
                self.claim(deadline=base + 10)
        self.assertIs(type(caught.exception), TimeoutError)
        self.assertEqual(str(caught.exception), "control_claim_expired")
        self.assertEqual(self.rows(), [])
        self.assertEqual(self.store.path.read_bytes(), before)
        self.assertFalse(Path(str(self.store.path) + "-journal").exists())
        self.assertIs(type(self.claim()), api.ControlClaim)  # the lock was released

    def test_expiry_after_commit_still_returns_the_durable_claim(self):
        self.register()
        base = time.monotonic()
        with mock.patch.object(api, "_now", side_effect=[base, base, base, base + 100]):
            claim = self.claim(deadline=base + 10)
            # The caller's next check, before resource work, sees the expiry.
            with self.assertRaises(TimeoutError):
                self.store.verify_claim(claim, settings_sha256=FIELDS["settings_sha256"], deadline=base + 10)
        self.assertEqual(len(self.rows()), 1)

    # 4. Failures around commit.
    def test_failures_after_commit_still_return_the_claim(self):
        cases = ["post_block_check", "connection_close", "lock_close"]
        for case in cases:
            with self.subTest(case=case):
                op = str(uuid.uuid4())
                self.register(operation_id=op)
                committed = lambda: any(row[2] == op for row in self.rows())
                if case == "post_block_check":
                    real = self.store._check_file

                    def check():
                        if committed():
                            raise api.ControlStoreError("control_store_unavailable")
                        return real()

                    with mock.patch.object(self.store, "_check_file", check):
                        claim = self.claim(operation_id=op)
                elif case == "connection_close":
                    with self.connections(fail_close_after_commit=True):
                        claim = self.claim(operation_id=op)
                else:
                    real_open, real_close, lock_fds = os.open, os.close, set()

                    def opener(path, *args, **kwargs):
                        fd = real_open(path, *args, **kwargs)
                        if str(path).endswith("control.lock"):
                            lock_fds.add(fd)
                        return fd

                    def closer(fd):
                        real_close(fd)
                        if fd in lock_fds and committed():
                            raise OSError("synthetic lock close failure")

                    with mock.patch.object(api.os, "open", opener), mock.patch.object(api.os, "close", closer):
                        claim = self.claim(operation_id=op)
                self.assertIs(type(claim), api.ControlClaim)
                self.assertEqual([row for row in self.rows() if row[2] == op], [
                    (OWNER, JOB, op, claim.store_generation_id, claim.worker_id, claim.request_id)
                ])

    def test_failures_before_commit_raise_and_write_nothing(self):
        self.register()
        with self.connections(fail_commit=True):
            with self.exact_store_error():
                self.claim()
        self.assertEqual(self.rows(), [])
        with self.connections(fail_commit=True, fail_close=True):
            with self.exact_store_error():
                self.claim()
        self.assertEqual(self.rows(), [])
        # A failing step-8 identity re-check refuses before commit.
        armed, calls = [], []
        real_now, real_check = api._now, self.store._check_directory

        def now():
            calls.append(1)
            if len(calls) == 3:  # after the insert: the next check is step 8
                armed.append(True)
            return real_now()

        def check():
            if armed:
                raise api.ControlStoreError("control_store_unavailable")
            return real_check()

        with mock.patch.object(api, "_now", now), mock.patch.object(self.store, "_check_directory", check):
            with self.exact_store_error():
                self.claim()
        self.assertEqual(self.rows(), [])
        self.assertIs(type(self.claim()), api.ControlClaim)  # a retry then succeeds

    def test_a_file_replaced_during_the_transaction_is_refused(self):
        self.register()
        copy = Path(self.temp.name) / "copy.sqlite3"
        copy.write_bytes(self.store.path.read_bytes())
        os.chmod(copy, 0o600)
        real_now, calls = api._now, []

        def now():
            calls.append(1)
            if len(calls) == 3:  # after the insert, before the step-8 check
                os.replace(copy, self.store.path)
            return real_now()

        with mock.patch.object(api, "_now", now):
            with self.exact_store_error():
                self.claim()
        self.assertEqual(self.rows(), [])

    # 5. Duplicate dispatch.
    def test_concurrent_threads_produce_exactly_one_claim(self):
        self.register()
        barrier = threading.Barrier(8)

        def attempt():
            barrier.wait()
            try:
                return self.claim()
            except api.ControlConflict as error:
                return str(error)

        with ThreadPoolExecutor(max_workers=8) as pool:
            outcomes = list(pool.map(lambda _: attempt(), range(8)))
        claims = [o for o in outcomes if isinstance(o, api.ControlClaim)]
        self.assertEqual(len(claims), 1)
        self.assertEqual(sorted(o for o in outcomes if isinstance(o, str)), ["control_claim_exists"] * 7)
        self.assertEqual(len(self.rows()), 1)

    def test_processes_with_their_own_generation_are_refused_as_stale(self):
        self.register()
        start = time.time() + 0.5
        children = [
            spawn(directory=str(self.directory), capacity=8, fields=FIELDS, action="claim", start=start)
            for _ in range(4)
        ]
        outcomes = [result(child) for child in children]
        self.assertEqual([o.get("reason") for o in outcomes], ["control_claim_stale_registration"] * 4)
        self.assertEqual(self.rows(), [])

    def test_processes_sharing_the_registrar_generation_produce_exactly_one_claim(self):
        self.register()
        start = time.time() + 0.5
        children = [
            spawn(
                directory=str(self.directory), capacity=8, fields=FIELDS, action="claim",
                start=start, generation=self.store.generation_id,
            )
            for _ in range(4)
        ]
        outcomes = [result(child) for child in children]
        self.assertEqual(sum(1 for o in outcomes if o.get("ok")), 1)
        self.assertEqual(sorted(o.get("reason") for o in outcomes if not o.get("ok")), ["control_claim_exists"] * 3)
        self.assertEqual(len(self.rows()), 1)

    # 6. Fence races.
    def test_claim_and_cancel_threads_never_leave_a_claim_after_a_cancel(self):
        store = api.ControlStore(Path(self.temp.name) / "races", capacity=1024)
        orderings = set()
        for _ in range(200):
            op = str(uuid.uuid4())
            self.register(store=store, operation_id=op)
            barrier = threading.Barrier(2)

            def do_claim():
                barrier.wait()
                try:
                    return self.claim(store=store, operation_id=op)
                except api.ControlConflict as error:
                    return str(error)

            def do_cancel():
                barrier.wait()
                return store.request_cancel(OWNER, JOB, op)

            with ThreadPoolExecutor(max_workers=2) as pool:
                claimed, cancelled = pool.submit(do_claim), pool.submit(do_cancel)
                claimed, cancelled = claimed.result(), cancelled.result()
            self.assertEqual(cancelled.state, "cancel_requested")
            if isinstance(claimed, api.ControlClaim):
                orderings.add("claimed_then_cancelled")
                self.assertEqual(store.lookup_cancellation(OWNER, JOB, op).claim_state, "claimed")
                with self.assertRaises(api.ControlConflict) as caught:
                    store.verify_claim(claimed, settings_sha256=FIELDS["settings_sha256"], deadline=later())
                self.assertEqual(str(caught.exception), "control_claim_cancel_requested")
            else:
                orderings.add("refused")
                self.assertEqual(claimed, "control_claim_cancel_requested")
                self.assertFalse(any(row[2] == op for row in self.rows(store.path)))
            if len(orderings) == 2:
                break
        self.assertEqual(orderings, {"claimed_then_cancelled", "refused"})

    def test_claim_and_cancel_processes_never_leave_a_claim_after_a_cancel(self):
        directory = Path(self.temp.name) / "process-races"
        store = api.ControlStore(directory, capacity=1024)
        orderings = set()
        for _ in range(60):
            op = str(uuid.uuid4())
            fields = FIELDS | {"operation_id": op}
            start = time.time() + 0.3
            claimer = spawn(directory=str(directory), capacity=1024, fields=fields, action="claim", register=True, start=start)
            canceller = spawn(directory=str(directory), capacity=1024, fields=fields, action="cancel", start=start)
            claimed, cancelled = result(claimer), result(canceller)
            self.assertTrue(cancelled.get("ok"), cancelled)
            status = store.lookup_status(OWNER, JOB, op)
            self.assertEqual(status.state, "cancel_requested")
            if claimed.get("ok"):
                orderings.add("claimed_then_cancelled")
                # Claimed by another process: no one here can vouch for it.
                self.assertEqual(status.claim_state, "unknown")
            else:
                orderings.add("refused")
                self.assertIn(claimed["reason"], {
                    "control_claim_cancel_requested", "control_registration_cancel_requested",
                })
                self.assertEqual(status.claim_state, "none")
                self.assertFalse(any(row[2] == op for row in self.rows(store.path)))
            if len(orderings) == 2:
                break
        self.assertEqual(orderings, {"claimed_then_cancelled", "refused"})

    # 7. Verify.
    def test_verify_claim_fences(self):
        self.register()
        claim = self.claim()
        settings = FIELDS["settings_sha256"]
        self.assertIsNone(self.store.verify_claim(claim, settings_sha256=settings, deadline=later()))
        with self.assertRaises(api.ControlConflict) as caught:
            self.store.verify_claim(claim, settings_sha256="f" * 64, deadline=later())
        self.assertEqual(str(caught.exception), "control_claim_settings_changed")
        for value in (True, float("nan"), None):
            with self.assertRaises(ValueError):
                self.store.verify_claim(claim, settings_sha256=settings, deadline=value)
        with mock.patch.object(self.store, "_exclusion", side_effect=AssertionError("locked")):
            with self.assertRaises(TimeoutError):
                self.store.verify_claim(claim, settings_sha256=settings, deadline=time.monotonic() - 1)
        base = time.monotonic()
        with mock.patch.object(api, "_now", side_effect=[base, base + 100]):
            with self.assertRaises(TimeoutError) as caught:
                self.store.verify_claim(claim, settings_sha256=settings, deadline=base + 10)
        self.assertIs(type(caught.exception), TimeoutError)
        with self.assertRaises(api.ControlConflict) as caught:
            self.fresh().verify_claim(claim, settings_sha256=settings, deadline=later())
        self.assertEqual(str(caught.exception), "control_claim_foreign_generation")
        for field in ("worker_id", "request_id"):
            forged = api.ControlClaim(**{**claim.__dict__, field: str(uuid.uuid4())})
            with self.assertRaises(api.ControlConflict) as caught:
                self.store.verify_claim(forged, settings_sha256=settings, deadline=later())
            self.assertEqual(str(caught.exception), "control_claim_not_owner")
        forged = api.ControlClaim(**{**claim.__dict__, "store_generation_id": str(uuid.uuid4())})
        with self.assertRaises(api.ControlConflict) as caught:
            self.store.verify_claim(forged, settings_sha256=settings, deadline=later())
        self.assertEqual(str(caught.exception), "control_claim_foreign_generation")
        with self.assertRaises(ValueError):
            self.store.verify_claim(claim.__dict__, settings_sha256=settings, deadline=later())

        class Subclaim(api.ControlClaim):
            pass

        with self.assertRaises(ValueError):
            self.store.verify_claim(Subclaim(**claim.__dict__), settings_sha256=settings, deadline=later())
        with self.assertRaises(ValueError):
            self.store.verify_claim(claim, settings_sha256="not-a-digest", deadline=later())
        self.store.request_cancel(OWNER, JOB, OP)
        with self.assertRaises(api.ControlConflict) as caught:
            self.store.verify_claim(claim, settings_sha256=settings, deadline=later())
        self.assertEqual(str(caught.exception), "control_claim_cancel_requested")

    def test_verify_claim_refuses_rows_changed_outside_the_store(self):
        self.register()
        claim = self.claim()
        settings = FIELDS["settings_sha256"]
        with support.fixture_database(self.store.path) as connection:
            connection.execute("UPDATE claims SET worker_id=?", (str(uuid.uuid4()),))
        with self.assertRaises(api.ControlConflict) as caught:
            self.store.verify_claim(claim, settings_sha256=settings, deadline=later())
        self.assertEqual(str(caught.exception), "control_claim_not_owner")
        with support.fixture_database(self.store.path) as connection:
            connection.execute("UPDATE claims SET operation_id=?", (str(uuid.uuid4()),))
        with self.exact_store_error():
            self.store.verify_claim(claim, settings_sha256=settings, deadline=later())

    def test_verify_claim_refuses_a_missing_registration(self):
        self.register()
        claim = self.claim()
        with support.fixture_database(self.store.path) as connection:
            connection.execute("UPDATE registrations SET operation_id=?", (str(uuid.uuid4()),))
        with self.exact_store_error():
            self.store.verify_claim(claim, settings_sha256=FIELDS["settings_sha256"], deadline=later())

    # 8. Worker exit.
    def test_a_worker_that_exits_after_claiming_leaves_an_unknown_unreplayable_claim(self):
        self.register()
        child = spawn(
            directory=str(self.directory), capacity=8, fields=FIELDS, action="claim",
            generation=self.store.generation_id, exit_after=True,
        )
        outcome = result(child)
        self.assertTrue(outcome["ok"], outcome)
        restarted = self.fresh()
        self.assertEqual(restarted.lookup_status(OWNER, JOB, OP).claim_state, "unknown")
        with self.assertRaises(api.ControlConflict) as caught:
            self.claim(store=restarted)
        self.assertEqual(str(caught.exception), "control_claim_exists")
        dead = api.ControlClaim(*outcome["claim"])
        with self.assertRaises(api.ControlConflict) as caught:
            restarted.verify_claim(dead, settings_sha256=FIELDS["settings_sha256"], deadline=later())
        self.assertEqual(str(caught.exception), "control_claim_foreign_generation")

    def test_a_worker_that_exits_inside_the_claim_leaves_the_store_unavailable_unchanged(self):
        self.register()
        before = self.store.path.read_bytes()
        child = spawn(
            directory=str(self.directory), capacity=8, fields=FIELDS, action="claim",
            generation=self.store.generation_id, exit_inside_claim=True,
        )
        self.assertEqual(result(child).get("exited"), 0)
        self.assertTrue(Path(str(self.store.path) + "-journal").exists())
        self.assertEqual(self.store.path.read_bytes(), before)
        for call in (
            lambda: self.store.lookup_status(OWNER, JOB, OP),
            lambda: self.claim(),
            lambda: self.fresh(),
        ):
            with self.exact_store_error():
                call()
        self.assertEqual(self.store.path.read_bytes(), before)

    # 9. Status (store level; the router is tested below).
    def test_status_reports_the_claim_state_with_the_original_bindings(self):
        registered = self.register()
        self.assertEqual((registered.claim_state, registered.version), ("none", 3))
        self.assertEqual(self.store.lookup_status(OWNER, JOB, OP), registered)
        claim = self.claim()
        repeat = self.register()
        self.assertIs(type(repeat), api.ControlClaimStatus)
        self.assertEqual(
            (repeat.generation_id, repeat.request_sha256, repeat.model_id, repeat.settings_sha256),
            (registered.generation_id, registered.request_sha256, registered.model_id, registered.settings_sha256),
        )
        self.assertEqual((repeat.state, repeat.claim_state), ("claimed", "claimed"))
        self.assertEqual(repeat.claim_store_generation_id, claim.store_generation_id)
        self.assertEqual(self.store.lookup(OWNER, JOB, OP).claim_state, "claimed")
        for record in (repeat, self.store.lookup_status(OWNER, JOB, OP)):
            self.assertFalse({"worker_id", "request_id"} & set(record.__dict__))
        cancelled = self.store.request_cancel(OWNER, JOB, OP)
        self.assertEqual((cancelled.claim_state, cancelled.claim_store_generation_id), ("claimed", claim.store_generation_id))
        self.assertEqual(self.store.lookup_status(OWNER, JOB, OP), cancelled)
        self.assertEqual(self.store.lookup_cancellation(OWNER, JOB, OP), cancelled)
        restarted = self.fresh()
        after = restarted.lookup_status(OWNER, JOB, OP)
        self.assertEqual((after.state, after.claim_state), ("cancel_requested", "unknown"))
        live = str(uuid.uuid4())
        self.register(operation_id=live)
        self.claim(operation_id=live)
        repeated = restarted.register(**(FIELDS | {"operation_id": live}))
        self.assertEqual((type(repeated), repeated.state, repeated.claim_state), (api.ControlClaimStatus, "unknown", "unknown"))
        unclaimed = self.store.request_cancel(OWNER, JOB, str(uuid.uuid4()))
        self.assertEqual((unclaimed.claim_state, unclaimed.claim_store_generation_id), ("none", None))

    def test_a_claim_without_its_registration_is_unknown_state(self):
        self.register()
        self.claim()
        with support.fixture_database(self.store.path) as connection:
            connection.execute("UPDATE registrations SET operation_id=?", (str(uuid.uuid4()),))
        with self.exact_store_error():
            self.store.lookup_status(OWNER, JOB, OP)
        before = self.store.path.read_bytes()
        with self.exact_store_error():
            self.register()
        self.assertEqual(self.store.path.read_bytes(), before)

    # 10. Schema and durability.
    def test_a_version_2_store_is_refused_and_left_unchanged(self):
        directory = Path(self.temp.name) / "old"
        directory.mkdir(mode=0o700)
        path = directory / "registrations.sqlite3"
        with closing(REAL_CONNECT(path)) as connection:
            connection.execute(api.METADATA_SCHEMA)
            connection.execute("INSERT INTO metadata VALUES ('capacity', 8)")
            connection.execute(api.REGISTRATION_SCHEMA)
            connection.execute(api.CANCELLATION_SCHEMA)
            connection.execute("PRAGMA user_version = 2")
            connection.commit()
        os.chmod(path, 0o600)
        before = path.read_bytes()
        with self.exact_store_error():
            api.ControlStore(directory, capacity=8)
        self.assertEqual(path.read_bytes(), before)

    def test_an_extra_index_or_a_malformed_claim_row_is_refused(self):
        self.register()
        self.claim()
        with support.fixture_database(self.store.path) as connection:
            connection.execute("UPDATE claims SET request_id='not-a-uuid'")
        with self.exact_store_error():
            self.store.lookup_status(OWNER, JOB, OP)
        with support.fixture_database(self.store.path) as connection:
            connection.execute("CREATE INDEX synthetic_extra ON claims(request_id)")
        with self.exact_store_error():
            self.store.lookup(OWNER, JOB, OP)

    def test_write_connections_use_fullfsync_and_reads_do_not(self):
        self.register()
        with self.connections() as log:
            self.claim()
        self.assertIn("PRAGMA fullfsync = ON", log)
        self.assertIn(("effective fullfsync", 1), log)
        with self.connections() as log:
            self.store.lookup_status(OWNER, JOB, OP)
        self.assertNotIn("PRAGMA fullfsync = ON", log)

    # 11. Capacity.
    def test_a_claim_at_full_capacity_adds_no_identity(self):
        small_dir = Path(self.temp.name) / "small"
        small = api.ControlStore(small_dir, capacity=2)
        small.register(**FIELDS)
        small.register(**(FIELDS | {"operation_id": str(uuid.uuid4())}))
        self.assertIs(type(small.claim(**(FIELDS | {"deadline": later()}))), api.ControlClaim)
        with self.assertRaises(api.ControlCapacity):
            small.register(**(FIELDS | {"operation_id": str(uuid.uuid4())}))
        self.assertEqual(small.request_cancel(OWNER, JOB, OP).claim_state, "claimed")


class ClaimRouteTests(unittest.TestCase):
    def setUp(self):
        self.base = http.ControlRouteTests(
            methodName="test_exact_authenticated_record_is_delivered_without_engine_authority"
        )
        self.base.setUp()
        self.addCleanup(self.base.doCleanups)
        self.store = self.base.state.prism_control_store

    def test_routes_report_claim_state_and_never_the_trusted_handle(self):
        body = self.base.get().json()
        self.assertEqual((body["state"], body["claim_state"], body["version"]), ("registered", "none", 3))
        claim = self.store.claim(**(http.REGISTRATION | {"deadline": later()}))
        body = self.base.get().json()
        self.assertEqual((body["state"], body["claim_state"]), ("claimed", "claimed"))
        self.assertEqual(body["store_generation_id"], self.base.record.generation_id)
        self.assertEqual(body["claim_store_generation_id"], claim.store_generation_id)
        self.assertEqual(body["request_sha256"], http.REGISTRATION["request_sha256"])
        for leaked in ("owner_id", "worker_id", "request_id"):
            self.assertNotIn(leaked, body)
        self.assertNotIn(claim.worker_id, json.dumps(body))
        self.assertNotIn(claim.request_id, json.dumps(body))
        self.assertIs(body["engine_generation_verified"], False)
        with self.assertRaises(TypeError):
            routes.registration_payload(claim)
        posted = self.base.client.post(
            http.PATH + "/cancel", headers={"Authorization": "Bearer " + http.KEY}
        ).json()
        self.assertEqual((posted["state"], posted["claim_state"]), ("cancel_requested", "claimed"))
        self.assertEqual(self.base.get().json(), posted)
        self.base.state.prism_control_store = route_store.ControlStore(self.store.path.parent, capacity=1024)
        restarted = self.base.get().json()
        self.assertEqual((restarted["state"], restarted["claim_state"]), ("cancel_requested", "unknown"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
