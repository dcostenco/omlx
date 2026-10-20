# SPDX-License-Identifier: Apache-2.0
"""Slice 27.1 durable terminal record: the real store, no engines."""

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
import test_provider_control_claim as claims

api = support.api
OWNER, JOB, OP = support.OWNER, support.JOB, support.OP
FIELDS = support.FIELDS
REAL_CONNECT = sqlite3.connect
SETTINGS = FIELDS["settings_sha256"]

SHAPES = [
    ("not_started", ("not_admitted",)),
    ("not_started", ("admission_refused", "stream_synchronized")),
    ("completed", ("finished_output", "absent_proved", "stream_synchronized")),
    ("completed", ("finished_output", "deferred_cleanup", "stream_synchronized")),
    ("failed", ("error_output", "absent_proved", "stream_synchronized")),
    ("cancelled", ("waiting_cleaned", "stream_synchronized")),
    ("cancelled", ("running_cleaned", "stream_synchronized")),
]

CHILD = r"""
import importlib.util, json, os, sys, time
from pathlib import Path
spec = importlib.util.spec_from_file_location(
    "terminal_child_store",
    Path(os.environ["OMLX_CONTROL_ROOT"]) / "omlx/api/prism_control_store.py",
)
api = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = api
spec.loader.exec_module(api)
args = json.loads(sys.argv[1])
store = api.ControlStore(args["directory"], capacity=8)
if args.get("generation"):
    store._generation_id = args["generation"]
while time.time() < args["start"]:
    time.sleep(0.0005)
try:
    record = store.finish(api.ControlClaim(*args["claim"]), outcome=args["outcome"], evidence=tuple(args["evidence"]))
    out = {"ok": True, "state": record.state}
except Exception as error:
    out = {"ok": False, "type": type(error).__name__, "reason": str(error)}
print(json.dumps(out), flush=True)
"""


class TerminalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="prism-control-terminal-")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / "control"
        self.store = api.ControlStore(self.directory, capacity=8)

    def claimed(self, op=OP, store=None):
        store = store or self.store
        store.register(**(FIELDS | {"operation_id": op}))
        return store.claim(**(FIELDS | {"operation_id": op, "deadline": time.monotonic() + 10}))

    def rows(self):
        with closing(REAL_CONNECT(f"file:{self.store.path}?mode=ro", uri=True)) as connection:
            return connection.execute("SELECT * FROM terminals").fetchall()

    def fresh(self):
        return api.ControlStore(self.directory, capacity=8)

    @contextmanager
    def refused(self, kind, reason):
        before = self.store.path.read_bytes()
        with self.assertRaises(kind) as caught:
            yield
        self.assertIs(type(caught.exception), kind)
        self.assertEqual(str(caught.exception), reason)
        self.assertEqual(self.store.path.read_bytes(), before)

    def test_every_accepted_shape_is_recorded_durably_and_survives_a_restart(self):
        for index, (outcome, evidence) in enumerate(SHAPES):
            with self.subTest(outcome=outcome, evidence=evidence):
                op = str(uuid.UUID(int=index + 1, version=4))
                claim = self.claimed(op)
                record = self.store.finish(claim, outcome=outcome, evidence=evidence)
                self.assertIs(type(record), api.ControlTerminalStatus)
                self.assertEqual((record.state, record.claim_state, record.backend_stopped), (outcome, "finished", True))
                self.assertEqual(record.evidence, tuple(t for t in api.EVIDENCE_ORDER if t in evidence))
                self.assertEqual(record.claim_store_generation_id, claim.store_generation_id)
                self.assertFalse(record.cancel_requested)
                restarted = self.fresh().lookup_status(OWNER, JOB, op)
                self.assertEqual(restarted, record)
        self.assertEqual(len(self.rows()), len(SHAPES))

    def test_refusals_write_nothing_and_name_their_reason(self):
        claim = self.claimed()
        for outcome, evidence in [
            ("completed", ("not_admitted",)),
            ("cancelled", ("running_cleaned",)),
            ("not_started", ("not_admitted", "not_admitted")),
            ("not_started", ("not_admitted", "stream_synchronized")),
            ("unknown", ("not_admitted",)),
            ("completed", ["finished_output", "deferred_cleanup", "stream_synchronized"]),
            ("cancelled", ("running_cleaned", "invented_fact")),
            # An error finish leaves no cache store, so deferred cleanup is no proof of it.
            ("failed", ("error_output", "deferred_cleanup", "stream_synchronized")),
            ("completed", ("finished_output", "stream_synchronized")),
        ]:
            with self.subTest(outcome=outcome, evidence=evidence), self.assertRaises(ValueError):
                self.store.finish(claim, outcome=outcome, evidence=evidence)
        with self.assertRaises(ValueError):
            self.store.finish(claim.__dict__, outcome="not_started", evidence=("not_admitted",))
        with self.refused(api.ControlConflict, "control_claim_foreign_generation"):
            self.fresh().finish(claim, outcome="not_started", evidence=("not_admitted",))
        for field in ("worker_id", "request_id"):
            forged = api.ControlClaim(**{**claim.__dict__, field: str(uuid.uuid4())})
            with self.refused(api.ControlConflict, "control_claim_not_owner"):
                self.store.finish(forged, outcome="not_started", evidence=("not_admitted",))
        unclaimed = api.ControlClaim(OWNER, JOB, str(uuid.uuid4()), claim.store_generation_id, str(uuid.uuid4()), str(uuid.uuid4()))
        with self.refused(api.ControlStoreError, "control_store_unavailable"):
            self.store.finish(unclaimed, outcome="not_started", evidence=("not_admitted",))
        self.assertEqual(self.rows(), [])
        first = self.store.finish(claim, outcome="cancelled", evidence=("running_cleaned", "stream_synchronized"))
        with self.refused(api.ControlConflict, "control_terminal_exists"):
            self.store.finish(claim, outcome="completed", evidence=("finished_output", "deferred_cleanup", "stream_synchronized"))
        with self.refused(api.ControlConflict, "control_terminal_exists"):
            self.store.finish(claim, outcome="cancelled", evidence=("waiting_cleaned", "stream_synchronized"))
        # An identical retry returns the first terminal and writes nothing.
        before = self.store.path.read_bytes()
        self.assertEqual(self.store.finish(claim, outcome="cancelled", evidence=("stream_synchronized", "running_cleaned")), first)
        self.assertEqual(self.store.path.read_bytes(), before)
        self.assertEqual(len(self.rows()), 1)

    def test_a_proof_is_recorded_after_the_admission_deadline(self):
        self.store.register(**FIELDS)
        claim = self.store.claim(**(FIELDS | {"deadline": time.monotonic() + 0.05}))
        time.sleep(0.1)
        with self.assertRaises(TimeoutError):
            self.store.verify_claim(claim, settings_sha256=SETTINGS, deadline=time.monotonic() - 1)
        record = self.store.finish(claim, outcome="cancelled", evidence=("running_cleaned", "stream_synchronized"))
        self.assertEqual(record.state, "cancelled")

    def test_failures_around_commit(self):
        claim = self.claimed()
        log = []
        with mock.patch.object(api.sqlite3, "connect", lambda target, *a, **k: claims.Recorder(REAL_CONNECT(target, *a, **k), log, fail_commit=True) if str(target) == str(self.store.path) else REAL_CONNECT(target, *a, **k)):
            with self.assertRaises(api.ControlStoreError) as caught:
                self.store.finish(claim, outcome="not_started", evidence=("not_admitted",))
        self.assertIs(type(caught.exception), api.ControlStoreError)
        self.assertEqual(self.rows(), [])
        # The retry writes it.
        self.assertEqual(self.store.finish(claim, outcome="not_started", evidence=("not_admitted",)).state, "not_started")
        other = self.claimed(str(uuid.uuid4()))
        with mock.patch.object(api.sqlite3, "connect", lambda target, *a, **k: claims.Recorder(REAL_CONNECT(target, *a, **k), log, fail_close_after_commit=True) if str(target) == str(self.store.path) else REAL_CONNECT(target, *a, **k)):
            record = self.store.finish(other, outcome="not_started", evidence=("not_admitted",))
        self.assertEqual(record.state, "not_started")
        self.assertEqual(len(self.rows()), 2)

    def test_status_precedence_cancel_register_and_verify(self):
        claim = self.claimed()
        self.store.request_cancel(OWNER, JOB, OP)
        self.assertEqual(self.store.lookup_status(OWNER, JOB, OP).state, "cancel_requested")
        record = self.store.finish(claim, outcome="cancelled", evidence=("running_cleaned", "stream_synchronized"))
        self.assertTrue(record.cancel_requested)
        for view in (
            self.store.lookup_status(OWNER, JOB, OP),
            self.store.lookup_cancellation(OWNER, JOB, OP),
            self.store.register(**FIELDS),
            self.store.request_cancel(OWNER, JOB, OP),
            self.fresh().lookup_status(OWNER, JOB, OP),
        ):
            self.assertEqual((type(view), view.state, view.claim_state, view.cancel_requested), (api.ControlTerminalStatus, "cancelled", "finished", True))
        self.assertEqual(self.store.lookup(OWNER, JOB, OP).claim_state, "finished")
        with self.assertRaises(api.ControlConflict) as caught:
            self.store.register(**(FIELDS | {"model_id": "synthetic-other-model"}))
        self.assertEqual(str(caught.exception), "control_registration_conflict")
        with self.assertRaises(api.ControlConflict) as caught:
            self.store.verify_claim(claim, settings_sha256=SETTINGS, deadline=time.monotonic() + 5)
        self.assertEqual(str(caught.exception), "control_claim_finished")
        # A finished identity without a cancellation: the first cancel records
        # its tombstone and reports the outcome.
        other = self.claimed(str(uuid.uuid4()))
        self.store.finish(other, outcome="completed", evidence=("finished_output", "deferred_cleanup", "stream_synchronized"))
        self.assertIsNone(self.store.lookup_cancellation(OWNER, JOB, other.operation_id))
        # A repeated registration of a finished, uncancelled operation reports
        # its outcome, never a claim without stop proof.
        repeat = self.store.register(**(FIELDS | {"operation_id": other.operation_id}))
        self.assertEqual((type(repeat), repeat.state, repeat.cancel_requested), (api.ControlTerminalStatus, "completed", False))
        late = self.store.request_cancel(OWNER, JOB, other.operation_id)
        self.assertEqual((late.state, late.cancel_requested), ("completed", True))
        self.assertTrue(self.store.lookup_cancellation(OWNER, JOB, other.operation_id).cancel_requested)
        with self.assertRaises(api.ControlConflict) as caught:
            self.store.claim(**(FIELDS | {"operation_id": other.operation_id, "deadline": time.monotonic() + 5}))
        self.assertEqual(str(caught.exception), "control_claim_cancel_requested")

    def test_two_threads_leave_one_terminal(self):
        claim = self.claimed()
        barrier = threading.Barrier(2)

        def attempt(outcome, evidence):
            barrier.wait()
            try:
                return self.store.finish(claim, outcome=outcome, evidence=evidence).state
            except api.ControlConflict as error:
                return str(error)

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = sorted(pool.map(lambda args: attempt(*args), [
                ("cancelled", ("running_cleaned", "stream_synchronized")),
                ("completed", ("finished_output", "deferred_cleanup", "stream_synchronized")),
            ]))
        self.assertEqual(len(self.rows()), 1)
        self.assertIn("control_terminal_exists", results)

    def test_processes_need_the_claiming_generation(self):
        claim = self.claimed()
        args = dict(directory=str(self.directory), claim=list(claim.__dict__.values()), outcome="not_started", evidence=["not_admitted"])
        start = time.time() + 0.5
        plain = [claims.result(subprocess.Popen([sys.executable, "-I", "-c", CHILD, json.dumps(args | {"start": start})], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)) for _ in range(2)]
        self.assertEqual([o.get("reason") for o in plain], ["control_claim_foreign_generation"] * 2)
        start = time.time() + 0.5
        children = [subprocess.Popen([sys.executable, "-I", "-c", CHILD, json.dumps(args | {"start": start, "generation": self.store.generation_id, "outcome": outcome, "evidence": evidence})], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    for outcome, evidence in [("not_started", ["not_admitted"]), ("cancelled", ["running_cleaned", "stream_synchronized"])]]
        outcomes = [claims.result(child) for child in children]
        self.assertEqual(sum(1 for o in outcomes if o.get("ok")), 1)
        self.assertEqual([o.get("reason") for o in outcomes if not o.get("ok")], ["control_terminal_exists"])
        self.assertEqual(len(self.rows()), 1)

    def test_schema_and_row_validation(self):
        claim = self.claimed()
        self.store.finish(claim, outcome="cancelled", evidence=("running_cleaned", "stream_synchronized"))
        for change in (
            "UPDATE terminals SET evidence='finished_output,deferred_cleanup,stream_synchronized'",
            "UPDATE terminals SET backend_stopped=0",
            "UPDATE terminals SET evidence='stream_synchronized,running_cleaned'",
            "UPDATE terminals SET store_generation_id='not-a-uuid'",
        ):
            with self.subTest(change=change):
                with support.fixture_database(self.store.path) as connection:
                    original = connection.execute("SELECT * FROM terminals").fetchone()
                    connection.execute(change)
                with self.assertRaises(api.ControlStoreError) as caught:
                    self.store.lookup_status(OWNER, JOB, OP)
                self.assertIs(type(caught.exception), api.ControlStoreError)
                with support.fixture_database(self.store.path) as connection:
                    connection.execute("UPDATE terminals SET store_generation_id=?, outcome=?, backend_stopped=?, evidence=?", original[3:])
        self.assertEqual(self.store.lookup_status(OWNER, JOB, OP).state, "cancelled")
        # A terminal must agree with its claim: another generation, or no claim, is unknown state.
        with support.fixture_database(self.store.path) as connection:
            connection.execute("UPDATE claims SET store_generation_id=?", (str(uuid.uuid4()),))
        with self.assertRaises(api.ControlStoreError):
            self.store.lookup_status(OWNER, JOB, OP)
        with support.fixture_database(self.store.path) as connection:
            connection.execute("UPDATE claims SET store_generation_id=?", (claim.store_generation_id,))
            connection.execute("UPDATE claims SET operation_id=?", (str(uuid.uuid4()),))
        with self.assertRaises(api.ControlStoreError):
            self.store.lookup_status(OWNER, JOB, OP)
        with support.fixture_database(self.store.path) as connection:
            connection.execute("UPDATE claims SET operation_id=?", (OP,))
        self.assertEqual(self.store.lookup_status(OWNER, JOB, OP).state, "cancelled")
        with support.fixture_database(self.store.path) as connection:
            connection.execute("CREATE INDEX synthetic_extra ON terminals(outcome)")
        with self.assertRaises(api.ControlStoreError):
            self.store.lookup(OWNER, JOB, OP)

    def test_a_version_3_store_is_refused_and_left_unchanged(self):
        directory = Path(self.temp.name) / "old"
        directory.mkdir(mode=0o700)
        path = directory / "registrations.sqlite3"
        with closing(REAL_CONNECT(path)) as connection:
            for statement in (api.METADATA_SCHEMA, api.REGISTRATION_SCHEMA, api.CANCELLATION_SCHEMA, api.CLAIM_SCHEMA):
                connection.execute(statement)
            connection.execute("INSERT INTO metadata VALUES ('capacity', 8)")
            connection.execute("PRAGMA user_version = 3")
            connection.commit()
        os.chmod(path, 0o600)
        before = path.read_bytes()
        with self.assertRaises(api.ControlStoreError):
            api.ControlStore(directory, capacity=8)
        self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
