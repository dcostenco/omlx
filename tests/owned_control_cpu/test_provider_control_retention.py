# SPDX-License-Identifier: Apache-2.0
"""Slice 28: control store retention of UUIDv7 operation identities.

A UUIDv7 operation identity is retired one horizon after it was minted,
behind a one-way mark stored with the data; any other identity is kept for
the store's life. A claim without a terminal is never retired.
"""

from contextlib import closing
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import time
import unittest
from unittest import mock
import uuid

import test_provider_control_store as support

api = support.api
OWNER, JOB, FIELDS = support.OWNER, support.JOB, support.FIELDS
HORIZON_MS = api.RETENTION_SECONDS * 1000
T0 = 1_800_000_000_000  # a fixed wall-clock instant, in milliseconds


def uuid7(ms):
    """A canonical RFC 9562 UUIDv7 minted at `ms`."""
    rand = int.from_bytes(os.urandom(10), "big")
    value = ((ms & ((1 << 48) - 1)) << 80) | (7 << 76) | (((rand >> 62) & 0xFFF) << 64) \
        | (0b10 << 62) | (rand & ((1 << 62) - 1))
    return str(uuid.UUID(int=value))


class RetentionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="prism-control-retention-")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / "control"
        self.now = T0
        patcher = mock.patch.object(api, "_wall_ms", lambda: self.now)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.store = api.ControlStore(self.directory, capacity=8)

    def register(self, op, store=None):
        return (store or self.store).register(**(FIELDS | {"operation_id": op}))

    def claim(self, op):
        self.register(op)
        return self.store.claim(**(FIELDS | {"operation_id": op, "deadline": time.monotonic() + 10}))

    def rows(self, op):
        with closing(sqlite3.connect(f"file:{self.store.path}?mode=ro", uri=True)) as connection:
            return {
                table: connection.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE operation_id=?", (op,)
                ).fetchone()[0]
                for table in ("registrations", "cancellation_requests", "claims", "terminals")
            }

    def mark(self):
        with closing(sqlite3.connect(f"file:{self.store.path}?mode=ro", uri=True)) as connection:
            return connection.execute(
                "SELECT value FROM metadata WHERE key='retired_through_ms'"
            ).fetchone()[0]

    def test_uuid7_time_and_other_identities(self):
        self.assertEqual(api._uuid7_ms(uuid7(123456789)), 123456789)
        self.assertIsNone(api._uuid7_ms(str(uuid.uuid4())))
        self.assertIsNone(api._uuid7_ms(support.OP))
        # Version 7 bits under a non-RFC variant are not a UUIDv7.
        microsoft = uuid.UUID(int=(uuid.UUID(uuid7(5)).int & ~(0b11 << 62)) | (0b11 << 62))
        self.assertIsNone(api._uuid7_ms(str(microsoft)))

    def populate_and_retire(self):
        minted = T0 - 10
        ops = {name: uuid7(minted) for name in (
            "registered", "cancelled", "finished", "unresolved")}
        self.register(ops["registered"])
        self.store.request_cancel(OWNER, JOB, ops["cancelled"])
        finished = self.claim(ops["finished"])
        self.store.finish(finished, outcome="not_started", evidence=("not_admitted",))
        self.claim(ops["unresolved"])
        self.store.request_cancel(OWNER, JOB, ops["unresolved"])
        ops["claimed"] = uuid7(minted)
        self.claim(ops["claimed"])
        ops["v4"] = str(uuid.uuid4())
        self.register(ops["v4"])
        self.now = T0 + HORIZON_MS + 1000
        ops["new"] = uuid7(self.now)
        self.register(ops["new"])  # this write retires
        return ops

    def test_a_write_retires_finished_and_unclaimed_identities_and_keeps_the_rest(self):
        ops = self.populate_and_retire()
        empty = dict(registrations=0, cancellation_requests=0, claims=0, terminals=0)
        for name in ("registered", "cancelled", "finished"):
            with self.subTest(name):
                self.assertEqual(self.rows(ops[name]), empty)
                self.assertIsNone(self.store.lookup(OWNER, JOB, ops[name]))
                self.assertIsNone(self.store.lookup_status(OWNER, JOB, ops[name]))
        # A claim without a terminal keeps every row, its cancellation included.
        self.assertEqual(self.rows(ops["unresolved"]),
                         dict(registrations=1, cancellation_requests=1, claims=1, terminals=0))
        self.assertEqual(self.store.lookup_status(OWNER, JOB, ops["unresolved"]).state,
                         "cancel_requested")
        for name in ("v4", "new"):
            self.assertIsNotNone(self.store.lookup(OWNER, JOB, ops[name]))
        self.assertEqual(self.mark(), self.now - HORIZON_MS)

    def test_a_kept_claim_still_cancels_and_registers_again(self):
        ops = self.populate_and_retire()
        # Kept claims answer as before retention, never as expired.
        self.assertEqual(self.register(ops["claimed"]).claim_state, "claimed")
        self.assertEqual(self.store.request_cancel(OWNER, JOB, ops["claimed"]).claim_state, "claimed")
        self.assertEqual(self.store.request_cancel(OWNER, JOB, ops["unresolved"]).claim_state, "claimed")
        with self.assertRaises(api.ControlConflict) as caught:
            self.register(ops["unresolved"])  # registration after a cancel, as today
        self.assertEqual(str(caught.exception), "control_registration_cancel_requested")

    def test_a_retired_identity_is_refused_on_register_and_cancel(self):
        ops = self.populate_and_retire()
        for name in ("registered", "cancelled", "finished"):
            with self.subTest(name):
                with self.assertRaises(api.ControlExpired):
                    self.register(ops[name])
                with self.assertRaises(api.ControlExpired):
                    self.store.request_cancel(OWNER, JOB, ops[name])
                with self.assertRaises(api.ControlConflict):
                    self.store.claim(**(FIELDS | {"operation_id": ops[name],
                                                  "deadline": time.monotonic() + 10}))
        self.assertTrue(issubclass(api.ControlExpired, api.ControlCapacity))

    def test_a_never_registered_old_identity_is_refused_on_an_empty_store(self):
        with self.assertRaises(api.ControlExpired):
            self.register(uuid7(T0 - HORIZON_MS))
        with self.assertRaises(api.ControlExpired):
            self.store.request_cancel(OWNER, JOB, uuid7(T0 - HORIZON_MS - 5))
        self.assertIsNotNone(self.register(uuid7(T0 - HORIZON_MS + 1000)))

    def test_the_mark_persists_and_never_decreases(self):
        self.register(uuid7(T0))
        retired = self.mark()
        self.assertEqual(retired, T0 - HORIZON_MS)
        self.now = T0 - 10 * HORIZON_MS  # the clock moves far back
        reopened = api.ControlStore(self.directory, capacity=8)
        with self.assertRaises(api.ControlExpired):
            self.register(uuid7(retired - 5), store=reopened)
        self.register(uuid7(T0), store=reopened)
        self.assertEqual(self.mark(), retired)

    def test_a_stored_identity_stays_readable_when_the_clock_moves_back(self):
        op = uuid7(T0)
        self.register(op)
        self.now = T0 - 2 * HORIZON_MS
        self.assertIsNotNone(self.store.lookup(OWNER, JOB, op))
        self.assertEqual(self.store.lookup_status(OWNER, JOB, op).state, "registered")
        self.register(uuid7(T0 + 1))  # a write under the earlier clock keeps it
        self.assertIsNotNone(self.store.lookup(OWNER, JOB, op))

    def test_a_full_store_accepts_new_work_after_retirement(self):
        old = [uuid7(T0 - 10) for _ in range(8)]
        for op in old:
            self.register(op)
        with self.assertRaises(api.ControlCapacity):
            self.register(uuid7(T0))
        self.now = T0 + HORIZON_MS
        self.register(uuid7(self.now))
        self.assertTrue(all(self.store.lookup(OWNER, JOB, op) is None for op in old))

    def test_an_old_claim_finishes_and_its_terminal_is_read_until_the_next_write(self):
        op = uuid7(T0)
        claim = self.claim(op)
        self.now = T0 + HORIZON_MS + 1000
        self.assertIsNotNone(self.store.lookup_status(OWNER, JOB, op))  # kept: no terminal yet
        record = self.store.finish(claim, outcome="not_started", evidence=("not_admitted",))
        self.assertEqual(record.state, "not_started")
        self.assertEqual(self.store.lookup_status(OWNER, JOB, op).state, "not_started")
        self.register(uuid7(self.now))  # R4: past the horizon, the next write retires it
        self.assertIsNone(self.store.lookup_status(OWNER, JOB, op))

    def test_a_version_4_file_or_a_missing_mark_is_refused(self):
        for change in ("PRAGMA user_version = 4",
                       "DELETE FROM metadata WHERE key='retired_through_ms'",
                       "UPDATE metadata SET value=-1 WHERE key='retired_through_ms'"):
            with self.subTest(change):
                directory = Path(self.temp.name) / f"control-{abs(hash(change))}"
                store = api.ControlStore(directory, capacity=8)
                with support.fixture_database(store.path) as connection:
                    connection.execute(change)
                with self.assertRaises(api.ControlStoreError):
                    store.lookup(OWNER, JOB, support.OP)


class RetentionRouteTests(unittest.TestCase):
    def test_a_retired_identity_reads_404_and_cancels_409(self):
        import test_provider_control_routes as http

        case = http.ControlRouteTests("test_exact_authenticated_record_is_delivered_without_engine_authority")
        case.setUp()
        self.addCleanup(case.doCleanups)
        store = case.state.prism_control_store
        module = sys.modules[type(store).__module__]
        now = [T0]
        patcher = mock.patch.object(module, "_wall_ms", lambda: now[0])
        patcher.start()
        self.addCleanup(patcher.stop)
        old = uuid7(T0 - 10)
        store.register(**(http.REGISTRATION | {"operation_id": old}))
        now[0] = T0 + HORIZON_MS + 1000
        store.register(**(http.REGISTRATION | {"operation_id": uuid7(now[0])}))
        path = f"/prism-control/v1/jobs/{http.JOB}/operations/{old}"
        headers = {"Authorization": "Bearer " + http.KEY}
        self.assertEqual(case.client.get(path, headers=headers).status_code, 404)
        self.assertEqual(case.client.post(path + "/cancel", headers=headers).status_code, 409)


if __name__ == "__main__":
    unittest.main()
