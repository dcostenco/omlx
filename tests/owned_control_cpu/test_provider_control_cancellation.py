# SPDX-License-Identifier: Apache-2.0
"""Cancellation fences only, with no model/device stop acknowledgment."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sqlite3
import tempfile
import unittest
import uuid

import test_provider_control_store as support
import test_provider_control_routes as http


class CancellationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="prism-control-cancel-")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.store = support.api.ControlStore(self.directory, capacity=2)

    def cancel(self, op=support.OP):
        return self.store.request_cancel(support.OWNER, support.JOB, op)

    def test_unknown_cancel_is_durable_idempotent_and_never_claims_stop(self):
        record = self.cancel()
        self.assertEqual(record.state, "cancel_requested")
        self.assertFalse(record.backend_stopped)
        # Version 3 records carry the slice 26 claim and slice 27 terminal fields.
        self.assertEqual(record.version, 3)
        self.assertEqual(record.claim_state, "none")
        self.assertEqual(self.cancel(), record)
        fresh = support.api.ControlStore(self.directory, capacity=2)
        self.assertEqual(
            fresh.lookup_status(support.OWNER, support.JOB, support.OP), record
        )
        with self.assertRaises(support.api.ControlConflict):
            fresh.register(**support.FIELDS)

    def test_known_cancel_preserves_original_immutable_registration(self):
        registered = self.store.register(**support.FIELDS)
        cancelled = self.cancel()
        self.assertEqual(
            self.store.lookup(support.OWNER, support.JOB, support.OP), registered
        )
        self.assertEqual(
            self.store.lookup_status(support.OWNER, support.JOB, support.OP), cancelled
        )
        with self.assertRaises(support.api.ControlConflict):
            self.store.register(**support.FIELDS)

    def test_known_cancel_is_allowed_at_full_union_capacity(self):
        first = self.store.register(**support.FIELDS)
        other = str(uuid.uuid4())
        self.cancel(other)
        self.assertEqual(self.cancel().state, "cancel_requested")
        self.assertEqual(
            self.store.lookup(support.OWNER, support.JOB, support.OP), first
        )
        with self.assertRaises(support.api.ControlCapacity):
            self.cancel(str(uuid.uuid4()))
        with self.assertRaises(support.api.ControlCapacity):
            self.store.register(
                **(support.FIELDS | dict(operation_id=str(uuid.uuid4())))
            )

    def test_auth_namespace_never_borrows_another_owner_cancellation(self):
        self.cancel()
        self.assertIsNone(
            self.store.lookup_status(support.OTHER_OWNER, support.JOB, support.OP)
        )
        self.assertEqual(
            self.store.register(
                **(support.FIELDS | dict(owner_id=support.OTHER_OWNER))
            ).state,
            "registered",
        )

    def test_register_cancel_race_always_leaves_a_fence_and_no_rebinding(self):
        def register():
            try:
                return self.store.register(**support.FIELDS)
            except support.api.ControlConflict:
                return None

        with ThreadPoolExecutor(max_workers=2) as pool:
            a = pool.submit(register)
            b = pool.submit(self.cancel)
            registered, cancelled = a.result(), b.result()
        self.assertEqual(
            self.store.lookup_status(support.OWNER, support.JOB, support.OP), cancelled
        )
        with self.assertRaises(support.api.ControlConflict):
            self.store.register(**support.FIELDS)
        if registered:
            self.assertEqual(
                self.store.lookup(support.OWNER, support.JOB, support.OP), registered
            )

    def test_older_schema_is_refused_without_database_or_sidecar_recovery(self):
        # Simulate the prior prototype's version while retaining bytes on refusal.
        with support.fixture_database(self.store.path) as connection:
            connection.execute("PRAGMA user_version=1")
        before = self.store.path.read_bytes()
        with self.assertRaises(support.api.ControlStoreError):
            support.api.ControlStore(self.directory, capacity=2)
        self.assertEqual(self.store.path.read_bytes(), before)


class CancellationApiTests(unittest.TestCase):
    def setUp(self):
        self.base = http.ControlRouteTests(
            methodName="test_exact_authenticated_record_is_delivered_without_engine_authority"
        )
        self.base.setUp()
        self.addCleanup(self.base.doCleanups)

    def test_actual_post_records_fence_and_get_delivers_pending_state(self):
        result = self.base.client.post(
            http.PATH + "/cancel",
            headers={"Authorization": "Bearer " + http.KEY},
            json={"backend_stopped": True, "owner_id": "b" * 64, "state": "cancelled"},
        )
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["state"], "cancel_requested")
        for field in [
            "backend_stopped",
            "engine_generation_verified",
            "provider_qualified",
            "production_enabled",
        ]:
            self.assertIs(result.json()[field], False)
        self.assertEqual(self.base.get().json(), result.json())
        self.assertNotIn("owner_id", result.json())

    def test_cancel_requires_strict_key_despite_inference_bypass(self):
        before = self.base.state.prism_control_store.path.read_bytes()
        self.assertEqual(self.base.client.post(http.PATH + "/cancel").status_code, 401)
        self.assertEqual(self.base.state.prism_control_store.path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
