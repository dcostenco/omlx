# SPDX-License-Identifier: Apache-2.0
"""Offline provider registration tests. Never import engines or installed state."""

import concurrent.futures
from contextlib import closing, contextmanager
import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

SOURCE = Path(os.environ["OMLX_CONTROL_ROOT"]) / "omlx/api/prism_control_store.py"
spec = importlib.util.spec_from_file_location("prism_control_store", SOURCE)
api = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = api
spec.loader.exec_module(api)

OWNER = "a" * 64
OTHER_OWNER = "b" * 64
JOB = "11111111-1111-4111-8111-111111111111"
OP = "22222222-2222-4222-8222-222222222222"
DIGEST = "c" * 64
SETTINGS = "d" * 64
MODEL = "synthetic-registration-model"
FIELDS = dict(
    owner_id=OWNER,
    job_id=JOB,
    operation_id=OP,
    request_sha256=DIGEST,
    model_id=MODEL,
    settings_sha256=SETTINGS,
)


@contextmanager
def fixture_database(path, **options):
    with closing(sqlite3.connect(path, **options)) as connection:
        with connection:
            yield connection


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="prism-control-store-")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / "control"
        self.store = api.ControlStore(self.directory, capacity=8)

    def test_registration_survives_fresh_instance_with_original_generation(self):
        record = self.store.register(**FIELDS)
        fresh = api.ControlStore(self.directory, capacity=8)
        self.assertNotEqual(fresh.generation_id, self.store.generation_id)
        self.assertEqual(fresh.lookup(OWNER, JOB, OP), record)
        self.assertEqual(fresh.register(**FIELDS), record)
        self.assertEqual(record.state, "registered")
        self.assertFalse(record.backend_stopped)
        self.assertEqual(record.generation_id, self.store.generation_id)

    def test_identical_registration_is_idempotent_and_records_are_immutable(self):
        record = self.store.register(**FIELDS)
        self.assertEqual(self.store.register(**FIELDS), record)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            record.model_id = "modified"

    def test_every_registration_digest_or_model_conflict_is_refused(self):
        record = self.store.register(**FIELDS)
        for change in [
            dict(request_sha256="e" * 64),
            dict(settings_sha256="e" * 64),
            dict(model_id="other-model"),
        ]:
            with self.subTest(change=change), self.assertRaises(api.ControlConflict):
                self.store.register(**(FIELDS | change))
        self.assertEqual(self.store.lookup(OWNER, JOB, OP), record)

    def test_owner_namespace_does_not_leak_other_records(self):
        first = self.store.register(**FIELDS)
        self.assertIsNone(self.store.lookup(OTHER_OWNER, JOB, OP))
        second = self.store.register(**(FIELDS | dict(owner_id=OTHER_OWNER)))
        self.assertNotEqual(first.owner_id, second.owner_id)
        self.assertEqual(self.store.lookup(OWNER, JOB, OP), first)

    def test_capacity_never_evicts_or_recycles_existing_identities(self):
        import uuid

        first = self.store.register(**FIELDS)
        for _ in range(7):
            self.store.register(**(FIELDS | dict(operation_id=str(uuid.uuid4()))))
        with self.assertRaises(api.ControlCapacity):
            self.store.register(**(FIELDS | dict(operation_id=str(uuid.uuid4()))))
        self.assertEqual(self.store.register(**FIELDS), first)
        self.assertEqual(self.store.lookup(OWNER, JOB, OP), first)
        with self.assertRaises(api.ControlStoreError):
            api.ControlStore(self.directory, capacity=16)

    def test_concurrent_workers_register_same_operation_once(self):
        def register(_):
            fresh = api.ControlStore(self.directory, capacity=8)
            return fresh.register(**FIELDS)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            records = list(pool.map(register, range(24)))
        self.assertTrue(all(record == records[0] for record in records))
        with fixture_database(self.store.path) as connection:
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM registrations").fetchone()[0],
                1,
            )

    def test_cross_process_read_preserves_exact_durable_record(self):
        record = self.store.register(**FIELDS)
        script = """import importlib.util,sys,json,dataclasses
s=importlib.util.spec_from_file_location('prism_control_store',sys.argv[1])
m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
store=m.ControlStore(sys.argv[2],capacity=8)
print(json.dumps(dataclasses.asdict(store.lookup(*sys.argv[3:]))))
"""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(SOURCE),
                str(self.directory),
                OWNER,
                JOB,
                OP,
            ],
            text=True,
            capture_output=True,
            timeout=10,
            check=True,
        )
        self.assertEqual(json.loads(result.stdout), dataclasses.asdict(record))

    def test_private_directory_and_file_permissions_are_required_on_every_read(self):
        self.store.register(**FIELDS)
        for path, mode in [(self.store.path, 0o644), (self.directory, 0o755)]:
            path.chmod(mode)
            try:
                with self.assertRaises(api.ControlStoreError):
                    self.store.lookup(OWNER, JOB, OP)
            finally:
                path.chmod(0o600 if path == self.store.path else 0o700)

    def test_symlinked_directory_and_database_are_refused_without_writing_target(self):
        alias = Path(self.temp.name) / "alias"
        alias.symlink_to(self.directory, target_is_directory=True)
        with self.assertRaises(api.ControlStoreError):
            api.ControlStore(alias, capacity=8)
        other = Path(self.temp.name) / "other"
        other.mkdir(mode=0o700)
        before = self.store.path.read_bytes()
        (other / "registrations.sqlite3").symlink_to(self.store.path)
        with self.assertRaises(api.ControlStoreError):
            api.ControlStore(other, capacity=8)
        self.assertEqual(self.store.path.read_bytes(), before)

    def test_invalid_fields_cannot_create_or_query_an_operation(self):
        for change in [
            dict(owner_id="not-a-hash"),
            dict(job_id="AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA"),
            dict(operation_id="../escape"),
            dict(request_sha256="x" * 64),
            dict(settings_sha256=""),
            dict(model_id=""),
            dict(model_id="x" * 257),
            dict(model_id="model\nheader"),
            dict(model_id=[]),
        ]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.store.register(**(FIELDS | change))
        self.assertIsNone(self.store.lookup(OWNER, JOB, OP))
        with self.assertRaises(ValueError):
            self.store.lookup(OWNER, "invalid", OP)

    def test_unknown_schema_or_corrupt_file_fails_closed(self):
        with fixture_database(self.store.path) as connection:
            connection.execute("PRAGMA user_version = 999")
        with self.assertRaises(api.ControlStoreError):
            self.store.lookup(OWNER, JOB, OP)
        self.store.path.write_bytes(b"synthetic-corrupt-database")
        with self.assertRaises(api.ControlStoreError):
            api.ControlStore(self.directory, capacity=8)

    def test_foreign_trigger_cannot_change_registration_after_an_acknowledged_insert(
        self,
    ):
        with fixture_database(self.store.path) as connection:
            connection.execute(
                """CREATE TRIGGER foreign_registration AFTER INSERT ON registrations
                BEGIN UPDATE registrations SET model_id='foreign-model' WHERE owner_id=NEW.owner_id; END"""
            )
        with self.assertRaises(api.ControlStoreError):
            self.store.register(**FIELDS)

    def test_matching_column_names_without_uniqueness_constraints_are_not_trusted_schema(
        self,
    ):
        directory = Path(self.temp.name) / "foreign-schema"
        directory.mkdir(mode=0o700)
        path = directory / "registrations.sqlite3"
        with fixture_database(path) as connection:
            connection.execute(
                "CREATE TABLE metadata (key TEXT PRIMARY KEY, value INTEGER NOT NULL)"
            )
            connection.execute("INSERT INTO metadata VALUES ('capacity', 8)")
            connection.execute("""CREATE TABLE registrations (
                owner_id TEXT, job_id TEXT, operation_id TEXT, generation_id TEXT,
                request_sha256 TEXT, model_id TEXT, settings_sha256 TEXT)""")
            connection.execute("PRAGMA user_version=1")
        path.chmod(0o600)
        with self.assertRaises(api.ControlStoreError):
            api.ControlStore(directory, capacity=8)

    def test_concurrent_schema_change_cannot_corrupt_an_acknowledged_registration(self):
        original = self.store._verify_schema

        def race(connection):
            original(connection)
            if not connection.execute("PRAGMA database_list").fetchone()[2]:
                return  # Inject only after validation of the live disk transaction.
            with fixture_database(self.store.path, timeout=0) as other:
                other.execute(
                    """CREATE TRIGGER racing_registration AFTER INSERT ON registrations
                    BEGIN UPDATE registrations SET model_id='foreign-model' WHERE owner_id=NEW.owner_id; END"""
                )

        with mock.patch.object(self.store, "_verify_schema", side_effect=race):
            with self.assertRaises(api.ControlStoreError):
                self.store.register(**FIELDS)
        self.assertIsNone(self.store.lookup(OWNER, JOB, OP))

    def test_lookup_schema_validation_and_record_read_share_a_transaction(self):
        expected = self.store.register(**FIELDS)
        original = self.store._verify_schema

        def race(connection):
            original(connection)
            if not connection.execute("PRAGMA database_list").fetchone()[2]:
                return  # The non-recovering preflight is a separate snapshot.
            with fixture_database(self.store.path, timeout=0) as other:
                other.execute("UPDATE registrations SET model_id='foreign-model'")

        with mock.patch.object(self.store, "_verify_schema", side_effect=race):
            with self.assertRaises(api.ControlStoreError):
                self.store.lookup(OWNER, JOB, OP)
        self.assertEqual(self.store.lookup(OWNER, JOB, OP), expected)

    def test_store_has_no_engine_imports_or_raw_input_storage(self):
        record = self.store.register(**FIELDS)
        encoded = json.dumps(dataclasses.asdict(record))
        self.assertNotIn("prompt", encoded)
        self.assertNotIn("reasoning", encoded)
        self.assertNotIn("mlx.core", sys.modules)
        self.assertNotIn("omlx.server", sys.modules)


if __name__ == "__main__":
    unittest.main(verbosity=2)
