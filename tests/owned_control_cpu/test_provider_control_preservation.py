# SPDX-License-Identifier: Apache-2.0
"""Non-recovering refusal of unsupported control storage; private fixtures only."""

from contextlib import closing
import hashlib
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
import time
from unittest import mock
from contextlib import contextmanager

import test_provider_control_store as support


class PreservationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="prism-control-preserve-")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.path = self.directory / "registrations.sqlite3"

    def old_database(self, *, wal=False):
        connection = sqlite3.connect(self.path)
        connection.execute("CREATE TABLE obsolete (value TEXT)")
        connection.executemany(
            "INSERT INTO obsolete VALUES (?)", [("a" * 3000,) for _ in range(80)]
        )
        connection.execute("PRAGMA user_version=999")
        connection.commit()
        self.path.chmod(0o600)
        if wal:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("INSERT INTO obsolete VALUES ('wal-value')")
            connection.commit()
        return connection

    def hashes(self):
        paths = [self.path] + [
            Path(str(self.path) + suffix) for suffix in ["-journal", "-wal", "-shm"]
        ]
        return {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
            if path.exists()
        }

    def assert_preserved(self):
        before = self.hashes()
        with self.assertRaises(support.api.ControlStoreError):
            support.api.ControlStore(self.directory, capacity=8)
        self.assertEqual(self.hashes(), before)

    def test_clean_unsupported_schema_never_opens_disk_sqlite_for_recovery(self):
        with closing(self.old_database()):
            pass
        self.assert_preserved()

    def test_crashed_writer_hot_journal_is_preserved_on_refusal(self):
        with closing(self.old_database()):
            pass
        script = """import sqlite3,sys,os
connection=sqlite3.connect(sys.argv[1]);connection.execute('PRAGMA cache_size=1')
connection.execute('BEGIN IMMEDIATE');connection.execute("UPDATE obsolete SET value='changed'")
os._exit(0)
"""
        result = subprocess.run(
            [sys.executable, "-c", script, str(self.path)], timeout=10
        )
        self.assertEqual(result.returncode, 0)
        journal = Path(str(self.path) + "-journal")
        self.assertTrue(journal.exists())
        self.assertGreater(journal.stat().st_size, 512)
        self.assert_preserved()

    def test_old_wal_and_shared_memory_remain_byte_identical(self):
        with closing(self.old_database(wal=True)):
            self.assertTrue(Path(str(self.path) + "-wal").exists())
            self.assertTrue(Path(str(self.path) + "-shm").exists())
            self.assert_preserved()

    def test_sidecar_ambiguity_on_supported_database_is_not_automatically_repaired(
        self,
    ):
        store = support.api.ControlStore(self.directory, capacity=8)
        Path(str(self.path) + "-journal").write_bytes(b"synthetic-ambiguous-sidecar")
        before = self.hashes()
        with self.assertRaises(support.api.ControlStoreError):
            store.lookup(support.OWNER, support.JOB, support.OP)
        self.assertEqual(self.hashes(), before)

    def test_missing_main_with_orphan_sidecar_never_creates_or_repairs_database(self):
        for suffix in ["-journal", "-wal", "-shm"]:
            with tempfile.TemporaryDirectory(
                prefix="prism-orphan-sidecar-"
            ) as directory:
                path = Path(directory) / "registrations.sqlite3"
                sidecar = Path(str(path) + suffix)
                sidecar.write_bytes(b"synthetic-orphan-sidecar")
                before = sidecar.read_bytes()
                with self.assertRaises(support.api.ControlStoreError):
                    support.api.ControlStore(directory, capacity=8)
                self.assertFalse(path.exists())
                self.assertEqual(sidecar.read_bytes(), before)

    def test_replaced_new_empty_file_is_not_an_initialization_authority(self):
        with closing(self.old_database()):
            image = self.path.read_bytes()
        original = support.api.ControlStore._disk_connection
        disk_opens = []
        connect = sqlite3.connect

        @contextmanager
        def replace(store, **options):
            if options.get("initializing"):
                store.path.write_bytes(image)
            with original(store, **options) as connection:
                yield connection

        def track(path, *args, **options):
            if str(path) != ":memory:":
                disk_opens.append(str(path))
            return connect(path, *args, **options)

        directory = self.directory / "new-store"
        with mock.patch.object(
            support.api.ControlStore, "_disk_connection", replace
        ), mock.patch.object(sqlite3, "connect", track):
            with self.assertRaises(support.api.ControlStoreError):
                support.api.ControlStore(directory, capacity=8)
        self.assertEqual(disk_opens, [])
        self.assertEqual((directory / "registrations.sqlite3").read_bytes(), image)

    def test_real_file_lock_granted_after_budget_cannot_enter_operation(self):
        store = support.api.ControlStore(self.directory, capacity=8)
        original = support.api.fcntl.flock

        def slow_grant(descriptor, operation):
            original(descriptor, operation)
            time.sleep(0.01)

        before = self.path.read_bytes()
        entered = False
        with mock.patch.object(
            support.api, "LOCK_WAIT_SECONDS", 0.002
        ), mock.patch.object(support.api.fcntl, "flock", slow_grant):
            with self.assertRaises(support.api.ControlStoreError):
                with store._exclusion():
                    entered = True
        self.assertFalse(entered)
        self.assertEqual(self.path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
