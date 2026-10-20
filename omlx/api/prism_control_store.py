# SPDX-License-Identifier: Apache-2.0
"""Durable control registration, execution claims and terminal records.

The future authenticated adapter must supply the owner namespace and digests
computed from effective requests/settings. Caller-provided digests alone do not
qualify execution. Old records retain their original generation after restart.
A claim records that one worker of this store generation may start resource
work for an operation; it is never replayed by a later generation.
"""

from contextlib import contextmanager
from dataclasses import dataclass, replace
import fcntl
import math
import os
from pathlib import Path
import re
import sqlite3
import stat
import time
from typing import Optional
from uuid import UUID, uuid4

MAX_OPERATIONS = 1024
MAX_DATABASE_BYTES = 4 * 1024 * 1024
# Slice 28: UUIDv7 operation identities are retired this long after they were
# minted; any other identity is kept for the store's life. A claim without a
# terminal is never retired, so the horizon only has to cover the moments an
# identity is unprotected: mint to execute, and the read after its terminal.
# Five minutes of one device's fastest operations (about 300 ms each) fit in
# MAX_OPERATIONS; an hour did not, and capacity refused ordinary work.
RETENTION_SECONDS = 300
SCHEMA_VERSION = 5
CONTROL_RECORD_VERSION = 3
LOCK_WAIT_SECONDS = 2
LOCK_POLL_SECONDS = 0.005
SQLITE_SIDECARS = ("-journal", "-wal", "-shm")
HASH = re.compile(r"[a-f0-9]{64}\Z")
MODEL = re.compile(r"[\x21-\x7e]{1,256}\Z")
REGISTRATION_COLUMNS = (
    "owner_id",
    "job_id",
    "operation_id",
    "generation_id",
    "request_sha256",
    "model_id",
    "settings_sha256",
)
METADATA_SCHEMA = "CREATE TABLE metadata (key TEXT PRIMARY KEY, value INTEGER NOT NULL)"
REGISTRATION_SCHEMA = """CREATE TABLE registrations (
    owner_id TEXT NOT NULL, job_id TEXT NOT NULL,
    operation_id TEXT NOT NULL, generation_id TEXT NOT NULL,
    request_sha256 TEXT NOT NULL, model_id TEXT NOT NULL,
    settings_sha256 TEXT NOT NULL,
    PRIMARY KEY (owner_id, job_id, operation_id))"""
CANCELLATION_SCHEMA = """CREATE TABLE cancellation_requests (
    owner_id TEXT NOT NULL, job_id TEXT NOT NULL,
    operation_id TEXT NOT NULL, generation_id TEXT NOT NULL,
    PRIMARY KEY (owner_id, job_id, operation_id))"""
CLAIM_SCHEMA = """CREATE TABLE claims (
    owner_id TEXT NOT NULL, job_id TEXT NOT NULL,
    operation_id TEXT NOT NULL, store_generation_id TEXT NOT NULL,
    worker_id TEXT NOT NULL UNIQUE, request_id TEXT NOT NULL UNIQUE,
    PRIMARY KEY (owner_id, job_id, operation_id))"""
TERMINAL_SCHEMA = """CREATE TABLE terminals (
    owner_id TEXT NOT NULL, job_id TEXT NOT NULL,
    operation_id TEXT NOT NULL, store_generation_id TEXT NOT NULL,
    outcome TEXT NOT NULL, backend_stopped INTEGER NOT NULL,
    evidence TEXT NOT NULL,
    PRIMARY KEY (owner_id, job_id, operation_id))"""
IDENTITY_WHERE = "owner_id=? AND job_id=? AND operation_id=?"
# Stop facts a coordinator may hold, in the canonical stored order.
EVIDENCE_ORDER = (
    "not_admitted",
    "admission_refused",
    "finished_output",
    "error_output",
    "waiting_cleaned",
    "running_cleaned",
    "absent_proved",
    "deferred_cleanup",
    "stream_synchronized",
)
# Every accepted evidence shape per outcome. A terminal is written only with
# stop proof, so there is no "unknown" outcome and backend_stopped is always 1.
TERMINAL_EVIDENCE = {
    "not_started": (
        frozenset({"not_admitted"}),
        frozenset({"admission_refused", "stream_synchronized"}),
    ),
    # absent_proved: the owned-cleanup entry found the request gone from the
    # scheduler on the executor; an in-flight cache store keeps it present.
    "completed": (
        frozenset({"finished_output", "absent_proved", "stream_synchronized"}),
        frozenset({"finished_output", "deferred_cleanup", "stream_synchronized"}),
    ),
    "failed": (frozenset({"error_output", "absent_proved", "stream_synchronized"}),),
    "cancelled": (
        frozenset({"waiting_cleaned", "stream_synchronized"}),
        frozenset({"running_cleaned", "stream_synchronized"}),
    ),
}


class ControlStoreError(RuntimeError):
    """Control state is unavailable; never synthesize completion."""


class ControlConflict(ControlStoreError):
    """A registered identity cannot be rebound."""


class ControlCapacity(ControlStoreError):
    """Exhaustion refuses new work without recycling existing identities."""


class ControlExpired(ControlCapacity):
    """A retired time-ordered identity; it can never be registered again."""


class _ClaimExpired(Exception):
    """Private: not an OSError, so the store helpers never rewrap it."""


def _now():
    return time.monotonic()


@dataclass(frozen=True)
class ControlRegistration:
    owner_id: str
    job_id: str
    operation_id: str
    generation_id: str
    request_sha256: str
    model_id: str
    settings_sha256: str
    version: int = CONTROL_RECORD_VERSION
    state: str = "registered"
    backend_stopped: bool = False
    claim_state: str = "none"


@dataclass(frozen=True)
class ControlCancellation:
    owner_id: str
    job_id: str
    operation_id: str
    generation_id: str
    version: int = CONTROL_RECORD_VERSION
    state: str = "cancel_requested"
    backend_stopped: bool = False
    claim_state: str = "none"
    claim_store_generation_id: Optional[str] = None


@dataclass(frozen=True)
class ControlClaimStatus:
    """Public status of a claimed operation, with the registration bindings."""

    owner_id: str
    job_id: str
    operation_id: str
    generation_id: str
    request_sha256: str
    model_id: str
    settings_sha256: str
    claim_store_generation_id: str
    version: int = CONTROL_RECORD_VERSION
    state: str = "claimed"
    claim_state: str = "claimed"
    backend_stopped: bool = False


@dataclass(frozen=True)
class ControlTerminalStatus:
    """Public status of a finished operation: durable stop proof, never replayed."""

    owner_id: str
    job_id: str
    operation_id: str
    generation_id: str
    request_sha256: str
    model_id: str
    settings_sha256: str
    claim_store_generation_id: str
    evidence: tuple
    cancel_requested: bool
    version: int = CONTROL_RECORD_VERSION
    state: str = "completed"
    claim_state: str = "finished"
    backend_stopped: bool = True


@dataclass(frozen=True)
class ControlClaim:
    """Trusted in-process handle for the coordinator; never serialized."""

    owner_id: str
    job_id: str
    operation_id: str
    store_generation_id: str
    worker_id: str
    request_id: str


def _wall_ms():
    return int(time.time() * 1000)


def _uuid7_ms(value):
    """The mint time of a UUIDv7 identity in milliseconds, or None otherwise."""
    parsed = UUID(value)
    return parsed.int >> 80 if parsed.version == 7 else None


def _uuid(value):
    try:
        if not isinstance(value, str) or str(UUID(value)) != value:
            raise ValueError()
    except (ValueError, AttributeError):
        raise ValueError("control_request_invalid") from None


def _hash(value):
    if not isinstance(value, str) or HASH.fullmatch(value) is None:
        raise ValueError("control_request_invalid")


def _deadline(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError("control_request_invalid")
    if _now() >= value:
        raise TimeoutError("control_claim_expired")


def _evidence(outcome, evidence):
    """The canonical stored form of one accepted evidence shape, or ValueError."""
    shapes = TERMINAL_EVIDENCE.get(outcome) if isinstance(outcome, str) else None
    if (
        shapes is None
        or type(evidence) is not tuple
        or any(type(token) is not str for token in evidence)
        or len(set(evidence)) != len(evidence)
        or frozenset(evidence) not in shapes
    ):
        raise ValueError("control_request_invalid")
    return ",".join(token for token in EVIDENCE_ORDER if token in evidence)


def _validate(
    owner_id, job_id, operation_id, request_sha256, model_id, settings_sha256
):
    _hash(owner_id)
    _uuid(job_id)
    _uuid(operation_id)
    _hash(request_sha256)
    _hash(settings_sha256)
    if not isinstance(model_id, str) or MODEL.fullmatch(model_id) is None:
        raise ValueError("control_request_invalid")


class ControlStore:
    """Private, bounded SQLite records with transactional duplicate admission.

    Storage paths and capacity are host configuration, never request fields.
    Cancellation is a registration fence; it never proves a stop. A terminal
    record is written only by the claiming process, with stop evidence.
    """

    def __init__(self, directory, *, capacity=MAX_OPERATIONS):
        if type(capacity) is not int or not 1 <= capacity <= MAX_OPERATIONS:
            raise ValueError("control_capacity_invalid")
        self._directory = Path(directory).absolute()
        self._path = self._directory / "registrations.sqlite3"
        self._capacity = capacity
        self._generation_id = str(uuid4())
        try:
            self._directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            self._check_directory()
            with self._exclusion():
                self._no_sidecars()
                created = False
                initial_stamp = None
                try:
                    descriptor = os.open(
                        self._path,
                        os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                        0o600,
                    )
                except FileExistsError:
                    pass
                else:
                    created = True
                    initial_stamp = self._stamp(os.fstat(descriptor))
                    os.close(descriptor)
                    descriptor = os.open(self._directory, os.O_RDONLY | os.O_DIRECTORY)
                    try:
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
                with self._disk_connection(
                    verify_schema=False,
                    write=True,
                    initializing=created,
                    initial_stamp=initial_stamp,
                ) as connection:
                    if created:
                        connection.execute(METADATA_SCHEMA)
                        connection.execute(
                            "INSERT INTO metadata VALUES ('capacity', ?)", (capacity,)
                        )
                        connection.execute(
                            "INSERT INTO metadata VALUES ('retired_through_ms', 0)"
                        )
                        connection.execute(REGISTRATION_SCHEMA)
                        connection.execute(CANCELLATION_SCHEMA)
                        connection.execute(CLAIM_SCHEMA)
                        connection.execute(TERMINAL_SCHEMA)
                        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
                    self._verify_schema(connection)
                    connection.commit()
        except (OSError, sqlite3.Error, ControlStoreError):
            raise ControlStoreError("control_store_unavailable") from None

    @property
    def path(self):
        return self._path

    @property
    def generation_id(self):
        return self._generation_id

    def _check_directory(self):
        info = self._directory.lstat()
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_mode & 0o077
        ):
            raise ControlStoreError("control_store_unavailable")
        return info.st_dev, info.st_ino, info.st_uid, info.st_mode

    def _check_file(self):
        info = self._path.lstat()
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_mode & 0o077
            or info.st_nlink != 1
            or info.st_size > MAX_DATABASE_BYTES
        ):
            raise ControlStoreError("control_store_unavailable")
        return info.st_dev, info.st_ino, info.st_uid, info.st_mode

    @contextmanager
    def _exclusion(self):
        descriptor = None
        try:
            root = self._check_directory()
            lock_path = self._directory / "control.lock"
            descriptor = os.open(
                lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600
            )
            info = os.fstat(descriptor)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()
                or info.st_mode & 0o077
                or info.st_nlink != 1
                or info.st_size
            ):
                raise ControlStoreError("control_store_unavailable")
            until = time.monotonic() + LOCK_WAIT_SECONDS
            while True:
                if time.monotonic() >= until:
                    raise ControlStoreError("control_store_unavailable")
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    if time.monotonic() >= until:
                        raise ControlStoreError("control_store_unavailable")
                    break
                except BlockingIOError:
                    if time.monotonic() >= until:
                        raise ControlStoreError("control_store_unavailable")
                    time.sleep(LOCK_POLL_SECONDS)
            if root != self._check_directory() or self._stamp(info) != self._stamp(
                lock_path.lstat()
            ):
                raise ControlStoreError("control_store_unavailable")
            yield
        except OSError:
            raise ControlStoreError("control_store_unavailable") from None
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def _no_sidecars(self):
        for suffix in SQLITE_SIDECARS:
            try:
                Path(str(self._path) + suffix).lstat()
            except FileNotFoundError:
                continue
            raise ControlStoreError("control_store_unavailable")

    @staticmethod
    def _stamp(info):
        return (
            info.st_dev,
            info.st_ino,
            info.st_uid,
            info.st_mode,
            info.st_nlink,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )

    def _preflight(self):
        """Inspect a bounded file image in RAM; SQLite never opens disk here.

        All ControlStore writers hold the separate exclusion. An external file
        editor is outside that protocol; identity/stamp changes fail closed.
        """
        self._no_sidecars()
        descriptor = os.open(self._path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            before = os.fstat(descriptor)
            self._check_file()
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_uid != os.getuid()
                or before.st_mode & 0o077
                or before.st_nlink != 1
                or before.st_size > MAX_DATABASE_BYTES
            ):
                raise ControlStoreError("control_store_unavailable")
            data = bytearray()
            while True:
                chunk = os.read(
                    descriptor, min(65536, MAX_DATABASE_BYTES + 1 - len(data))
                )
                if not chunk:
                    break
                data.extend(chunk)
                if len(data) > MAX_DATABASE_BYTES:
                    raise ControlStoreError("control_store_unavailable")
            stamp = self._stamp(before)
            if stamp != self._stamp(os.fstat(descriptor)) or stamp != self._stamp(
                self._path.lstat()
            ):
                raise ControlStoreError("control_store_unavailable")
        finally:
            os.close(descriptor)
        self._no_sidecars()
        # WAL-mode images are ambiguous even without visible sidecars.
        if (
            len(data) < 100
            or data[:16] != b"SQLite format 3\x00"
            or data[18:20] != b"\x01\x01"
        ):
            raise ControlStoreError("control_store_unavailable")
        memory = sqlite3.connect(":memory:", isolation_level=None)
        try:
            memory.deserialize(bytes(data))
            self._verify_schema(memory)
        finally:
            memory.close()
        if stamp != self._stamp(self._path.lstat()):
            raise ControlStoreError("control_store_unavailable")
        self._no_sidecars()
        return stamp

    @contextmanager
    def _connection(self, *, verify_schema=True, write=False, initializing=False):
        with self._exclusion():
            with self._disk_connection(
                verify_schema=verify_schema, write=write, initializing=initializing
            ) as connection:
                yield connection

    @contextmanager
    def _disk_connection(
        self, *, verify_schema=True, write=False, initializing=False, initial_stamp=None
    ):
        connection = None
        try:
            directory = self._check_directory()
            file_identity = self._check_file()
            self._no_sidecars()
            stamp = initial_stamp if initializing else self._preflight()
            if initializing and (
                stamp is None
                or stamp != self._stamp(self._path.lstat())
                or self._path.stat().st_size != 0
            ):
                raise ControlStoreError("control_store_unavailable")
            connection = sqlite3.connect(self._path, timeout=2, isolation_level=None)
            if stamp is not None:
                self._no_sidecars()
                if stamp != self._stamp(self._path.lstat()):
                    raise ControlStoreError("control_store_unavailable")
            if (
                directory != self._check_directory()
                or file_identity != self._check_file()
            ):
                raise ControlStoreError("control_store_unavailable")
            connection.execute("PRAGMA synchronous = FULL")
            if write:
                # macOS fsync does not flush the drive cache; F_FULLFSYNC does.
                connection.execute("PRAGMA fullfsync = ON")
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            if verify_schema:
                self._verify_schema(connection)
            yield connection
            if (
                directory != self._check_directory()
                or file_identity != self._check_file()
            ):
                raise ControlStoreError("control_store_unavailable")
        except (OSError, sqlite3.Error):
            raise ControlStoreError("control_store_unavailable") from None
        finally:
            if connection is not None:
                connection.close()

    def _verify_schema(self, connection):
        # Validate the complete schema, including implicit unique indices. Extra
        # triggers/views or weakened constraints are unknown control state.
        entries = connection.execute(
            "SELECT type,name,tbl_name,sql FROM sqlite_master"
        ).fetchall()

        def normalize(sql):
            return " ".join(sql.split()) if sql is not None else None

        actual = {
            (kind, name, table, normalize(sql)) for kind, name, table, sql in entries
        }
        expected = {
            ("table", "metadata", "metadata", normalize(METADATA_SCHEMA)),
            ("table", "registrations", "registrations", normalize(REGISTRATION_SCHEMA)),
            (
                "table",
                "cancellation_requests",
                "cancellation_requests",
                normalize(CANCELLATION_SCHEMA),
            ),
            ("table", "claims", "claims", normalize(CLAIM_SCHEMA)),
            ("index", "sqlite_autoindex_claims_1", "claims", None),
            ("index", "sqlite_autoindex_claims_2", "claims", None),
            ("index", "sqlite_autoindex_claims_3", "claims", None),
            ("table", "terminals", "terminals", normalize(TERMINAL_SCHEMA)),
            ("index", "sqlite_autoindex_terminals_1", "terminals", None),
            ("index", "sqlite_autoindex_metadata_1", "metadata", None),
            ("index", "sqlite_autoindex_registrations_1", "registrations", None),
            (
                "index",
                "sqlite_autoindex_cancellation_requests_1",
                "cancellation_requests",
                None,
            ),
        }
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if version != SCHEMA_VERSION or actual != expected:
            raise ControlStoreError("control_store_unavailable")
        metadata = dict(connection.execute("SELECT key,value FROM metadata").fetchall())
        retired = metadata.get("retired_through_ms")
        if (
            metadata.keys() != {"capacity", "retired_through_ms"}
            or metadata["capacity"] != self._capacity
            or type(retired) is not int
            or retired < 0
        ):
            raise ControlStoreError("control_store_unavailable")

    @staticmethod
    def _retire(connection):
        """Advance the one-way retention mark and drop what it passed (slice 28).

        Runs first in every write transaction that adds an identity, so the
        mark and the removals commit together or not at all. A claim
        without a terminal keeps all its rows.
        """
        (retired,) = connection.execute(
            "SELECT value FROM metadata WHERE key='retired_through_ms'"
        ).fetchone()
        retired = max(retired, _wall_ms() - RETENTION_SECONDS * 1000)
        connection.execute(
            "UPDATE metadata SET value=? WHERE key='retired_through_ms'", (retired,)
        )
        identities = connection.execute(
            """SELECT owner_id,job_id,operation_id FROM registrations
            UNION SELECT owner_id,job_id,operation_id FROM cancellation_requests
            UNION SELECT owner_id,job_id,operation_id FROM claims
            UNION SELECT owner_id,job_id,operation_id FROM terminals"""
        ).fetchall()
        for identity in identities:
            try:
                minted = _uuid7_ms(identity[2])
            except (ValueError, TypeError, AttributeError):
                raise ControlStoreError("control_store_unavailable") from None
            if minted is None or minted > retired:
                continue
            claimed = connection.execute(
                f"SELECT 1 FROM claims WHERE {IDENTITY_WHERE}", identity
            ).fetchone()
            finished = connection.execute(
                f"SELECT 1 FROM terminals WHERE {IDENTITY_WHERE}", identity
            ).fetchone()
            if claimed and not finished:
                continue
            for table in ("terminals", "claims", "cancellation_requests", "registrations"):
                connection.execute(f"DELETE FROM {table} WHERE {IDENTITY_WHERE}", identity)
        return retired

    @staticmethod
    def _refuse_retired(operation_id, retired):
        """A new UUIDv7 identity at or before the mark was, or may have been, dropped."""
        minted = _uuid7_ms(operation_id)
        if minted is not None and minted <= retired:
            raise ControlExpired("control_registration_expired")

    @staticmethod
    def _record(row):
        if row is None:
            return None
        try:
            record = ControlRegistration(*row)
            _validate(
                record.owner_id,
                record.job_id,
                record.operation_id,
                record.request_sha256,
                record.model_id,
                record.settings_sha256,
            )
            _uuid(record.generation_id)
            return record
        except (ValueError, TypeError):
            raise ControlStoreError("control_store_unavailable") from None

    @staticmethod
    def _claim_record(row):
        try:
            claim = ControlClaim(*row)
            _hash(claim.owner_id)
            for value in [
                claim.job_id,
                claim.operation_id,
                claim.store_generation_id,
                claim.worker_id,
                claim.request_id,
            ]:
                _uuid(value)
            return claim
        except (ValueError, TypeError):
            raise ControlStoreError("control_store_unavailable") from None

    def _claim_view(self, connection, identity):
        """The claim for an identity and its state as seen by this generation."""
        row = connection.execute(
            f"SELECT * FROM claims WHERE {IDENTITY_WHERE}", identity
        ).fetchone()
        if row is None:
            return None, "none", None
        claim = self._claim_record(row)
        # Another generation's claim cannot be vouched for by this process.
        state = (
            "claimed" if claim.store_generation_id == self._generation_id else "unknown"
        )
        return claim, state, claim.store_generation_id

    @staticmethod
    def _claim_status(record, claim_state, claim_generation):
        return ControlClaimStatus(
            record.owner_id,
            record.job_id,
            record.operation_id,
            record.generation_id,
            record.request_sha256,
            record.model_id,
            record.settings_sha256,
            claim_generation,
            state=claim_state,
            claim_state=claim_state,
        )

    @staticmethod
    def _terminal_row(row):
        """(store generation, outcome, evidence) of a stored terminal, validated."""
        try:
            owner_id, job_id, operation_id, generation, outcome, stopped, joined = row
            _hash(owner_id)
            for value in (job_id, operation_id, generation):
                _uuid(value)
            if type(stopped) is not int or stopped != 1 or type(joined) is not str:
                raise ValueError()
            evidence = tuple(joined.split(","))
            if _evidence(outcome, evidence) != joined:
                raise ValueError()
            return generation, outcome, evidence
        except (ValueError, TypeError):
            raise ControlStoreError("control_store_unavailable") from None

    def _terminal_view(self, connection, identity):
        """The terminal status of an identity, or None. A terminal is durable stop
        proof, so it reads the same after a restart: it never turns unknown."""
        row = connection.execute(
            f"SELECT * FROM terminals WHERE {IDENTITY_WHERE}", identity
        ).fetchone()
        if row is None:
            return None
        generation, outcome, evidence = self._terminal_row(row)
        record = self._record(
            connection.execute(
                f"SELECT * FROM registrations WHERE {IDENTITY_WHERE}", identity
            ).fetchone()
        )
        claim_row = connection.execute(
            f"SELECT * FROM claims WHERE {IDENTITY_WHERE}", identity
        ).fetchone()
        if (
            record is None
            or claim_row is None
            or self._claim_record(claim_row).store_generation_id != generation
        ):
            # A terminal always has its registration and claim; anything else
            # is unknown state.
            raise ControlStoreError("control_store_unavailable")
        cancelled = (
            connection.execute(
                f"SELECT 1 FROM cancellation_requests WHERE {IDENTITY_WHERE}",
                identity,
            ).fetchone()
            is not None
        )
        return ControlTerminalStatus(
            record.owner_id,
            record.job_id,
            record.operation_id,
            record.generation_id,
            record.request_sha256,
            record.model_id,
            record.settings_sha256,
            generation,
            evidence,
            cancelled,
            state=outcome,
        )

    def lookup(self, owner_id, job_id, operation_id):
        _hash(owner_id)
        _uuid(job_id)
        _uuid(operation_id)
        identity = (owner_id, job_id, operation_id)
        with self._connection() as connection:
            row = connection.execute(
                f"SELECT * FROM registrations WHERE {IDENTITY_WHERE}", identity
            ).fetchone()
            record = self._record(row)
            if record is None:
                return None
            if self._terminal_view(connection, identity) is not None:
                return replace(record, claim_state="finished")
            _, claim_state, _ = self._claim_view(connection, identity)
            return replace(record, claim_state=claim_state)

    @staticmethod
    def _cancellation(row, claim_state="none", claim_generation=None):
        if row is None:
            return None
        try:
            record = ControlCancellation(
                *row,
                claim_state=claim_state,
                claim_store_generation_id=claim_generation,
            )
            _hash(record.owner_id)
            for value in [record.job_id, record.operation_id, record.generation_id]:
                _uuid(value)
            return record
        except (ValueError, TypeError):
            raise ControlStoreError("control_store_unavailable") from None

    @staticmethod
    def _identity_count(connection):
        return connection.execute(
            """SELECT COUNT(*) FROM (
            SELECT owner_id,job_id,operation_id FROM registrations
            UNION SELECT owner_id,job_id,operation_id FROM cancellation_requests)"""
        ).fetchone()[0]

    def lookup_cancellation(self, owner_id, job_id, operation_id):
        _hash(owner_id)
        _uuid(job_id)
        _uuid(operation_id)
        identity = (owner_id, job_id, operation_id)
        with self._connection() as connection:
            row = connection.execute(
                f"SELECT * FROM cancellation_requests WHERE {IDENTITY_WHERE}",
                identity,
            ).fetchone()
            if row is None:
                return None
            terminal = self._terminal_view(connection, identity)
            if terminal is not None:
                return terminal
            _, claim_state, claim_generation = self._claim_view(connection, identity)
            return self._cancellation(row, claim_state, claim_generation)

    def lookup_status(self, owner_id, job_id, operation_id):
        _hash(owner_id)
        _uuid(job_id)
        _uuid(operation_id)
        identity = (owner_id, job_id, operation_id)
        with self._connection() as connection:
            terminal = self._terminal_view(connection, identity)
            if terminal is not None:
                return terminal
            claim, claim_state, claim_generation = self._claim_view(
                connection, identity
            )
            row = connection.execute(
                f"SELECT * FROM cancellation_requests WHERE {IDENTITY_WHERE}",
                identity,
            ).fetchone()
            if row is not None:
                return self._cancellation(row, claim_state, claim_generation)
            record = self._record(
                connection.execute(
                    f"SELECT * FROM registrations WHERE {IDENTITY_WHERE}", identity
                ).fetchone()
            )
            if claim is None:
                return record
            if record is None:
                # A claim always has a registration; anything else is unknown state.
                raise ControlStoreError("control_store_unavailable")
            return self._claim_status(record, claim_state, claim_generation)

    def request_cancel(self, owner_id, job_id, operation_id):
        _hash(owner_id)
        _uuid(job_id)
        _uuid(operation_id)
        identity = (owner_id, job_id, operation_id)
        with self._connection(write=True) as connection:
            retired = self._retire(connection)
            terminal = self._terminal_view(connection, identity)
            if terminal is not None:
                # Too late to stop anything: record the request, report the outcome.
                if not terminal.cancel_requested:
                    connection.execute(
                        "INSERT INTO cancellation_requests VALUES (?,?,?,?)",
                        (*identity, self._generation_id),
                    )
                    connection.commit()
                return replace(terminal, cancel_requested=True)
            _, claim_state, claim_generation = self._claim_view(connection, identity)
            row = connection.execute(
                f"SELECT * FROM cancellation_requests WHERE {IDENTITY_WHERE}",
                identity,
            ).fetchone()
            if row is not None:
                return self._cancellation(row, claim_state, claim_generation)
            known = connection.execute(
                f"SELECT 1 FROM registrations WHERE {IDENTITY_WHERE}", identity
            ).fetchone()
            if not known:
                self._refuse_retired(operation_id, retired)
                if self._identity_count(connection) >= self._capacity:
                    raise ControlCapacity("control_registration_capacity")
            record = ControlCancellation(
                *identity,
                self._generation_id,
                claim_state=claim_state,
                claim_store_generation_id=claim_generation,
            )
            connection.execute(
                "INSERT INTO cancellation_requests VALUES (?,?,?,?)",
                (*identity, record.generation_id),
            )
            connection.commit()
            return record

    def register(
        self,
        *,
        owner_id,
        job_id,
        operation_id,
        request_sha256,
        model_id,
        settings_sha256,
        admission_check=None,
    ):
        _validate(
            owner_id, job_id, operation_id, request_sha256, model_id, settings_sha256
        )
        identity = (owner_id, job_id, operation_id)
        with self._connection(write=True) as connection:
            if admission_check is not None:
                admission_check()
            retired = self._retire(connection)
            terminal = self._terminal_view(connection, identity)
            if terminal is not None:
                # A finished operation reports its outcome, cancelled or not.
                if (terminal.request_sha256, terminal.model_id, terminal.settings_sha256) != (
                    request_sha256,
                    model_id,
                    settings_sha256,
                ):
                    raise ControlConflict("control_registration_conflict")
                connection.commit()
                return terminal
            cancelled = connection.execute(
                f"SELECT 1 FROM cancellation_requests WHERE {IDENTITY_WHERE}",
                identity,
            ).fetchone()
            if cancelled:
                raise ControlConflict("control_registration_cancel_requested")
            row = connection.execute(
                f"SELECT * FROM registrations WHERE {IDENTITY_WHERE}", identity
            ).fetchone()
            if row is not None:
                record = self._record(row)
                if (record.request_sha256, record.model_id, record.settings_sha256) != (
                    request_sha256,
                    model_id,
                    settings_sha256,
                ):
                    raise ControlConflict("control_registration_conflict")
                claim, claim_state, claim_generation = self._claim_view(
                    connection, identity
                )
                connection.commit()
                if claim is None:
                    return record
                # A repeat of a claimed registration reports the claim, keeping
                # the original bindings, never a stale "registered".
                return self._claim_status(record, claim_state, claim_generation)
            if connection.execute(
                f"SELECT 1 FROM claims WHERE {IDENTITY_WHERE}", identity
            ).fetchone():
                # A claim always has a registration; anything else is unknown.
                raise ControlStoreError("control_store_unavailable")
            self._refuse_retired(operation_id, retired)
            if self._identity_count(connection) >= self._capacity:
                raise ControlCapacity("control_registration_capacity")
            record = ControlRegistration(
                owner_id,
                job_id,
                operation_id,
                self._generation_id,
                request_sha256,
                model_id,
                settings_sha256,
            )
            connection.execute(
                "INSERT INTO registrations VALUES (?,?,?,?,?,?,?)",
                tuple(getattr(record, key) for key in REGISTRATION_COLUMNS),
            )
            if admission_check is not None:
                admission_check()
            connection.commit()
            return record

    def claim(
        self,
        *,
        owner_id,
        job_id,
        operation_id,
        request_sha256,
        model_id,
        settings_sha256,
        deadline,
    ):
        """Persist the one execution claim for a registered operation.

        TimeoutError always means nothing was written. ControlStoreError means
        possibly written: the caller does no resource work. Once commit
        returns, the handle is returned even if a later cleanup step fails.
        """
        _validate(
            owner_id, job_id, operation_id, request_sha256, model_id, settings_sha256
        )
        _deadline(deadline)
        identity = (owner_id, job_id, operation_id)
        claim = None
        committed = False
        try:
            with self._connection(write=True) as connection:
                baseline = (self._check_directory(), self._check_file())
                if _now() >= deadline:
                    raise _ClaimExpired()
                if connection.execute(
                    f"SELECT 1 FROM cancellation_requests WHERE {IDENTITY_WHERE}",
                    identity,
                ).fetchone():
                    raise ControlConflict("control_claim_cancel_requested")
                record = self._record(
                    connection.execute(
                        f"SELECT * FROM registrations WHERE {IDENTITY_WHERE}",
                        identity,
                    ).fetchone()
                )
                if record is None:
                    raise ControlConflict("control_claim_unregistered")
                if connection.execute(
                    f"SELECT 1 FROM claims WHERE {IDENTITY_WHERE}", identity
                ).fetchone():
                    raise ControlConflict("control_claim_exists")
                if (record.request_sha256, record.model_id, record.settings_sha256) != (
                    request_sha256,
                    model_id,
                    settings_sha256,
                ):
                    raise ControlConflict("control_claim_digest_mismatch")
                if record.generation_id != self._generation_id:
                    raise ControlConflict("control_claim_stale_registration")
                claim = ControlClaim(
                    *identity, self._generation_id, str(uuid4()), str(uuid4())
                )
                connection.execute(
                    "INSERT INTO claims VALUES (?,?,?,?,?,?)",
                    (
                        *identity,
                        claim.store_generation_id,
                        claim.worker_id,
                        claim.request_id,
                    ),
                )
                if _now() >= deadline:
                    raise _ClaimExpired()
                if (self._check_directory(), self._check_file()) != baseline:
                    raise ControlStoreError("control_store_unavailable")
                connection.commit()
                committed = True
        except _ClaimExpired:
            # The connection closed without commit: nothing was written.
            raise TimeoutError("control_claim_expired") from None
        except ControlStoreError:
            if committed:
                # A durable claim whose handle is lost could never be resolved.
                return claim
            raise
        except Exception:
            if committed:
                return claim
            # A raw failure (e.g. closing the connection) means possibly written.
            raise ControlStoreError("control_store_unavailable") from None
        return claim

    def verify_claim(self, claim, *, settings_sha256, deadline):
        """Point-in-time fence check before each resource stage; raises or None."""
        if type(claim) is not ControlClaim:
            raise ValueError("control_request_invalid")
        _hash(claim.owner_id)
        _uuid(claim.job_id)
        _uuid(claim.operation_id)
        _hash(settings_sha256)
        _deadline(deadline)
        identity = (claim.owner_id, claim.job_id, claim.operation_id)
        try:
            with self._connection() as connection:
                if _now() >= deadline:
                    raise _ClaimExpired()
                if claim.store_generation_id != self._generation_id:
                    raise ControlConflict("control_claim_foreign_generation")
                row = connection.execute(
                    f"SELECT * FROM claims WHERE {IDENTITY_WHERE}", identity
                ).fetchone()
                if row is None:
                    raise ControlStoreError("control_store_unavailable")
                if self._claim_record(row) != claim:
                    raise ControlConflict("control_claim_not_owner")
                if connection.execute(
                    f"SELECT 1 FROM terminals WHERE {IDENTITY_WHERE}", identity
                ).fetchone():
                    raise ControlConflict("control_claim_finished")
                if connection.execute(
                    f"SELECT 1 FROM cancellation_requests WHERE {IDENTITY_WHERE}",
                    identity,
                ).fetchone():
                    raise ControlConflict("control_claim_cancel_requested")
                record = self._record(
                    connection.execute(
                        f"SELECT * FROM registrations WHERE {IDENTITY_WHERE}",
                        identity,
                    ).fetchone()
                )
                if record is None:
                    raise ControlStoreError("control_store_unavailable")
                if record.settings_sha256 != settings_sha256:
                    raise ControlConflict("control_claim_settings_changed")
        except _ClaimExpired:
            raise TimeoutError("control_claim_expired") from None
        except ControlStoreError:
            raise
        except Exception:
            raise ControlStoreError("control_store_unavailable") from None
        return None

    def finish(self, claim, *, outcome, evidence):
        """Record the one terminal of a claimed operation, with its stop proof.

        No deadline: a proof the coordinator already holds is always recorded.
        An identical existing terminal is returned, so a retry after an
        uncertain commit is safe. ControlStoreError means possibly written.
        """
        if type(claim) is not ControlClaim:
            raise ValueError("control_request_invalid")
        _hash(claim.owner_id)
        for value in (
            claim.job_id,
            claim.operation_id,
            claim.store_generation_id,
            claim.worker_id,
            claim.request_id,
        ):
            _uuid(value)
        joined = _evidence(outcome, evidence)
        identity = (claim.owner_id, claim.job_id, claim.operation_id)
        result = None
        committed = False
        try:
            with self._connection(write=True) as connection:
                baseline = (self._check_directory(), self._check_file())
                if claim.store_generation_id != self._generation_id:
                    raise ControlConflict("control_claim_foreign_generation")
                row = connection.execute(
                    f"SELECT * FROM claims WHERE {IDENTITY_WHERE}", identity
                ).fetchone()
                if row is None:
                    raise ControlStoreError("control_store_unavailable")
                if self._claim_record(row) != claim:
                    raise ControlConflict("control_claim_not_owner")
                existing = self._terminal_view(connection, identity)
                if existing is not None:
                    if (
                        existing.claim_store_generation_id,
                        existing.state,
                        existing.evidence,
                    ) == (claim.store_generation_id, outcome, tuple(joined.split(","))):
                        return existing
                    raise ControlConflict("control_terminal_exists")
                connection.execute(
                    "INSERT INTO terminals VALUES (?,?,?,?,?,?,?)",
                    (*identity, claim.store_generation_id, outcome, 1, joined),
                )
                result = self._terminal_view(connection, identity)
                if (self._check_directory(), self._check_file()) != baseline:
                    raise ControlStoreError("control_store_unavailable")
                connection.commit()
                committed = True
        except ControlStoreError:
            if committed:
                return result
            raise
        except Exception:
            if committed:
                return result
            raise ControlStoreError("control_store_unavailable") from None
        return result
