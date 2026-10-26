# SPDX-License-Identifier: Apache-2.0
"""Server-owned execution of registered control operations.

The coordinator, not the HTTP request, owns an execution: a client disconnect
never cancels it. The engine stage is injected; until owned preparation and
admission exist, the default refuses before any dispatch, so every claimed
execution ends ``not_started`` with ``not_admitted``. No engine import here.
"""

from functools import partial
from pathlib import Path
import asyncio
import fcntl
import os
import stat
import time

from .prism_control_store import ControlConflict, ControlStoreError

DRAIN_WINDOW_SECONDS = 10
FINISH_BACKOFF_SECONDS = 0.05
FINISH_BACKOFF_CAP_SECONDS = 0.5
EXECUTION_LOCK = "execution.lock"
NOT_ADMITTED = ("not_started", ("not_admitted",))
FINAL_EVENTS = ("refused", "terminal", "unresolved")


class ExecutionExists(Exception):
    """A live execution already owns this identity in this process."""


class CoordinatorClosed(Exception):
    """The coordinator is draining for shutdown and takes no new work."""


class EngineStageUnavailable(Exception):
    """Raised by an engine stage only when nothing reached the scheduler."""


async def refuse_engine_stage(execution):
    raise EngineStageUnavailable()


def _lock_stamp(info):
    return (info.st_dev, info.st_ino, info.st_uid, info.st_mode, info.st_nlink)


def acquire_execution_lock(directory):
    """Process-lifetime exclusive lock; None when unsafe or held elsewhere.

    The descriptor is never closed by the coordinator: process exit releases
    it, so a successor can never execute while this process still might.
    """
    directory = Path(directory)
    path = directory / EXECUTION_LOCK
    descriptor = None
    try:
        root = directory.lstat()
        if (
            not stat.S_ISDIR(root.st_mode)
            or root.st_uid != os.getuid()
            or root.st_mode & 0o077
        ):
            return None
        descriptor = os.open(
            path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600
        )
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_mode & 0o077
            or info.st_nlink != 1
            or info.st_size
        ):
            raise OSError()
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if _lock_stamp(info) != _lock_stamp(path.lstat()):
            raise OSError()
        return descriptor
    except OSError:
        if descriptor is not None:
            os.close(descriptor)
        return None


async def _settle(future, cancellations):
    """Wait for an executor future even through task cancellation.

    Blocking store calls cannot be interrupted, and a result lost to a
    cancellation could orphan a claim or a terminal. Each cancellation is
    counted so the caller re-raises it once the stop proof is recorded.
    """
    while not future.done():
        try:
            await asyncio.wait((future,))
        except asyncio.CancelledError:
            cancellations.append(None)
    return future.result()


class Execution:
    """One live execution; touched only on the event-loop thread."""

    def __init__(self, coordinator, identity, request_sha256, model_id, settings_sha256, deadline,
                 request=None, image_policy=None):
        self._coordinator = coordinator
        self.identity = identity
        # The parsed registered request; owned admission reads it (27.4b).
        self.request = request
        self.image_policy = image_policy
        # Set on the loop thread once admission returned; the fence reads it.
        self.admitted = False
        self.request_sha256 = request_sha256
        self.model_id = model_id
        self.settings_sha256 = settings_sha256
        self.deadline = deadline
        self.claim = None
        self.cancel_requested = False
        self.stop_requested = False
        self.task = None
        self.final = False
        self._final_event = None
        # Owned by the engine stage; released exactly once there.
        self.lease = None
        # Set when preparation refuses; bounds the reclaim wait and finish.
        self.drain_deadline = None
        # The owned cleanup started by a cancel after admission (27.4b).
        self.cleanup_task = None
        # The reader's final output; a fence after it starts no cleanup (27.5).
        self.final_output = None
        # Cancellations absorbed while awaiting work that must not be lost;
        # raised again once the stop proof is recorded.
        self.cancellations = []
        self._fence = None
        self._events = asyncio.Queue()

    def emit(self, event, payload=None):
        if event in FINAL_EVENTS:
            self.final = True
            self._final_event = event
        self._events.put_nowait((event, payload))

    async def next_event(self):
        return await self._events.get()

    def publish_fence(self, fence):
        """Publish before the final verify; the cancel route then reaches it."""
        self._fence = fence

    async def verify(self):
        """The fence check before each resource stage; raises on refusal."""
        if self.cancel_requested or self.stop_requested:
            raise ControlConflict("control_claim_cancel_requested")
        # Computed on the loop thread, as the registration route does. Any
        # failure means settings unavailable, which the caller treats as a
        # refusal before dispatch.
        settings_sha256 = self._coordinator._settings_digest(self.model_id)
        await asyncio.get_running_loop().run_in_executor(
            None,
            partial(
                self._coordinator._store.verify_claim,
                self.claim,
                settings_sha256=settings_sha256,
                deadline=self.deadline,
            ),
        )
        # A fence that landed while the store was read is honoured here too.
        if self.cancel_requested or self.stop_requested:
            raise ControlConflict("control_claim_cancel_requested")


class ControlCoordinator:
    def __init__(self, store, settings_digest, engine_stage=None, *, lock):
        self._store = store
        self._settings_digest = settings_digest
        self._engine_stage = engine_stage or refuse_engine_stage
        self._lock = lock
        self._executions = {}
        self._closed = False

    @property
    def lock_descriptor(self):
        """Read-only, for test cleanup; nothing here ever closes it."""
        return self._lock

    def start(self, identity, *, request_sha256, model_id, settings_sha256, deadline,
              request=None, image_policy=None):
        """Register and start one execution; no await, so no second can race."""
        if self._closed:
            raise CoordinatorClosed()
        live = self._executions.get(identity)
        # A finished task's done callback runs a loop iteration later.
        if live is not None and not live.task.done():
            raise ExecutionExists()
        execution = Execution(
            self, identity, request_sha256, model_id, settings_sha256, deadline, request, image_policy
        )
        execution.task = asyncio.get_running_loop().create_task(self._run(execution))
        self._executions[identity] = execution
        execution.task.add_done_callback(partial(self._forget, identity, execution))
        return execution

    def _forget(self, identity, execution, _task):
        if self._executions.get(identity) is execution:
            del self._executions[identity]
        if not execution.final:
            # The task ended without its last event (an unexpected error, or a
            # cancellation at loop close); its reader must never wait forever.
            if execution.claim is None:
                execution.emit("refused", "unavailable")
            else:
                execution.emit("unresolved")
        if execution._final_event in ("refused", "terminal") and execution.lease is None:
            # A slow SSE reader may still hold the settled Execution and task.
            # Keep its queued events, but drop completed stage payloads. An
            # unresolved execution or held lease must keep its ownership.
            execution.request = None
            execution.image_policy = None
            execution.preparation_handle = None
            execution._fence = None
            execution.cleanup_task = None
            execution.final_output = None

    def fence(self, identity):
        """Called by the cancel route after the tombstone commits, before it answers."""
        execution = self._executions.get(identity)
        if execution is None or execution.task.done():
            return
        execution.cancel_requested = True
        if execution._fence is not None:
            execution._fence()

    async def drain(self, window=DRAIN_WINDOW_SECONDS):
        """Shutdown: refuse new work, then wait at most the window; never cancel."""
        self._closed = True
        pending = []
        for execution in list(self._executions.values()):
            execution.stop_requested = True
            if not execution.task.done():
                pending.append(execution.task)
        if pending:
            await asyncio.wait(pending, timeout=window)

    async def _run(self, execution):
        loop = asyncio.get_running_loop()
        cancellations = execution.cancellations
        owner_id, job_id, operation_id = execution.identity
        future = loop.run_in_executor(
            None,
            partial(
                self._store.claim,
                owner_id=owner_id,
                job_id=job_id,
                operation_id=operation_id,
                request_sha256=execution.request_sha256,
                model_id=execution.model_id,
                settings_sha256=execution.settings_sha256,
                deadline=execution.deadline,
            ),
        )
        try:
            execution.claim = await _settle(future, cancellations)
        except TimeoutError:
            execution.emit("refused", "timeout")
        except ControlConflict:
            execution.emit("refused", "conflict")
        except ControlStoreError:
            # Possibly claimed: no resource work; it reads claimed, then unknown.
            execution.emit("refused", "unavailable")
        except ValueError:
            execution.emit("refused", "invalid")
        except Exception:
            execution.emit("refused", "unavailable")
        if execution.claim is None:
            if cancellations:
                raise asyncio.CancelledError()
            return
        execution.emit("claimed")
        outcome, evidence = NOT_ADMITTED
        if not cancellations:
            try:
                await execution.verify()
            except asyncio.CancelledError:
                cancellations.append(None)
            except Exception:
                pass
            else:
                try:
                    outcome, evidence = await self._engine_stage(execution)
                except EngineStageUnavailable:
                    pass
                except Exception:
                    # The stage cannot say nothing reached the scheduler, so no
                    # terminal: the operation stays claimed and fails closed.
                    execution.emit("unresolved")
                    return
        record = await self._finish(execution, outcome, evidence, cancellations)
        execution.emit("unresolved" if record is None else "terminal", record)
        if cancellations:
            raise asyncio.CancelledError()

    async def _finish(self, execution, outcome, evidence, cancellations):
        """Bounded retry while the store is unavailable; a conflict is final."""
        loop = asyncio.get_running_loop()
        # One window: a refused preparation already started it.
        until = execution.drain_deadline
        if until is None:
            until = time.monotonic() + DRAIN_WINDOW_SECONDS
        delay = FINISH_BACKOFF_SECONDS
        while True:
            future = loop.run_in_executor(
                None,
                partial(
                    self._store.finish,
                    execution.claim,
                    outcome=outcome,
                    evidence=evidence,
                ),
            )
            try:
                return await _settle(future, cancellations)
            except ControlConflict:
                # A subclass of ControlStoreError, so it is caught first.
                return None
            except ControlStoreError:
                pass
            except Exception:
                return None
            if time.monotonic() + delay >= until:
                return None
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                cancellations.append(None)
            delay = min(delay * 2, FINISH_BACKOFF_CAP_SECONDS)
