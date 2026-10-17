# SPDX-License-Identifier: Apache-2.0
"""Retained executor admission facts. Nothing here certifies backend stop."""

import asyncio
from concurrent.futures import Future
from dataclasses import dataclass
import math
from threading import Lock
from time import monotonic
from typing import Any

from .host_submit import submission_failures


@dataclass(frozen=True, eq=False)
class _AdmissionBinding:
    core: Any
    generation_id: str
    engine_id: str
    request_id: str
    scheduler: Any
    executor: Any
    stream: Any
    deadline: float


@dataclass(frozen=True)
class AdmissionJoined:
    """Process-local stage fact; a failed attempt may have partially inserted."""

    handle: Any
    future: Future
    generation_id: str
    request_id: str
    insertion_attempted: bool
    insertion_returned: bool
    admission_failed: bool


@dataclass(frozen=True)
class StreamSynchronized:
    """Synchronization stage only; request generation may subsequently resume."""

    handle: Any
    future: Future
    generation_id: str
    request_id: str


@dataclass(frozen=True)
class WaitingCleaned:
    """Pristine queued-request cleanup only; not a provider stop receipt."""

    handle: Any
    request: Any
    generation_id: str
    request_id: str


@dataclass(frozen=True)
class RunningCleaned:
    """Decoding or chunked-prefill cleanup only; not a provider stop receipt."""

    handle: Any
    request: Any
    generation_id: str
    request_id: str
    state: str


_VLM_FIELDS = ("vlm_image_hash", "vlm_cache_key_ranges", "vlm_inputs_embeds", "vlm_extra_kwargs")


def _present(value) -> bool:
    # Never call bool() on an array: empty containers are absent, any other
    # non-None value (an mx.array of embeddings included) is present.
    if value is None:
        return False
    if isinstance(value, (dict, list, tuple, str, bytes)):
        return len(value) > 0
    return True


def _admitted_text_only(request) -> bool:
    """Decided from the Request as admitted; later decode clears VLM fields."""
    return not any(_present(getattr(request, name, None)) for name in ("images", "videos", *_VLM_FIELDS))


class OwnedAdmission:
    """Trusted Python handle; never accepted from public request metadata."""

    def __init__(
        self,
        core,
        request_id: str,
        deadline: float,
        *,
        retry: bool = True,
        submit_failures: int | None = None,
    ):
        if (
            not isinstance(request_id, str)
            or not request_id
            or not isinstance(deadline, (int, float))
            or isinstance(deadline, bool)
            or not math.isfinite(deadline)
            or deadline <= monotonic()
            or not isinstance(retry, bool)
            or not (submit_failures is None or type(submit_failures) is int)
        ):
            raise ValueError("owned_admission_invalid_identity_or_deadline")
        # 28: fixed before admission; False makes every scheduler retry site
        # end the request with an error instead of re-running it.
        self._retry = retry
        # 28: host_submit's count when preparation began. The dispatch on the
        # MLX executor inserts nothing once it has moved.
        self._submit_failures = submit_failures
        self._binding = _AdmissionBinding(
            core,
            core._prism_control_generation_id,
            core._engine_id,
            request_id,
            core.scheduler,
            core._mlx_executor,
            core._mlx_stream,
            deadline,
        )
        self._lock = Lock()
        self._future = None
        self._submitted_future = None
        self._request = None
        self._cancelled = False
        self._attempted = False
        self._returned = False
        self._stream_pinned_deadline = None
        self._stream_effective_deadline = None
        self._stream_started = False
        self._stream_future = None
        self._stream_submitted_future = None
        self._waiting_pristine = True
        self._cleanup_started = False
        self._cleanup_future = None
        self._cleanup_submitted_future = None
        self._cleanup_fact = None
        self._cleanup_outcome = None
        # 25b.4: fixed at admit(); the executor job records its classification.
        self._text_only = False
        self._classified_state = None
        # Set once scheduling commits the request to compute. A started request
        # that is requeued is no longer pristine and must never stay fenced.
        self._started = False
        # Names of retirement steps that failed; empty unless a cleanup failed.
        self._cleanup_failures = ()
        # 27.4a: the classification-only absence probe's latest run. Kept
        # apart from the cleanup fields, so a probe never blocks a barrier.
        self._probe_future = None
        self._probe_result = None
        # The dispatch future the latest probe ran against; a result binds to it.
        self._probe_dispatch = None
        self.validate(core, request_id)

    @property
    def binding(self):
        return self._binding

    @property
    def future(self):
        return self._future

    @property
    def request(self):
        return self._request

    @property
    def retry(self):
        return self._retry

    def validate(self, core, request_id, *, unused=False, deadline=False):
        b = self.binding
        if (
            core is not b.core
            or request_id != b.request_id
            or core._prism_control_generation_id != b.generation_id
            or core._engine_id != b.engine_id
            or core.scheduler is not b.scheduler
            or core._mlx_executor is not b.executor
            or core._mlx_stream is not b.stream
            or b.stream is None
            or b.executor is None
            or b.scheduler is None
            or getattr(core, "_closed", False)
            or getattr(core, "_closing", False)
            or (self._request is not None and self._request.request_id != b.request_id)
        ):
            raise ValueError("owned_admission_binding_changed")
        if unused and self._request is not None:
            raise ValueError("owned_admission_already_used")
        if deadline and monotonic() >= b.deadline:
            raise TimeoutError("owned_admission_expired")

    def cancel(self):
        # This same lock orders the worker's check+attempt record.
        with self._lock:
            self._cancelled = True

    def pin_stream_deadline(self, core, deadline):
        if (
            not isinstance(deadline, (int, float))
            or isinstance(deadline, bool)
            or not math.isfinite(deadline)
        ):
            raise ValueError("owned_stream_invalid_deadline")
        with self._lock:
            self.validate(core, self.binding.request_id)
            if self._stream_pinned_deadline is None:
                self._stream_pinned_deadline = deadline
                self._stream_effective_deadline = deadline
            else:
                self._stream_effective_deadline = min(
                    self._stream_effective_deadline, deadline
                )
            return self._stream_effective_deadline

    def _dispatch(self):
        with self._lock:
            b = self.binding
            self.validate(b.core, b.request_id)
            if self._cancelled or monotonic() >= b.deadline:
                return
            if (
                self._submit_failures is not None
                and submission_failures() != self._submit_failures
            ):
                # A host read may run with no future to join: admit nothing.
                return
            if b.scheduler.requests.get(b.request_id) is not None:
                raise ValueError("owned_admission_request_id_in_use")
            self._attempted = True
            request = self._request
        b.scheduler.add_request(request)
        with self._lock:
            self._returned = True

    @staticmethod
    async def _wait(future, deadline):
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError("owned_admission_wait_expired")
        wrapped = asyncio.wrap_future(future)
        # A cancelled caller may never consume a later worker exception.
        wrapped.add_done_callback(
            lambda f: f.exception() if not f.cancelled() else None
        )
        async with asyncio.timeout(remaining):
            await asyncio.shield(wrapped)

    async def admit(self, core, request):
        with self._lock:
            self.validate(core, request.request_id, unused=True, deadline=True)
            if self._cancelled:
                raise asyncio.CancelledError()
            self._request = request
            self._text_only = _admitted_text_only(request)
            request._prism_owned_admission = self
            # Holding the lock makes the worker wait until the real future is
            # retained. The request also fences retries if submit raises.
            self._future = self.binding.executor.submit(self._dispatch)
            self._submitted_future = self._future
            future = self._future
        try:
            await self._wait(future, self.binding.deadline)
            self.validate(core, request.request_id, deadline=True)
            with self._lock:
                if self._cancelled:
                    raise asyncio.CancelledError()
                if not self._returned:
                    raise TimeoutError("owned_admission_not_inserted")
        except BaseException:
            self.cancel()
            raise

    async def join(self, core, deadline: float):
        """Join admission only; timeout/cancellation yields no stage fact."""
        if (
            not isinstance(deadline, (int, float))
            or isinstance(deadline, bool)
            or not math.isfinite(deadline)
        ):
            return None
        with self._lock:
            try:
                self.validate(core, self.binding.request_id)
            except ValueError:
                return None
            future = self._future
        if future is None:
            return None
        try:
            await self._wait(future, deadline)
        except asyncio.CancelledError:
            return None
        except TimeoutError:
            if not future.done():
                return None
        except Exception:
            # The actual worker may have raised after partial insertion.
            pass
        with self._lock:
            try:
                self.validate(core, self.binding.request_id)
            except ValueError:
                return None
            if (
                self._future is not future
                or self._submitted_future is not future
                or not future.done()
                or future.cancelled()
                or monotonic() >= deadline
            ):
                return None
            return AdmissionJoined(
                self,
                future,
                self.binding.generation_id,
                self.binding.request_id,
                self._attempted,
                self._returned,
                future.exception() is not None,
            )
