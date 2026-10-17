# SPDX-License-Identifier: Apache-2.0
"""Counted host-read submissions.

``ThreadPoolExecutor.submit`` enqueues the work item before it starts a
worker thread, so a failed ``Thread.start()`` can leave a queued read with no
future for any caller to wait on. Owned work cannot prove such a read
stopped: every failed submission is counted here, and owned admission and
termination refuse once the count is non-zero or has moved.
"""

import threading

_lock = threading.Lock()
_failures = 0


def submit_tracked(pool, fn, /, *args, **kwargs):
    """``pool.submit(fn, *args, **kwargs)``, counting any failure.

    A failure may or may not have enqueued the work; both are counted.
    """
    global _failures
    try:
        return pool.submit(fn, *args, **kwargs)
    except BaseException:
        with _lock:
            _failures += 1
        raise


def submission_failures() -> int:
    """Failed submissions in this process; never decreases."""
    with _lock:
        return _failures
