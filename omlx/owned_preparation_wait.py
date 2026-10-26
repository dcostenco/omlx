# SPDX-License-Identifier: Apache-2.0
"""Keep preparation's executor work owned until its coroutine actually settles."""

import asyncio
from time import monotonic


async def wait_owned_preparation(execution, handle, awaitable):
    """Fence late admission without cancelling an executor's awaiting task.

    The caller retains its engine lease throughout this wait. Its admission
    deadline fences submission, but cannot interrupt already running thread
    work. A client may time out before this finishes; that is not stop proof.
    """
    task = asyncio.ensure_future(awaitable)
    fenced = False
    while not task.done():
        remaining = execution.deadline - monotonic()
        if not fenced and remaining <= 0:
            handle.cancel()
            fenced = True
        try:
            done, _ = await asyncio.wait((task,), timeout=None if fenced else remaining)
            if not done and not fenced:
                handle.cancel()
                fenced = True
        except asyncio.CancelledError:
            execution.cancellations.append(None)
            handle.cancel()
            fenced = True
    return task.result()
