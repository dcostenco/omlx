# SPDX-License-Identifier: Apache-2.0
"""A drained owned response must not keep its model busy until the orphan reaper."""

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from omlx import owned_preparation as prep
from omlx.engine.batched import BatchedEngine
from omlx.engine_core import EngineCore
from omlx.output_collector import RequestOutputCollector
from omlx.owned_admission import StreamSynchronized


def completed_request():
    core = EngineCore.__new__(EngineCore)
    core._closed = core._closing = False
    core._output_collectors = {"owned": RequestOutputCollector()}
    core._stream_states = {"owned": object()}
    core._finished_events = {"owned": object()}
    core._finished_at = {"owned": time.monotonic()}
    handle = object()
    core._owned_stream_barrier = AsyncMock(
        return_value=StreamSynchronized(handle, object(), "generation", "owned")
    )
    engine = BatchedEngine.__new__(BatchedEngine)
    engine._engine = SimpleNamespace(engine=core)
    execution = SimpleNamespace(
        claim=SimpleNamespace(request_id="owned"),
        drain_deadline=time.monotonic() + 5,
        cleanup_task=None,
    )
    return core, engine, execution, handle


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", [False, True])
async def test_consumed_final_output_retires_busy_tracking_after_stop_proofs(monkeypatch, failed):
    core, engine, execution, handle = completed_request()
    assert engine.has_active_requests()

    async def absent(*args):
        assert engine.has_active_requests(), "Retirement must wait for absence proof"

    monkeypatch.setattr(prep, "_absent", absent)

    async def barrier(*args):
        assert engine.has_active_requests(), "Retirement must wait for stream synchronization"
        return StreamSynchronized(handle, object(), "generation", "owned")

    core._owned_stream_barrier.side_effect = barrier
    final = SimpleNamespace(error="failed" if failed else None, finish_reason="error" if failed else "stop")
    outcome, _ = await prep._terminate(execution, core, handle, final, prep.submission_failures())
    assert outcome == ("failed" if failed else "completed")
    assert not engine.has_active_requests(), "A completed classifier must be evictable immediately"
    for tracking in (core._stream_states, core._finished_events, core._finished_at):
        assert "owned" not in tracking


@pytest.mark.asyncio
async def test_completion_preserves_another_requests_collector(monkeypatch):
    core, engine, execution, handle = completed_request()
    other = RequestOutputCollector()
    core._output_collectors["other"] = other
    monkeypatch.setattr(prep, "_absent", AsyncMock())
    await prep._terminate(execution, core, handle, SimpleNamespace(error=None, finish_reason="stop"), prep.submission_failures())
    assert core._output_collectors == {"other": other}
    assert engine.has_active_requests(), "Another consumer still keeps this model busy"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["absence", "barrier", "no_final"])
async def test_unproved_completion_keeps_tracking(monkeypatch, failure):
    core, engine, execution, handle = completed_request()
    absent = AsyncMock()
    if failure == "absence":
        absent.side_effect = prep.ExecutionUnresolved("absence_unproved")
    if failure == "barrier":
        core._owned_stream_barrier.return_value = None
    monkeypatch.setattr(prep, "_absent", absent)
    final = None if failure == "no_final" else SimpleNamespace(error=None, finish_reason="stop")
    with pytest.raises(prep.ExecutionUnresolved):
        await prep._terminate(execution, core, handle, final, prep.submission_failures())
    assert engine.has_active_requests()
    assert "owned" in core._stream_states


@pytest.mark.asyncio
async def test_ambiguous_host_submission_keeps_tracking_after_other_proofs(monkeypatch):
    core, engine, execution, handle = completed_request()
    monkeypatch.setattr(prep, "_absent", AsyncMock())
    monkeypatch.setattr(prep, "submission_failures", lambda: 1)
    with pytest.raises(prep.ExecutionUnresolved, match="host_read_ambiguous"):
        await prep._terminate(execution, core, handle, SimpleNamespace(error=None, finish_reason="stop"), 0)
    assert engine.has_active_requests()
    assert "owned" in core._finished_at
