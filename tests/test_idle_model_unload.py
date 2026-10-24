import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omlx import server
from omlx.engine_pool import EngineEntry, EnginePool


class _Engine:
    def __init__(self, active=False, scheduler=None, failed=None):
        self.active = active
        self.scheduler = scheduler
        self.runtime_failed_reason = failed

    def has_active_requests(self):
        return self.active


def _entry(engine, **kwargs):
    return EngineEntry(
        model_id="model/with-slash",
        model_path="/models/model",
        model_type="llm",
        engine_type="batched",
        estimated_size=1,
        engine=engine,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_idle_unload_requires_strict_quiescence(monkeypatch):
    pool = EnginePool()
    entry = _entry(_Engine())
    pool._entries[entry.model_id] = entry
    unloaded = []

    async def unload(model_id, *, require_stop_success=False):
        assert require_stop_success is True
        unloaded.append(model_id)

    monkeypatch.setattr(pool, "_unload_engine", unload)
    assert await pool.unload_if_idle_unpinned(entry.model_id)
    assert unloaded == [entry.model_id]

    for change in (
        {"in_use": 1},
        {"is_loading": True},
        {"is_pinned": True},
        {"pending_unload_reason": "pending"},
        {"abort_requested": True},
    ):
        entry = _entry(_Engine(), **change)
        pool._entries[entry.model_id] = entry
        assert not await pool.unload_if_idle_unpinned(entry.model_id)

    entry = _entry(_Engine(active=True))
    pool._entries[entry.model_id] = entry
    assert not await pool.unload_if_idle_unpinned(entry.model_id)
    entry.engine = _Engine(scheduler=SimpleNamespace(running=["request"]))
    assert not await pool.unload_if_idle_unpinned(entry.model_id)
    entry.engine = _Engine(scheduler=SimpleNamespace(has_requests=lambda: False, waiting=["request"]))
    assert not await pool.unload_if_idle_unpinned(entry.model_id)
    entry.engine = _Engine(failed="failed")
    entry.in_use = 1
    assert not await pool.unload_if_idle_unpinned(entry.model_id)


@pytest.mark.asyncio
async def test_idle_unload_refuses_unknown_or_raising_telemetry(monkeypatch):
    pool = EnginePool()
    entry = _entry(SimpleNamespace())
    pool._entries[entry.model_id] = entry
    assert not await pool.unload_if_idle_unpinned(entry.model_id)

    class Raising:
        def has_active_requests(self):
            raise RuntimeError("telemetry unavailable")

    entry.engine = Raising()
    assert not await pool.unload_if_idle_unpinned(entry.model_id)

    class UnknownRank:
        def has_active_requests(self):
            return False

        @classmethod
        def rank_side_active_requests(cls, engine):
            return None

    entry.engine = UnknownRank()
    assert not await pool.unload_if_idle_unpinned(entry.model_id)


def test_authenticated_idle_route_never_uses_abort_unload(monkeypatch):
    app = FastAPI()
    app.post("/v1/models/{model_id:path}/unload-idle")(server.unload_model_if_idle)
    pool = SimpleNamespace(unload_if_idle_unpinned=AsyncMock(return_value=True))
    app.dependency_overrides[server.verify_api_key] = lambda: True
    try:
        original_pool = server._server_state.engine_pool
        server._server_state.engine_pool = pool
        with TestClient(app) as client:
            response = client.post("/v1/models/model/with-slash/unload-idle")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "model_id": "model/with-slash"}
        pool.unload_if_idle_unpinned.assert_awaited_once_with("model/with-slash")
    finally:
        server._server_state.engine_pool = original_pool
        app.dependency_overrides.pop(server.verify_api_key, None)


def test_idle_route_uses_real_api_key_before_pool_mutation():
    app = FastAPI()
    app.post("/v1/models/{model_id:path}/unload-idle")(server.unload_model_if_idle)
    pool = SimpleNamespace(unload_if_idle_unpinned=AsyncMock(return_value=True))
    state = server._server_state
    old = (state.engine_pool, state.api_key, state.bind_host, state.global_settings)
    state.engine_pool, state.api_key, state.bind_host, state.global_settings = (
        pool, "test-api-key", "127.0.0.1", None
    )
    try:
        with TestClient(app) as client:
            assert client.post("/v1/models/model/unload-idle").status_code == 401
            assert client.post("/v1/models/model/unload-idle", headers={"Authorization": "Bearer wrong"}).status_code == 401
            response = client.post("/v1/models/model/unload-idle", headers={"Authorization": "Bearer test-api-key"})
        assert response.status_code == 200
        pool.unload_if_idle_unpinned.assert_awaited_once_with("model")
    finally:
        state.engine_pool, state.api_key, state.bind_host, state.global_settings = old


@pytest.mark.asyncio
async def test_cancelled_idle_unload_keeps_lock_until_teardown_finishes(monkeypatch):
    pool = EnginePool()
    entry = _entry(_Engine())
    pool._entries[entry.model_id] = entry
    entered, finish = asyncio.Event(), asyncio.Event()

    async def blocked_teardown(model_id, **_kwargs):
        entered.set()
        await finish.wait()

    monkeypatch.setattr(pool, "_stop_and_unload_engine", blocked_teardown)
    task = asyncio.create_task(pool.unload_if_idle_unpinned(entry.model_id))
    await entered.wait()
    task.cancel()
    task.cancel()
    await asyncio.sleep(0)
    waiter = asyncio.create_task(pool._lock.acquire())
    await asyncio.sleep(0)
    assert not waiter.done()
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await asyncio.wait_for(waiter, timeout=1)
    pool._lock.release()


@pytest.mark.asyncio
async def test_get_engine_cannot_lease_engine_during_idle_unload(monkeypatch):
    pool = EnginePool()
    entry = _entry(_Engine())
    pool._entries[entry.model_id] = entry
    entered, finish = asyncio.Event(), asyncio.Event()

    async def blocked_teardown(model_id, **_kwargs):
        entered.set()
        await finish.wait()

    async def no_splash(*_args, **_kwargs):
        return None

    async def no_failed(*_args, **_kwargs):
        return False

    monkeypatch.setattr(pool, "_stop_and_unload_engine", blocked_teardown)
    monkeypatch.setattr(pool, "_check_splash_launcher", no_splash)
    monkeypatch.setattr(pool, "_reloadable_failed_engine", no_failed)
    monkeypatch.setattr(pool, "_validate_llm_engine_ready", lambda *_args: None)
    baseline = await pool.get_engine(entry.model_id, _lease=True)
    assert baseline is entry.engine
    assert entry.in_use == 1
    await pool.release_engine(entry.model_id)
    assert entry.in_use == 0
    unload = asyncio.create_task(pool.unload_if_idle_unpinned(entry.model_id))
    await entered.wait()
    assert entry.model_id in pool._unloading_models
    acquire = asyncio.create_task(pool.get_engine(entry.model_id, _lease=True))
    await asyncio.sleep(0)
    assert not acquire.done()
    assert entry.in_use == 0
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        unload.cancel()
        await unload
    await asyncio.wait_for(acquire, timeout=1)
    assert entry.in_use == 1
    await pool.release_engine(entry.model_id)
    assert entry.in_use == 0


@pytest.mark.asyncio
async def test_idle_stop_failure_retains_engine_and_memory(monkeypatch):
    class FailingEngine(_Engine):
        async def stop(self, **_kwargs):
            raise RuntimeError("stop failed")

    pool = EnginePool()
    entry = _entry(FailingEngine())
    entry.runtime_settle_size = 123
    pool._entries[entry.model_id] = entry
    pool._current_model_memory = 123
    monkeypatch.setattr("omlx.engine_pool.mx.get_active_memory", lambda: 0)
    monkeypatch.setattr("omlx.engine_pool.get_phys_footprint", lambda: 0)
    with pytest.raises(RuntimeError, match="stop failed"):
        await pool.unload_if_idle_unpinned(entry.model_id)
    assert entry.engine is not None
    assert pool._current_model_memory == 123


@pytest.mark.asyncio
async def test_idle_route_refuses_without_pool_or_when_busy():
    app = FastAPI()
    app.post("/v1/models/{model_id:path}/unload-idle")(server.unload_model_if_idle)
    app.dependency_overrides[server.verify_api_key] = lambda: True
    old = server._server_state.engine_pool
    try:
        server._server_state.engine_pool = None
        with TestClient(app) as client:
            assert client.post("/v1/models/model/unload-idle").status_code == 503
        server._server_state.engine_pool = SimpleNamespace(
            unload_if_idle_unpinned=AsyncMock(return_value=False)
        )
        with TestClient(app) as client:
            response = client.post("/v1/models/model/unload-idle")
        assert response.status_code == 409
    finally:
        server._server_state.engine_pool = old
        app.dependency_overrides.pop(server.verify_api_key, None)
