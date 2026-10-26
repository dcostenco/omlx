"""Real pool lease/exclusion logic with synthetic engines, never model weights."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import omlx.engine_pool as module
from omlx.engine_pool import EngineEntry, EnginePool
from omlx.exceptions import ModelBusyError
from omlx.owned_preparation import OwnedPreparationRefused
from omlx import owned_preparation as preparation
from omlx.owned_preparation import ExecutionUnresolved
from omlx.api.responses_models import ResponsesRequest
from omlx.owned_image_policy import OwnedImagePolicy


def make_pool():
    assert (
        Path(module.__file__).resolve()
        == Path(__file__).resolve().parents[1] / "omlx/engine_pool.py"
    )
    pool = EnginePool.__new__(EnginePool)
    entry = EngineEntry(
        "vision-fixture",
        "/fixture",
        "vlm",
        "vlm",
        1,
        config_model_type="qwen3_5",
        engine=object(),
        runtime_settings_signature=(),
    )
    pool._entries = {entry.model_id: entry}
    pool._settings_manager = SimpleNamespace(get_settings=lambda _: SimpleNamespace())
    pool._unloading_models = set()
    pool._lock = asyncio.Lock()
    pool._lease_release_tasks = set()
    pool._owned_prepare_tasks = set()
    pool._distributed_deployment_for_entry = lambda _: None
    pool._entry_is_diffusion_model = lambda _: False
    pool._engine_runtime_signature = lambda *_: ()
    pool._validate_llm_engine_ready = lambda *_: None
    pool._owned_identity_holds = lambda *_: True
    pool._owned_model_refusal = AsyncMock(return_value=None)
    pool._owned_instance_refusal = AsyncMock(return_value=None)
    pool._reloadable_failed_engine = AsyncMock(return_value=False)
    pool._check_splash_launcher = AsyncMock()
    return pool, entry


async def fence():
    return None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        ExecutionUnresolved("absence_unproved"),
        ExecutionUnresolved("collector_missing"),
        RuntimeError("unexpected probe failure"),
    ],
)
async def test_unresolved_image_stage_keeps_actual_pool_exclusion(monkeypatch, failure):
    pool, entry = make_pool()
    # A valid Pillow-generated PNG avoids making decoder rejection the cause.
    from PIL import Image
    from io import BytesIO
    import base64

    encoded = BytesIO()
    Image.new("RGB", (1, 1)).save(encoded, format="PNG")
    png = base64.b64encode(encoded.getvalue()).decode()
    request = ResponsesRequest.model_validate(
        {
            "model": entry.model_id,
            "max_output_tokens": 5,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": "data:image/png;base64," + png,
                        }
                    ],
                }
            ],
        }
    )
    execution = SimpleNamespace(
        model_id=entry.model_id,
        request=request,
        settings_sha256="a" * 64,
        image_policy=OwnedImagePolicy(
            entry.model_id, "a" * 64, True, lambda _: "a" * 64
        ),
        cancellations=[],
        lease=None,
        preparation_handle=None,
        verify=fence,
    )

    async def unresolved(*_):
        raise failure

    monkeypatch.setattr(preparation, "_admit", unresolved)
    stage = preparation.make_admission_stage(
        lambda: pool, None, preparation.owned_request_refusal
    )
    with pytest.raises(type(failure)):
        await stage(execution)
    assert (
        execution.lease is not None and entry.in_use == 1 and entry.owned_image_active
    )
    with pytest.raises(ModelBusyError):
        await pool.get_engine(entry.model_id, _lease=True)
    # Synthetic admission never dispatched; test-only fixture retirement.
    await pool.release_engine(entry.model_id)


@pytest.mark.asyncio
async def test_enabled_image_policy_preserves_text_tool_output_without_claim_wedge(
    monkeypatch,
):
    pool, entry = make_pool()
    request = ResponsesRequest.model_validate(
        {
            "model": entry.model_id,
            "max_output_tokens": 5,
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "fixture-call",
                    "output": "text",
                }
            ],
        }
    )
    execution = SimpleNamespace(
        model_id=entry.model_id,
        request=request,
        settings_sha256="a" * 64,
        image_policy=OwnedImagePolicy(
            entry.model_id, "a" * 64, True, lambda _: "a" * 64
        ),
        cancellations=[],
        lease=None,
        preparation_handle=None,
        verify=fence,
    )
    admit = AsyncMock(
        return_value=("not_started", ("admission_refused", "stream_synchronized"))
    )
    monkeypatch.setattr(preparation, "_admit", admit)
    stage = preparation.make_admission_stage(
        lambda: pool, None, preparation.owned_request_refusal
    )
    assert await stage(execution) == admit.return_value
    assert admit.await_count == 1 and entry.in_use == 0 and not entry.owned_image_active


@pytest.mark.asyncio
async def test_image_lease_blocks_ordinary_acquisition_until_actual_release():
    pool, entry = make_pool()
    lease = await pool.acquire_owned_image(entry.model_id, fence=fence)
    assert (
        lease.engine is entry.engine and entry.in_use == 1 and entry.owned_image_active
    )
    with pytest.raises(ModelBusyError):
        await pool.get_engine(entry.model_id, _lease=True)
    assert entry.in_use == 1
    await pool.release_engine(lease.model_id)
    assert entry.in_use == 0 and not entry.owned_image_active
    assert await pool.get_engine(entry.model_id, _lease=True) is entry.engine
    await pool.release_engine(entry.model_id)


@pytest.mark.asyncio
async def test_existing_ordinary_lease_is_not_aborted_or_borrowed():
    pool, entry = make_pool()
    await pool.get_engine(entry.model_id, _lease=True)
    with pytest.raises(OwnedPreparationRefused, match="image_busy"):
        await pool.acquire_owned_image(entry.model_id, fence=fence)
    assert entry.in_use == 1 and not entry.owned_image_active
    await pool.release_engine(entry.model_id)


@pytest.mark.asyncio
async def test_ordinary_acquisition_during_owned_checks_fences_image_preparation():
    pool, entry = make_pool()

    async def another_request(_):
        await pool.get_engine(entry.model_id, _lease=True)
        return None

    pool._owned_model_refusal = another_request
    with pytest.raises(OwnedPreparationRefused, match="image_busy"):
        await pool.acquire_owned_image(entry.model_id, fence=fence)
    assert entry.in_use == 1 and not entry.owned_image_active
    await pool.release_engine(entry.model_id)


@pytest.mark.asyncio
async def test_other_owned_work_refuses_before_taking_a_second_lease():
    pool, entry = make_pool()
    await pool.acquire_owned_image(entry.model_id, fence=fence)
    for operation in (pool.acquire_owned, pool.acquire_owned_image):
        with pytest.raises(OwnedPreparationRefused, match="image_busy"):
            await operation(entry.model_id, fence=fence)
        assert entry.in_use == 1
    await pool.release_engine(entry.model_id)


@pytest.mark.asyncio
async def test_ordinary_text_leases_keep_existing_sharing_behavior():
    pool, entry = make_pool()
    first = await pool.get_engine(entry.model_id, _lease=True)
    second = await pool.get_engine(entry.model_id, _lease=True)
    assert first is second and entry.in_use == 2 and not entry.owned_image_active
    await pool.release_engine(entry.model_id)
    await pool.release_engine(entry.model_id)


@pytest.mark.asyncio
async def test_cancelled_acquisition_releases_without_leaving_exclusion():
    pool, entry = make_pool()
    entered = asyncio.Event()
    calls = 0

    async def blocked_fence():
        nonlocal calls
        calls += 1
        if calls == 2:
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(
        pool.acquire_owned_image(entry.model_id, fence=blocked_fence)
    )
    try:
        await asyncio.wait_for(entered.wait(), 1)
        assert entry.in_use == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert entry.in_use == 0 and not entry.owned_image_active
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_non_vision_entry_cannot_get_image_exclusion():
    pool, entry = make_pool()
    entry.model_type = "llm"
    with pytest.raises(OwnedPreparationRefused, match="image_engine"):
        await pool.acquire_owned_image(entry.model_id, fence=fence)
    assert entry.in_use == 0 and not entry.owned_image_active
