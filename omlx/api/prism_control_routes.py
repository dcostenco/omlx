# SPDX-License-Identifier: Apache-2.0
"""Authenticated control registration, status, cancellation and execution."""

from dataclasses import asdict
from functools import partial
from hashlib import sha256
import asyncio
import copy
import json
import time

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import ValidationError

from .prism_control_coordinator import (
    ControlCoordinator,
    CoordinatorClosed,
    ExecutionExists,
    acquire_execution_lock,
)
from .prism_control_store import (
    ControlClaim,
    ControlStore,
    ControlStoreError,
    ControlConflict,
    ControlCapacity,
)
from .responses_models import ResponsesRequest
from ..owned_images import validate_owned_images
from ..owned_image_policy import capture_image_policy
from .utils import find_lone_surrogate

MAX_CREDENTIAL_BYTES = 4096
CONTROL_PREFIX = "/prism-control/v1"
MAX_REGISTRATION_BYTES = 256 * 1024
MAX_CONTROL_OUTPUT_TOKENS = 8192
BODY_TIMEOUT_SECONDS = 3
MAX_ADMISSION_TIMEOUT_MS = 120_000
EXPECTED_SETTINGS_FIELD = "expected_settings_sha256"
SETTINGS_DIGEST_LENGTH = 64
SETTINGS_DIGEST_CHARACTERS = frozenset("0123456789abcdef")
REFUSALS = {
    "conflict": (409, "Control execution refused"),
    "timeout": (408, "Control request timed out"),
    "invalid": (400, "Invalid control request"),
    "unavailable": (503, "Control state unavailable"),
}


def registration_payload(record):
    if isinstance(record, ControlClaim):
        # The trusted claim handle (worker and engine request IDs) never
        # reaches a response body.
        raise TypeError("control_claim_not_serializable")
    payload = asdict(record)
    payload.pop("owner_id")
    payload["store_generation_id"] = payload.pop("generation_id")
    return {
        **payload,
        "engine_generation_verified": False,
        "provider_qualified": False,
        "production_enabled": False,
        "qualification_scope": "registration_only_not_input_capability",
    }


def canonical_digest(value):
    return sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def reject_nonfinite(_value):
    raise ValueError("control_request_invalid")


async def read_json_body(request, check_deadline):
    """Bounded in size and time; raises HTTPException, TimeoutError or ValueError."""
    chunks = bytearray()
    async with asyncio.timeout(BODY_TIMEOUT_SECONDS):
        async for chunk in request.stream():
            check_deadline()
            if len(chunks) + len(chunk) > MAX_REGISTRATION_BYTES:
                raise HTTPException(status_code=413, detail="Control request too large")
            chunks.extend(chunk)
    check_deadline()
    return json.loads(chunks, parse_constant=reject_nonfinite)


def parse_control_request(raw):
    """The registered request and its canonical digest, or ValueError."""
    if not isinstance(raw, dict):
        raise ValueError()
    # The tokenizer cannot encode a lone surrogate, and the digest would fail
    # with UnicodeEncodeError (a 500); refuse it here as an invalid request.
    if find_lone_surrogate(raw) is not None:
        raise ValueError()
    cap = raw.get("max_output_tokens")
    if type(cap) is not int or not 1 <= cap <= MAX_CONTROL_OUTPUT_TOKENS:
        raise ValueError()
    # Validate before Pydantic can serialize dict tool outputs to text and
    # before either registration or execution can reach VLM preflight.
    validate_owned_images(raw.get("input"))
    parsed = ResponsesRequest.model_validate(raw)
    if (
        parsed.previous_response_id is not None
        or parsed.conversation is not None
        or parsed.background
    ):
        raise ValueError()
    return parsed, canonical_digest(parsed.model_dump(mode="json"))


def settings_digest(get_settings_snapshot, model_id):
    snapshot = get_settings_snapshot(model_id)
    if not isinstance(snapshot, dict) or not snapshot:
        raise ValueError()
    return canonical_digest(snapshot)


def expected_settings_digest(body):
    """An optional client restriction, never server capability authority."""
    if EXPECTED_SETTINGS_FIELD not in body:
        return None
    expected = body[EXPECTED_SETTINGS_FIELD]
    if (
        type(expected) is not str
        or len(expected) != SETTINGS_DIGEST_LENGTH
        or any(character not in SETTINGS_DIGEST_CHARACTERS for character in expected)
    ):
        raise ValueError()
    return expected


def settings_changed_response():
    return JSONResponse(
        status_code=409,
        content={"error": {"code": "settings_changed", "message": "Control settings changed"}},
        headers={"X-Prism-Control-Error": "settings_changed", "Cache-Control": "no-store"},
    )


def server_event(event, data):
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return f"event: {event}\ndata: {payload}\n\n"


def configure_control_store(state, base_path, is_loopback_bind):
    """Called only at server setup. Failure leaves ordinary inference intact."""
    state.prism_control_store = None
    if not state.api_key or not is_loopback_bind(state.bind_host):
        return
    try:
        state.prism_control_store = ControlStore(base_path / "prism-control")
    except ControlStoreError:
        # No raw path, credential or database exception in the API/logs.
        return


def configure_control_coordinator(state, get_settings_snapshot, engine_stage=None):
    """Called at server setup after the store; only the lock holder executes."""
    state.prism_control_coordinator = None
    store = state.prism_control_store
    if store is None:
        return
    lock = acquire_execution_lock(store.path.parent)
    if lock is None:
        return
    state.prism_control_coordinator = ControlCoordinator(
        store,
        partial(settings_digest, get_settings_snapshot),
        engine_stage,
        lock=lock,
    )


def create_control_router(
    get_state, verify_credentials, is_loopback_bind, get_settings_snapshot=None
):
    """Callbacks are server wiring, never request-supplied authorization."""
    router = APIRouter(prefix=CONTROL_PREFIX, tags=["prism-control"])
    security = HTTPBearer(auto_error=False)

    def authenticated_owner(
        request: Request,
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ):
        state = get_state()
        if not is_loopback_bind(state.bind_host):
            raise HTTPException(
                status_code=403, detail="Control requires loopback bind"
            )
        key = (
            credentials.credentials
            if credentials is not None
            else request.headers.get("x-api-key")
        )
        settings = state.global_settings
        sub_keys = settings.auth.sub_keys if settings is not None else []
        if (
            not isinstance(state.api_key, str)
            or not state.api_key
            or not isinstance(key, str)
            or not key
            or len(key.encode("utf-8", "surrogatepass")) > MAX_CREDENTIAL_BYTES
            or not verify_credentials(key, state.api_key, sub_keys)
        ):
            raise HTTPException(
                status_code=401, detail="Valid control API key required"
            )
        return sha256(key.encode("utf-8", "surrogatepass")).hexdigest(), state

    @router.get("/models/{model_id:path}/image-input")
    def image_input(model_id: str, actor=Depends(authenticated_owner)):
        try:
            snapshot = get_settings_snapshot(model_id)
            capability = snapshot.get("owned_image_input")
            if not isinstance(capability, dict) or capability.get("model_id") != model_id:
                raise ValueError()
            return {**capability, "settings_sha256": canonical_digest(snapshot)}
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=503, detail="Control image capability unavailable") from None

    @router.get("/jobs/{job_id}/operations/{operation_id}")
    def registration_status(
        job_id: str, operation_id: str, actor=Depends(authenticated_owner)
    ):
        owner_id, state = actor
        store = state.prism_control_store
        if store is None:
            raise HTTPException(status_code=503, detail="Control state unavailable")
        try:
            record = store.lookup_status(owner_id, job_id, operation_id)
        except ValueError:
            raise HTTPException(
                status_code=400, detail="Invalid control identity"
            ) from None
        except ControlStoreError:
            raise HTTPException(
                status_code=503, detail="Control state unavailable"
            ) from None
        if record is None:
            raise HTTPException(status_code=404, detail="Control operation unknown")
        return registration_payload(record)

    @router.post("/jobs/{job_id}/operations/{operation_id}/cancel")
    async def request_cancel(
        job_id: str, operation_id: str, actor=Depends(authenticated_owner)
    ):
        owner_id, state = actor
        store = state.prism_control_store
        if store is None:
            raise HTTPException(status_code=503, detail="Control state unavailable")
        try:
            # 1. Commit the tombstone, off the event loop.
            record = await asyncio.get_running_loop().run_in_executor(
                None, store.request_cancel, owner_id, job_id, operation_id
            )
        except ValueError:
            raise HTTPException(
                status_code=400, detail="Invalid control identity"
            ) from None
        except ControlCapacity:
            raise HTTPException(
                status_code=409, detail="Control cancellation capacity"
            ) from None
        except ControlStoreError:
            raise HTTPException(
                status_code=503, detail="Control state unavailable"
            ) from None
        # 2. Fence any live execution, 3. only then acknowledge.
        coordinator = getattr(state, "prism_control_coordinator", None)
        if coordinator is not None:
            try:
                coordinator.fence((owner_id, job_id, operation_id))
            except Exception:
                # Committed but not fenced; a retried cancel fences again.
                raise HTTPException(
                    status_code=503, detail="Control cancellation incomplete"
                ) from None
        return registration_payload(record)

    if get_settings_snapshot is not None:

        @router.put("/jobs/{job_id}/operations/{operation_id}")
        async def register(
            job_id: str,
            operation_id: str,
            request: Request,
            actor=Depends(authenticated_owner),
        ):
            owner_id, state = actor
            store = state.prism_control_store
            if store is None:
                raise HTTPException(status_code=503, detail="Control state unavailable")
            loop = asyncio.get_running_loop()
            try:
                # Validate the identity before reading any request body or model
                # settings. Store calls run off the event loop: another
                # request's claim or finish may hold the store lock.
                await loop.run_in_executor(
                    None, store.lookup, owner_id, job_id, operation_id
                )
                body_deadline = loop.time() + BODY_TIMEOUT_SECONDS

                def check_deadline():
                    if loop.time() >= body_deadline:
                        raise HTTPException(
                            status_code=408, detail="Control request timed out"
                        )

                body = await read_json_body(request, check_deadline)
                if (
                    not isinstance(body, dict)
                    or "request" not in body
                    or not set(body) <= {"request", EXPECTED_SETTINGS_FIELD}
                    or not isinstance(body["request"], dict)
                ):
                    raise ValueError()
                expected_hash = expected_settings_digest(body)
                parsed, request_hash = parse_control_request(body["request"])
                check_deadline()
            except TimeoutError:
                raise HTTPException(
                    status_code=408, detail="Control request timed out"
                ) from None
            except (ValueError, ValidationError, RecursionError):
                raise HTTPException(
                    status_code=400, detail="Invalid control request"
                ) from None
            except ControlStoreError:
                raise HTTPException(
                    status_code=503, detail="Control state unavailable"
                ) from None
            try:
                settings_hash = settings_digest(get_settings_snapshot, parsed.model)
                check_deadline()
            except HTTPException:
                raise
            except Exception:
                raise HTTPException(
                    status_code=503, detail="Control settings unavailable"
                ) from None
            if expected_hash is not None and expected_hash != settings_hash:
                return settings_changed_response()
            try:
                check_deadline()
                record = await loop.run_in_executor(
                    None,
                    partial(
                        store.register,
                        owner_id=owner_id,
                        job_id=job_id,
                        operation_id=operation_id,
                        request_sha256=request_hash,
                        model_id=parsed.model,
                        settings_sha256=settings_hash,
                        admission_check=check_deadline,
                    ),
                )
            except (ControlConflict, ControlCapacity):
                raise HTTPException(
                    status_code=409, detail="Control registration refused"
                ) from None
            except ValueError:
                raise HTTPException(
                    status_code=400, detail="Invalid control request"
                ) from None
            except ControlStoreError:
                raise HTTPException(
                    status_code=503, detail="Control state unavailable"
                ) from None
            check_deadline()
            return registration_payload(record)

        @router.post("/jobs/{job_id}/operations/{operation_id}/execute")
        async def execute(
            job_id: str,
            operation_id: str,
            request: Request,
            actor=Depends(authenticated_owner),
        ):
            # The admission budget runs from entry, so the body read counts.
            entered = time.monotonic()
            owner_id, state = actor
            store = state.prism_control_store
            coordinator = getattr(state, "prism_control_coordinator", None)
            if store is None:
                raise HTTPException(status_code=503, detail="Control state unavailable")
            loop = asyncio.get_running_loop()
            try:
                await loop.run_in_executor(
                    None, store.lookup, owner_id, job_id, operation_id
                )
            except ValueError:
                raise HTTPException(
                    status_code=400, detail="Invalid control identity"
                ) from None
            except ControlStoreError:
                raise HTTPException(
                    status_code=503, detail="Control state unavailable"
                ) from None
            if coordinator is None:
                raise HTTPException(
                    status_code=503, detail="Control execution unavailable"
                )
            body_deadline = loop.time() + BODY_TIMEOUT_SECONDS

            def check_deadline():
                if loop.time() >= body_deadline:
                    raise HTTPException(status_code=408, detail="Control request timed out")

            try:
                body = await read_json_body(request, check_deadline)
                if (
                    not isinstance(body, dict)
                    or not {"request", "admission_timeout_ms"} <= set(body)
                    or not set(body) <= {"request", "admission_timeout_ms", EXPECTED_SETTINGS_FIELD}
                    or not isinstance(body["request"], dict)
                ):
                    raise ValueError()
                expected_hash = expected_settings_digest(body)
                budget = body["admission_timeout_ms"]
                if type(budget) is not int or not 1 <= budget <= MAX_ADMISSION_TIMEOUT_MS:
                    raise ValueError()
                parsed, request_hash = parse_control_request(body["request"])
                check_deadline()
            except TimeoutError:
                raise HTTPException(
                    status_code=408, detail="Control request timed out"
                ) from None
            except (ValueError, ValidationError, RecursionError):
                raise HTTPException(
                    status_code=400, detail="Invalid control request"
                ) from None
            try:
                # Actual digests and capability authority come from this
                # request and current settings. The optional client digest
                # can only restrict which current settings are accepted.
                settings_snapshot = copy.deepcopy(get_settings_snapshot(parsed.model))
                if not isinstance(settings_snapshot, dict) or not settings_snapshot:
                    raise ValueError()
                settings_hash = canonical_digest(settings_snapshot)
                image_policy = capture_image_policy(parsed.model, settings_snapshot, settings_hash,
                    lambda model_id: settings_digest(get_settings_snapshot, model_id))
            except HTTPException:
                raise
            except Exception:
                raise HTTPException(
                    status_code=503, detail="Control settings unavailable"
                ) from None
            if expected_hash is not None and expected_hash != settings_hash:
                return settings_changed_response()
            try:
                execution = coordinator.start(
                    (owner_id, job_id, operation_id),
                    request_sha256=request_hash,
                    model_id=parsed.model,
                    settings_sha256=settings_hash,
                    deadline=entered + budget / 1000,
                    request=parsed,
                    image_policy=image_policy,
                )
            except ExecutionExists:
                raise HTTPException(
                    status_code=409, detail="Control execution exists"
                ) from None
            except CoordinatorClosed:
                raise HTTPException(
                    status_code=503, detail="Control execution unavailable"
                ) from None
            # Only the execution's queue is read here: the coordinator task is
            # not a child of this request, so a disconnect never cancels it.
            event, payload = await execution.next_event()
            if event == "refused":
                status_code, detail = REFUSALS[payload]
                raise HTTPException(status_code=status_code, detail=detail)

            async def events():
                yield server_event("claimed", {"state": "claimed"})
                while True:
                    event, payload = await execution.next_event()
                    if event == "terminal":
                        yield server_event("terminal", registration_payload(payload))
                        return
                    if event == "unresolved":
                        yield server_event("unresolved", {"state": "claimed"})
                        return
                    # Output events from owned streaming (27.4b), e.g. "delta".
                    yield server_event(event, payload)

            return StreamingResponse(
                events(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-store"},
            )

    return router
