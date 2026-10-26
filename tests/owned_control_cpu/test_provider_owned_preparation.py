# SPDX-License-Identifier: Apache-2.0
"""Slice 27.3 CPU proof: owned preparation, exact-model acquisition without fallback.

The real EnginePool methods (acquire_owned and its helpers, get_engine and its
extracted locked helper, the lease release) run against a stub pool whose
loader, memory readers and engines are stubs; the real preparation stage, the
real coordinator and a real control store run around them. Metal loads,
eviction under real memory pressure and engine start are not observable here.
"""

import __future__
import ast
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import importlib.abc
import inspect
import logging
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
from typing import Literal
import unittest
from unittest import mock

ROOT = Path(os.environ["OMLX_CONTROL_ROOT"])


class ForbidEngines(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (
            fullname in ("mlx", "omlx.server", "omlx.engine_pool")
            or fullname.startswith("mlx.")
        ):
            raise RuntimeError("engine_import_forbidden")


sys.meta_path.insert(0, ForbidEngines())
sys.path.insert(0, str(ROOT))
from omlx import exceptions as pool_errors
from omlx import owned_preparation as prep
from omlx.api import prism_control_coordinator as coord
from omlx.api.prism_control_store import ControlStore, ControlStoreError
import omlx_get_engine_ca93487 as baseline

FUTURE = __future__.annotations.compiler_flag
ENGINE_THREADS = ThreadPoolExecutor(1, thread_name_prefix="engine-mlx")


class FakeScheduler:
    def __init__(self, refusal=None):
        self.refusal = refusal
        self.threads = []

    def _owned_model_refusal(self):
        self.threads.append(threading.current_thread().name)
        if isinstance(self.refusal, BaseException):
            raise self.refusal
        return self.refusal


class BaseEngine:
    def __init__(self, model_id, refusal=None):
        self.model_id = model_id
        self.scheduler = FakeScheduler(refusal)
        self._mlx_executor = ENGINE_THREADS


class BatchedEngine(BaseEngine):
    pass


class VLMBatchedEngine(BaseEngine):
    pass


class DistributedBatchedEngine(BatchedEngine):
    pass


class OtherEngine:
    pass


# The real EngineEntry, the reason table and the EnginePool methods under test.
POOL_TREE = ast.parse((ROOT / "omlx/engine_pool.py").read_text())
POOL_CLASS = next(
    n for n in POOL_TREE.body if isinstance(n, ast.ClassDef) and n.name == "EnginePool"
)
REAL = {
    "get_engine",
    "_get_engine_locked",
    "_acquire_loaded_engine",
    "_owned_fence",
    "_raise_owned_refusal",
    "_release_owned_locked",
    "_owned_identity_holds",
    "_acquire_owned_locked",
    "_owned_model_refusal",
    "_owned_instance_refusal",
    "acquire_owned",
    "_release_engine_lease",
    "release_engine",
    "_finish_lease_release_task",
    "_resolve_engine_core_from_engine",
    "_entry_is_diffusion_model",
}
real_methods = [
    n
    for n in POOL_CLASS.body
    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in REAL
]
assert {m.name for m in real_methods} == REAL, REAL - {m.name for m in real_methods}
module_nodes = [
    n
    for n in POOL_TREE.body
    if (isinstance(n, ast.ClassDef) and n.name == "EngineEntry")
    or (isinstance(n, ast.FunctionDef) and n.name == "_owned_load_refusal_reason")
]
assert len(module_nodes) == 2


def class_node(name, body):
    return ast.ClassDef(
        name=name, bases=[], keywords=[], body=body, decorator_list=[], type_params=[]
    )


namespace = dict(
    asyncio=asyncio,
    time=time,
    logger=logging.getLogger("owned-preparation-test"),
    mx=SimpleNamespace(get_active_memory=lambda: 0),
    _settled_phys_footprint=lambda: 0,
    format_size=str,
    DEFAULT_CEILING_ADVICE="advice",
    _ADMISSION_CEILING_RECHECKS=0,
    _ADMISSION_CEILING_RECHECK_S=0.0,
    ModelNotFoundError=pool_errors.ModelNotFoundError,
    ModelBusyError=pool_errors.ModelBusyError,
    ModelLoadingError=pool_errors.ModelLoadingError,
    ModelUnavailableError=pool_errors.ModelUnavailableError,
    ModelTooLargeError=pool_errors.ModelTooLargeError,
    InsufficientMemoryError=pool_errors.InsufficientMemoryError,
    BaseEngine=BaseEngine,
    BatchedEngine=BatchedEngine,
    VLMBatchedEngine=VLMBatchedEngine,
    EmbeddingEngine=OtherEngine,
    RerankerEngine=OtherEngine,
    STTEngine=OtherEngine,
    STSEngine=OtherEngine,
    TTSEngine=OtherEngine,
    OwnedLease=prep.OwnedLease,
    OwnedPreparationRefused=prep.OwnedPreparationRefused,
    owned_entry_refusal=prep.owned_entry_refusal,
    owned_family=prep.owned_family,
    owned_loaded_variant_refusal=prep.owned_loaded_variant_refusal,
    submission_failures=prep.submission_failures,
    # Slice 28's loaded-instance check, stubbed per engine (test_provider_flash_next_owned).
    qwen4_instance_refusal_for_engine=lambda engine: getattr(engine, "instance_refusal", None),
    dataclass=dataclass,
    field=field,
    Literal=Literal,
)
exec(
    compile(
        ast.fix_missing_locations(
            ast.Module(body=[*module_nodes, class_node("RealPool", real_methods)], type_ignores=[])
        ),
        str(ROOT / "omlx/engine_pool.py"),
        "exec",
        flags=FUTURE,
        dont_inherit=True,
    ),
    namespace,
)
EngineEntry = namespace["EngineEntry"]
RealPool = namespace["RealPool"]

# The verbatim get_engine of oMLX ca93487, the differential baseline.
BASELINE_TREE = ast.parse(Path(baseline.__file__).read_text())
baseline_class = next(n for n in BASELINE_TREE.body if isinstance(n, ast.ClassDef))
exec(
    compile(
        ast.fix_missing_locations(
            ast.Module(body=[class_node("BaselineGetEngine", baseline_class.body)], type_ignores=[])
        ),
        baseline.__file__,
        "exec",
        flags=FUTURE,
        dont_inherit=True,
    ),
    namespace,
)
BaselineGetEngine = namespace["BaselineGetEngine"]


class CountingLock:
    def __init__(self):
        self._lock = asyncio.Lock()
        self.acquired = 0

    async def __aenter__(self):
        await self._lock.acquire()
        self.acquired += 1

    async def __aexit__(self, *exc):
        self._lock.release()

    def locked(self):
        return self._lock.locked()


SETTINGS_DEFAULTS = dict(
    variant="v1",
    external_engine=None,
    dflash_enabled=False,
    mtp_enabled=False,
    vlm_mtp_enabled=False,
    specprefill_enabled=False,
    qwen35_ane_prefill_enabled=False,
)


class StubPool:
    """Pool state and every helper the real methods call, as stubs."""

    def __init__(self):
        self._entries = {}
        self._lock = CountingLock()
        self.settings = {}
        self._settings_manager = SimpleNamespace(
            get_settings=lambda model_id: self.settings.get(model_id)
        )
        self._unloading_models = set()
        self._failed_load_reclaim_task = None
        self._owned_prepare_tasks = set()
        self._lease_release_tasks = set()
        self._current_model_memory = 0
        self.distributed = set()
        self.loads = []
        self.unloads = []
        self.releases = []
        self.load_plan = "ok"
        self.load_gate = None
        self.loading = asyncio.Event()
        self.reclaim_seconds = 0.0
        self.scheduler_refusal = None
        self.release_gate = None
        self.releasing = asyncio.Event()

    def add(self, model_id="m", engine_type="batched", model_type="llm", **settings):
        entry = EngineEntry(
            model_id=model_id,
            model_path=f"/models/{model_id}",
            model_type=model_type,
            engine_type=engine_type,
            estimated_size=1,
        )
        self._entries[model_id] = entry
        self.settings[model_id] = SimpleNamespace(**(SETTINGS_DEFAULTS | settings))
        return entry

    def preload(self, model_id="m", engine_class=BatchedEngine, variant=None):
        entry = self._entries[model_id]
        entry.engine = engine_class(model_id)
        settings = self.settings[model_id]
        entry.runtime_settings_signature = (("variant", variant or settings.variant),)
        return entry.engine

    async def _reloadable_failed_engine(self, model_id):
        return False

    async def _check_splash_launcher(self, model_id, runtime_settings):
        return None

    def _engine_runtime_signature(self, model_id, runtime_settings=None):
        settings = runtime_settings if runtime_settings is not None else self.settings.get(model_id)
        return None if settings is None else (("variant", settings.variant),)

    @staticmethod
    def _runtime_failed_engine(engine):
        return False

    def _validate_llm_engine_ready(self, model_id, engine):
        return None

    def _distributed_deployment_for_entry(self, entry):
        return object() if entry.model_id in self.distributed else None

    def _raise_if_model_path_missing_locked(self, model_id, entry):
        return None

    def _raise_if_load_failed(self, model_id, entry):
        return None

    def _raise_if_reload_busy(self, entry, operation):
        return None

    def _has_mlx_lm_path(self, entry):
        return True

    def _entry_resident_size(self, entry):
        return entry.estimated_size

    def _entry_runtime_resident_size(self, entry, settings, base_size=0, **kwargs):
        return base_size

    def _effective_qwen4_model_settings(self, entry, settings, ceiling=None):
        return settings

    def _effective_deepseek_v41_model_settings(self, entry, settings, ceiling=None):
        return settings

    def _current_ceiling(self):
        return 0

    def _fallback_admission_ceiling(self):
        return 0

    def _admission_soft_target(self):
        return 0

    def _dynamic_ceiling_binds(self):
        return False

    def _find_lru_victim(self):
        return None

    async def _unload_engine(self, model_id):
        self.unloads.append(model_id)
        entry = self._entries[model_id]
        entry.engine = None
        entry.runtime_settings_signature = None

    async def _unload_pending_if_idle_locked(self, model_id):
        return False

    async def _reclaim(self):
        await asyncio.sleep(self.reclaim_seconds)

    async def _load_engine(self, model_id, force_lm=False, runtime_settings=None):
        self.loads.append((model_id, force_lm, runtime_settings))
        self.loading.set()
        if self.load_gate is not None:
            await self.load_gate.wait()
        entry = self._entries[model_id]
        plan = self.load_plan
        if plan in ("fail", "abort"):
            self.failed_at = time.monotonic()
            task = asyncio.get_running_loop().create_task(self._reclaim())
            self._failed_load_reclaim_task = task
            error = pool_errors.ModelUnavailableError if plan == "fail" else pool_errors.ModelLoadingError
            raise error(model_id, "synthetic")
        if plan == "fail_without_reclaim":
            raise pool_errors.ModelUnavailableError(model_id, "synthetic")
        if plan == "vlm_to_llm":
            entry.model_type, entry.engine_type = "llm", "batched"
            entry.vlm_fallback_rewritten = True
            engine = BatchedEngine(model_id, self.scheduler_refusal)
        elif plan == "types_changed":
            # An admin type override applied while the model loads.
            entry.model_type, entry.engine_type = "vlm", "vlm"
            engine = BatchedEngine(model_id, self.scheduler_refusal)
        elif plan == "distributed_class":
            engine = DistributedBatchedEngine(model_id, self.scheduler_refusal)
        else:
            effective = "batched" if force_lm else entry.engine_type
            engine_class = {"batched": BatchedEngine, "vlm": VLMBatchedEngine}[effective]
            engine = engine_class(model_id, self.scheduler_refusal)
        entry.engine = engine
        entry.runtime_settings_signature = self._engine_runtime_signature(model_id, runtime_settings)
        if plan == "settings_change":
            self.settings[model_id].variant = "changed during load"


class Pool(RealPool, StubPool):
    async def release_engine(self, model_id):
        self.releases.append(model_id)
        await RealPool.release_engine(self, model_id)

    async def _release_engine_lease(self, model_id):
        # Inside the pool's own shielded release task, as in production.
        self.releasing.set()
        if self.release_gate is not None:
            await self.release_gate.wait()
        await RealPool._release_engine_lease(self, model_id)


class BaselinePool(BaselineGetEngine, RealPool, StubPool):
    pass


class Fence:
    """The execution's verify: records (pool lock held, loads so far)."""

    def __init__(self, pool, refuse_at=None, block_at=None):
        self.pool = pool
        self.refuse_at = refuse_at
        self.block_at = block_at
        self.blocked = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = []

    async def __call__(self):
        self.calls.append((self.pool._lock.locked(), len(self.pool.loads)))
        if self.block_at == len(self.calls):
            self.blocked.set()
            await self.release.wait()
        if self.refuse_at == len(self.calls):
            raise RuntimeError("synthetic fence refusal")


def run(coroutine):
    return asyncio.run(asyncio.wait_for(coroutine, 20))


async def until(predicate, seconds=5):
    deadline = time.monotonic() + seconds
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.005)


def refused(test, coroutine):
    with test.assertRaises(prep.OwnedPreparationRefused) as caught:
        run(coroutine)
    return caught.exception


def execution_for(fence, model_id="m"):
    return SimpleNamespace(
        model_id=model_id, verify=fence, cancellations=[], lease=None, drain_deadline=None
    )


class PreLoadRefusalTests(unittest.TestCase):
    def test_each_refusal_takes_no_lock_and_loads_nothing(self):
        cases = [
            ("external_engine", dict(external_engine="splash"), None),
            ("dflash", dict(dflash_enabled=True), None),
            ("mtp", dict(mtp_enabled=True), None),
            ("vlm_mtp", dict(vlm_mtp_enabled=True), None),
            ("specprefill", dict(specprefill_enabled=True), None),
            ("ane_prefill", dict(qwen35_ane_prefill_enabled=True), None),
            ("distributed", {}, lambda pool, entry: pool.distributed.add("m")),
            ("diffusion", {}, lambda pool, entry: setattr(entry, "config_model_type", "diffusion-gemma")),
            # Slice 28: qwen4_exp is admitted only with its exact signature allowlist.
            ("variant_unreviewed", {}, lambda pool, entry: setattr(entry, "config_model_type", "qwen4-exp")),
            ("pressure_override", {}, lambda pool, entry: setattr(entry, "config_model_type", "deepseek_v41")),
            ("engine_type", {}, lambda pool, entry: setattr(entry, "engine_type", "embedding")),
            ("fallback_engine", {}, lambda pool, entry: setattr(entry, "vlm_fallback_rewritten", True)),
            ("settings_unavailable", {}, lambda pool, entry: pool.settings.pop("m")),
        ]
        for reason, settings, change in cases:
            with self.subTest(reason=reason, settings=settings):
                pool = Pool()
                entry = pool.add(**settings)
                if change is not None:
                    change(pool, entry)
                fence = Fence(pool)
                error = refused(self, pool.acquire_owned("m", fence=fence))
                self.assertEqual(error.reason, reason)
                self.assertIsNone(error.reclaim)
                self.assertEqual((pool.loads, pool._lock.acquired, entry.in_use), ([], 0, 0))
                self.assertEqual(fence.calls, [(False, 0)])

    def test_an_alias_is_never_resolved(self):
        pool = Pool()
        pool.add("m")
        pool._settings_manager.resolve_model_id = lambda name: "m"
        error = refused(self, pool.acquire_owned("alias-of-m", fence=Fence(pool)))
        self.assertEqual(error.reason, "model_unknown")
        self.assertEqual(pool.loads, [])

    def test_a_fresh_entry_carries_no_fallback_flag(self):
        self.assertFalse(Pool().add().vlm_fallback_rewritten)

    def test_the_real_loader_flags_its_vlm_to_llm_rewrite(self):
        # The stub loader cannot prove the real one sets the flag: pin the
        # source. The rewrite of both types is followed by the flag, and the
        # flag is set nowhere else.
        loader = next(n for n in POOL_CLASS.body if getattr(n, "name", "") == "_load_engine")
        blocks = [
            getattr(node, name) for node in ast.walk(loader)
            for name in ("body", "orelse", "finalbody")
            if isinstance(getattr(node, name, None), list)
        ]
        found = []
        for block in blocks:
            text = [ast.unparse(statement) for statement in block]
            for index, line in enumerate(text):
                if line == "entry.engine_type = 'batched'":
                    found.append(text[index - 1:index + 2])
        self.assertEqual(found, [[
            "entry.model_type = 'llm'",
            "entry.engine_type = 'batched'",
            "entry.vlm_fallback_rewritten = True",
        ]])
        sets = [
            node for node in ast.walk(POOL_TREE)
            if isinstance(node, ast.Assign)
            and any(ast.unparse(t).endswith(".vlm_fallback_rewritten") for t in node.targets)
        ]
        self.assertEqual(len(sets), 1)


class AcquisitionTests(unittest.TestCase):
    def test_owned_loads_use_persisted_settings_and_never_force_lm(self):
        for engine_type, engine_class in (("batched", BatchedEngine), ("vlm", VLMBatchedEngine)):
            with self.subTest(engine_type):
                pool = Pool()
                entry = pool.add(engine_type=engine_type, model_type="vlm" if engine_type == "vlm" else "llm")
                lease = run(pool.acquire_owned("m", fence=Fence(pool)))
                self.assertIs(type(lease.engine), engine_class)
                self.assertEqual(pool.loads, [("m", False, None)])
                self.assertEqual(entry.in_use, 1)

    def test_the_fast_path_leases_a_matching_engine_without_the_lock(self):
        pool = Pool()
        entry = pool.add()
        engine = pool.preload()
        fence = Fence(pool)
        lease = run(pool.acquire_owned("m", fence=fence))
        self.assertIs(lease.engine, engine)
        self.assertEqual((entry.in_use, pool.loads, pool._lock.acquired), (1, [], 0))
        self.assertEqual(fence.calls, [(False, 0), (False, 0)])

    def test_the_fast_path_refuses_a_resident_engine_of_the_wrong_class(self):
        # An ordinary forced-LM request leaves a BatchedEngine on a VLM entry
        # without rewriting the entry's types; owned work must not lease it.
        for engine_type, resident in (("vlm", BatchedEngine), ("batched", VLMBatchedEngine)):
            with self.subTest(engine_type=engine_type):
                pool = Pool()
                entry = pool.add(engine_type=engine_type,
                                 model_type="vlm" if engine_type == "vlm" else "llm")
                pool.preload(engine_class=resident)
                error = refused(self, pool.acquire_owned("m", fence=Fence(pool)))
                self.assertEqual(error.reason, "fallback_engine")
                self.assertEqual(
                    (pool.releases, entry.in_use, pool.loads, pool._lock.acquired),
                    (["m"], 0, [], 0),
                )

    def test_any_error_after_the_lease_releases_it(self):
        pool = Pool()
        entry = pool.add()
        pool.scheduler_refusal = RuntimeError("cannot schedule new futures after shutdown")
        with self.assertRaises(RuntimeError):
            run(pool.acquire_owned("m", fence=Fence(pool)))
        self.assertEqual((pool.releases, entry.in_use), (["m"], 0))

    def test_the_locked_slow_path_never_calls_the_lock_taking_release(self):
        # release_engine can take the pool lock (pending unload); under the
        # lock the slow path must use _release_owned_locked instead.
        method = next(n for n in POOL_CLASS.body if getattr(n, "name", "") == "_acquire_owned_locked")
        calls = {ast.unparse(n.func) for n in ast.walk(method) if isinstance(n, ast.Call)}
        self.assertNotIn("self.release_engine", calls)
        self.assertIn("self._release_owned_locked", calls)

    def test_a_loaded_variant_is_refused_and_never_unloaded(self):
        pool = Pool()
        entry = pool.add()
        engine = pool.preload(variant="profile")
        error = refused(self, pool.acquire_owned("m", fence=Fence(pool)))
        self.assertEqual(error.reason, "variant_loaded")
        self.assertIs(entry.engine, engine)
        self.assertEqual((pool.unloads, pool.loads, entry.in_use), ([], [], 0))

    def test_the_fallback_flag_refuses_on_both_paths(self):
        for loaded in (False, True):
            with self.subTest(loaded=loaded):
                pool = Pool()
                entry = pool.add()
                if loaded:
                    pool.preload()
                entry.vlm_fallback_rewritten = True
                error = refused(self, pool.acquire_owned("m", fence=Fence(pool)))
                self.assertEqual(error.reason, "fallback_engine")
                self.assertEqual((entry.in_use, pool.loads), (0, []))

    def test_pending_unload_and_loading_are_refused_after_the_lock(self):
        for reason in ("pending_unload", "loading"):
            with self.subTest(reason):
                pool = Pool()
                entry = pool.add()
                if reason == "pending_unload":
                    pool._unloading_models.add("m")
                else:
                    entry.is_loading = True
                error = refused(self, pool.acquire_owned("m", fence=Fence(pool)))
                self.assertEqual(error.reason, reason)
                self.assertEqual((pool.loads, entry.in_use, pool._lock.acquired), ([], 0, 1))


class FenceTests(unittest.TestCase):
    def test_fence_order_on_the_slow_path(self):
        pool = Pool()
        pool.add()
        fence = Fence(pool)
        run(pool.acquire_owned("m", fence=fence))
        # Before acquisition; after the pool lock wait; after the load.
        self.assertEqual(fence.calls, [(False, 0), (True, 0), (False, 1)])

    def test_each_fence_refusal_releases_and_stops(self):
        expectations = {
            1: dict(loads=0, lock=0, releases=[]),
            2: dict(loads=0, lock=1, releases=[]),
            3: dict(loads=1, lock=1, releases=["m"]),
        }
        for refuse_at, expected in expectations.items():
            with self.subTest(refuse_at=refuse_at):
                pool = Pool()
                entry = pool.add()
                error = refused(self, pool.acquire_owned("m", fence=Fence(pool, refuse_at=refuse_at)))
                self.assertEqual(error.reason, "fence")
                self.assertEqual(
                    (len(pool.loads), pool._lock.acquired, pool.releases, entry.in_use),
                    (expected["loads"], expected["lock"], expected["releases"], 0),
                )
                self.assertFalse(pool._lock.locked())


class IdentityTests(unittest.TestCase):
    def test_the_vlm_to_llm_rewrite_is_detected_and_released(self):
        pool = Pool()
        entry = pool.add(engine_type="vlm", model_type="vlm")
        pool.load_plan = "vlm_to_llm"
        error = refused(self, pool.acquire_owned("m", fence=Fence(pool)))
        self.assertEqual(error.reason, "fallback_engine")
        self.assertEqual(entry.in_use, 0)
        self.assertTrue(entry.vlm_fallback_rewritten)
        # The flag outlives the engine: the next attempt refuses before loading.
        entry.engine = None
        error = refused(self, pool.acquire_owned("m", fence=Fence(pool)))
        self.assertEqual((error.reason, len(pool.loads)), ("fallback_engine", 1))

    def test_a_subclass_engine_is_never_accepted(self):
        pool = Pool()
        entry = pool.add()
        pool.load_plan = "distributed_class"
        error = refused(self, pool.acquire_owned("m", fence=Fence(pool)))
        self.assertEqual((error.reason, entry.in_use), ("fallback_engine", 0))

    def test_types_changed_during_the_load_are_refused_with_the_right_class(self):
        pool = Pool()
        entry = pool.add()
        pool.load_plan = "types_changed"
        error = refused(self, pool.acquire_owned("m", fence=Fence(pool)))
        self.assertEqual((error.reason, entry.in_use), ("fallback_engine", 0))

    def test_settings_changed_during_the_load_are_caught_without_the_fence(self):
        pool = Pool()
        entry = pool.add()
        pool.load_plan = "settings_change"
        error = refused(self, pool.acquire_owned("m", fence=Fence(pool)))
        self.assertEqual((error.reason, entry.in_use), ("fallback_engine", 0))


class ModelRefusalTests(unittest.TestCase):
    def test_an_entangled_model_is_refused_on_the_engine_thread(self):
        pool = Pool()
        entry = pool.add()
        pool.scheduler_refusal = "block_drafter"
        error = refused(self, pool.acquire_owned("m", fence=Fence(pool)))
        self.assertEqual(error.reason, "model_entangled")
        self.assertEqual((pool.releases, entry.in_use), (["m"], 0))
        threads = entry.engine.scheduler.threads
        self.assertEqual(len(threads), 1)
        self.assertTrue(threads[0].startswith("engine-mlx"), threads)

    def test_the_scheduler_split_keeps_the_request_clause_in_cleanup(self):
        tree = ast.parse((ROOT / "omlx/scheduler.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
        nodes = [
            n for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name in {"_owned_model_refusal", "_owned_running_refusal"}
        ]
        self.assertEqual(len(nodes), 2)
        state = dict(drafter=None)
        names = dict(
            inspect=inspect,
            _mtp_priming=SimpleNamespace(_eligible_host=lambda model: None),
            _model_mtp_decode_enabled=lambda model: False,
            _block_drafter_for=lambda model: state["drafter"],
        )
        exec(compile(ast.fix_missing_locations(
            ast.Module(body=[class_node("Split", nodes)], type_ignores=[])), "scheduler.py", "exec"), names)

        class Scheduler(names["Split"]):
            def __init__(self):
                self.model = object()
                self._vlm_mtp_drafter = None
                self._specprefill_draft_model = None
                self._specprefill_active_request_id = None
                self._step_failed = False
                self._pending_abort_ids = set()
                self._prefill_eviction_awaiting = set()

            def _owned_admission_complete(self, owned, request):
                return True

        owned = SimpleNamespace(_lock=threading.Lock(), _text_only=True, _prepared_vision=False)
        request = SimpleNamespace(rope_deltas=0)
        s = Scheduler()
        self.assertEqual((s._owned_model_refusal(), s._owned_running_refusal(owned, request, "r")), (None, None))
        s._specprefill_active_request_id = "r"
        self.assertIsNone(s._owned_model_refusal())
        self.assertEqual(s._owned_running_refusal(owned, request, "r"), "specprefill")
        self.assertIsNone(s._owned_running_refusal(owned, request, "other"))
        s._specprefill_active_request_id = None
        s._specprefill_draft_model = object()
        self.assertEqual((s._owned_model_refusal(), s._owned_running_refusal(owned, request, "r")),
                         ("specprefill", "specprefill"))
        s._specprefill_draft_model = None
        state["drafter"] = object()
        self.assertEqual((s._owned_model_refusal(), s._owned_running_refusal(owned, request, "r")),
                         ("block_drafter", "block_drafter"))
        # Both entangled and the active SpecPrefill request: the model-level
        # reason still comes first, as before the split.
        s._specprefill_active_request_id = "r"
        self.assertEqual(s._owned_running_refusal(owned, request, "r"), "block_drafter")


class LeaseOwnershipTests(unittest.TestCase):
    def test_cancellations_never_lose_or_double_the_release(self):
        pool = Pool()
        entry = pool.add()
        pool.preload()
        entry.in_use = 1  # an ordinary request already holds the engine
        fence = Fence(pool, block_at=2)
        execution = execution_for(fence)
        stage = prep.make_preparation_stage(lambda: pool)

        async def scenario():
            task = asyncio.create_task(stage(execution))
            await fence.blocked.wait()
            self.assertEqual(entry.in_use, 2)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            fence.release.set()
            with self.assertRaises(coord.EngineStageUnavailable):
                await task

        run(scenario())
        self.assertEqual(pool.releases, ["m"])
        self.assertEqual(entry.in_use, 1)
        self.assertEqual(len(execution.cancellations), 2)
        self.assertIsNone(execution.lease)

    def test_a_slow_path_cancelled_during_the_load_releases_once(self):
        pool = Pool()
        entry = pool.add()
        pool.load_gate = asyncio.Event()
        execution = execution_for(Fence(pool))
        stage = prep.make_preparation_stage(lambda: pool)

        async def scenario():
            task = asyncio.create_task(stage(execution))
            await pool.loading.wait()
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            pool.load_gate.set()
            with self.assertRaises(coord.EngineStageUnavailable):
                await task

        run(scenario())
        self.assertEqual((pool.releases, entry.in_use, len(pool.loads)), (["m"], 0, 1))

    def test_a_failing_release_is_never_retried(self):
        pool = Pool()
        entry = pool.add()
        pool.preload()

        async def failing_release(model_id):
            pool.releases.append(model_id)
            entry.in_use -= 1
            raise RuntimeError("synthetic release failure")

        pool.release_engine = failing_release
        execution = execution_for(Fence(pool))
        with self.assertRaises(RuntimeError):
            run(prep.make_preparation_stage(lambda: pool)(execution))
        self.assertEqual((pool.releases, entry.in_use), (["m"], 0))
        self.assertIsNone(execution.lease)

    def test_a_cancellation_during_the_release_still_reaches_the_terminal(self):
        pool = Pool()
        entry = pool.add()
        pool.preload()
        pool.release_gate = asyncio.Event()
        execution = execution_for(Fence(pool))

        async def scenario():
            task = asyncio.create_task(prep.make_preparation_stage(lambda: pool)(execution))
            await pool.releasing.wait()
            task.cancel()
            await asyncio.sleep(0)
            pool.release_gate.set()
            try:
                await task
            except BaseException as error:
                return error

        error = run(scenario())
        # Not a CancelledError: the coordinator still writes not_started, then
        # raises the absorbed cancellation.
        self.assertIsInstance(error, coord.EngineStageUnavailable)
        self.assertEqual(len(execution.cancellations), 1)
        self.assertEqual((pool.releases, entry.in_use), (["m"], 0))

    def test_no_pool_means_nothing_dispatched(self):
        with self.assertRaises(coord.EngineStageUnavailable):
            run(prep.make_preparation_stage(lambda: None)(execution_for(None)))


class ReclaimTests(unittest.TestCase):
    def test_the_refusal_carries_only_this_loads_reclaim(self):
        pool = Pool()
        pool.add()
        pool.load_plan = "fail"
        error = refused(self, pool.acquire_owned("m", fence=Fence(pool)))
        self.assertEqual(error.reason, "load_failed")
        self.assertIsNotNone(error.reclaim)
        self.assertIs(error.reclaim, pool._failed_load_reclaim_task)

        async def stale_case():
            stale = asyncio.get_running_loop().create_future()
            stale.set_result(None)
            pool2 = Pool()
            pool2.add()
            pool2._failed_load_reclaim_task = stale
            pool2.load_plan = "fail_without_reclaim"
            try:
                await pool2.acquire_owned("m", fence=Fence(pool2))
            except prep.OwnedPreparationRefused as error2:
                return error2

        error2 = run(stale_case())
        self.assertEqual((error2.reason, error2.reclaim), ("load_failed", None))

    def test_an_aborted_load_is_refused_with_its_reclaim(self):
        pool = Pool()
        pool.add()
        pool.load_plan = "abort"
        error = refused(self, pool.acquire_owned("m", fence=Fence(pool)))
        self.assertEqual(error.reason, "load_aborted")
        self.assertIsNotNone(error.reclaim)

    def test_the_stage_waits_for_the_reclaim_within_one_deadline(self):
        cases = [
            ("finishes", 0.1, None, coord.EngineStageUnavailable),
            ("outlasts the window", 2.0, 0.1, prep.ReclaimUnfinished),
            ("cancelled by pool shutdown", 2.0, None, prep.ReclaimUnfinished),
            ("fails", "raise", None, prep.ReclaimUnfinished),
        ]
        for name, seconds, window, expected in cases:
            with self.subTest(name):
                pool = Pool()
                pool.add()
                pool.load_plan = "fail"
                if seconds == "raise":
                    async def failing():
                        raise RuntimeError("synthetic reclaim failure")
                    pool._reclaim = failing
                else:
                    pool.reclaim_seconds = seconds
                execution = execution_for(Fence(pool))

                async def scenario():
                    stage = asyncio.create_task(prep.make_preparation_stage(lambda: pool)(execution))
                    if name == "cancelled by pool shutdown":
                        await until(lambda: pool._failed_load_reclaim_task is not None)
                        await asyncio.sleep(0.02)
                        pool._failed_load_reclaim_task.cancel()
                    started = time.monotonic()
                    try:
                        await stage
                    except BaseException as error:
                        return error, time.monotonic() - started, pool._failed_load_reclaim_task
                    return None, 0, None

                patch = mock.patch.object(prep, "DRAIN_WINDOW_SECONDS", window) if window else mock.patch.object(prep, "DRAIN_WINDOW_SECONDS", prep.DRAIN_WINDOW_SECONDS)
                with patch:
                    error, elapsed, reclaim = run(scenario())
                self.assertIsInstance(error, expected)
                self.assertIsNotNone(execution.drain_deadline)
                if name == "finishes":
                    self.assertTrue(reclaim.done())
                    self.assertGreaterEqual(elapsed, 0.05)
                if name == "outlasts the window":
                    self.assertLess(elapsed, 1.0)


class DifferentialTests(unittest.TestCase):
    """Ordinary get_engine behaves exactly as at ca93487."""

    SCENARIOS = {
        "load and lease": dict(),
        "ready engine": dict(preload=True),
        "variant reload": dict(preload=True, loaded_variant="profile"),
        "forced LM over a VLM": dict(preload=True, engine_type="vlm", force_lm=True),
        "runtime settings variant": dict(runtime_settings=SimpleNamespace(variant="profile")),
        "pending unload": dict(preload=True, pending=True),
        "unknown model": dict(model_id="nope"),
        "load failure": dict(plan="fail_without_reclaim"),
    }

    def outcome(self, pool_class, spec):
        pool = pool_class()
        engine_type = spec.get("engine_type", "batched")
        entry = pool.add(engine_type=engine_type, model_type="vlm" if engine_type == "vlm" else "llm")
        if spec.get("preload"):
            engine_class = VLMBatchedEngine if engine_type == "vlm" else BatchedEngine
            pool.preload(engine_class=engine_class, variant=spec.get("loaded_variant"))
        if spec.get("pending"):
            entry.pending_unload_reason = "manual unload"
        pool.load_plan = spec.get("plan", "ok")

        async def call():
            try:
                engine = await pool.get_engine(
                    spec.get("model_id", "m"),
                    force_lm=spec.get("force_lm", False),
                    _lease=True,
                    runtime_settings=spec.get("runtime_settings"),
                )
                return ("engine", type(engine).__name__)
            except Exception as error:
                return ("error", type(error).__name__)

        result = run(call())
        return (
            result,
            [(m, f, None if r is None else r.variant) for m, f, r in pool.loads],
            pool.unloads,
            entry.in_use,
            None if entry.engine is None else type(entry.engine).__name__,
            entry.runtime_settings_signature,
        )

    def test_every_scenario_matches_the_baseline(self):
        for name, spec in self.SCENARIOS.items():
            with self.subTest(name):
                self.assertEqual(self.outcome(Pool, spec), self.outcome(BaselinePool, spec))

    def test_the_locked_helper_is_the_baseline_body_plus_the_owned_guard(self):
        old = next(n for n in baseline_class.body if n.name == "get_engine")
        old_lock = next(s for s in old.body if isinstance(s, ast.AsyncWith))
        helper = next(n for n in POOL_CLASS.body if getattr(n, "name", "") == "_get_engine_locked")
        body = [s for s in helper.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
        image_guards = [s for s in body if isinstance(s, ast.If)
                        and ast.unparse(s.test) == "getattr(entry, 'owned_image_active', False)"]
        self.assertEqual(len(image_guards), 1)
        self.assertEqual(ast.unparse(image_guards[0].body[0]),
                         "raise ModelBusyError(model_id, 'start work during owned image execution')")
        body = [s for s in body if s not in image_guards]

        def strip_guard(node):
            for child in ast.walk(node):
                for name in ("body", "orelse"):
                    block = getattr(child, name, None)
                    if isinstance(block, list):
                        block[:] = [
                            s for s in block
                            if not (isinstance(s, ast.If) and ast.unparse(s.test) == "owned")
                        ]
            return node

        stripped = [strip_guard(s) for s in body]
        self.assertEqual(
            [ast.dump(s) for s in stripped], [ast.dump(s) for s in old_lock.body]
        )
        new = next(n for n in POOL_CLASS.body if getattr(n, "name", "") == "get_engine")
        pre_new = [ast.dump(s) for s in new.body if not isinstance(s, ast.AsyncWith)]
        pre_old = [ast.dump(s) for s in old.body if not isinstance(s, ast.AsyncWith)]
        self.assertEqual(pre_new, pre_old)


class CoordinatorIntegrationTests(unittest.TestCase):
    OWNER = "a" * 64
    JOB = "11111111-1111-4111-8111-111111111111"
    OP = "22222222-2222-4222-8222-222222222222"

    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="prism-owned-preparation-")
        self.addCleanup(temp.cleanup)
        self.store = ControlStore(Path(temp.name) / "prism-control")
        self.store.register(
            owner_id=self.OWNER, job_id=self.JOB, operation_id=self.OP,
            request_sha256="c" * 64, model_id="m", settings_sha256="d" * 64,
        )
        self.verifies = []
        real_verify = self.store.verify_claim

        def counting_verify(*args, **kwargs):
            self.verifies.append(None)
            return real_verify(*args, **kwargs)

        self.store.verify_claim = counting_verify
        self.lock = coord.acquire_execution_lock(self.store.path.parent)
        self.addCleanup(os.close, self.lock)
        self.pool = Pool()
        self.entry = self.pool.add()
        self.coordinator = coord.ControlCoordinator(
            self.store, lambda model: "d" * 64,
            prep.make_preparation_stage(lambda: self.pool), lock=self.lock,
        )

    def start(self):
        return self.coordinator.start(
            (self.OWNER, self.JOB, self.OP), request_sha256="c" * 64, model_id="m",
            settings_sha256="d" * 64, deadline=time.monotonic() + 5,
        )

    async def events(self, execution):
        out = []
        while not execution.final:
            out.append(await execution.next_event())
        await asyncio.sleep(0)
        return out

    def test_preparation_then_release_then_not_started(self):
        async def scenario():
            return await self.events(self.start())

        events = run(scenario())
        self.assertEqual([e for e, _ in events], ["claimed", "terminal"])
        terminal = events[-1][1]
        self.assertEqual((terminal.state, terminal.evidence), ("not_started", ("not_admitted",)))
        self.assertEqual((len(self.pool.loads), self.pool.releases, self.entry.in_use), (1, ["m"], 0))
        # The coordinator's check before the stage, then three preparation fences.
        self.assertEqual(len(self.verifies), 4)

    def test_a_cancellation_during_the_load_waits_for_it_then_refuses(self):
        self.pool.load_gate = asyncio.Event()

        async def scenario():
            execution = self.start()
            await self.pool.loading.wait()
            self.store.request_cancel(self.OWNER, self.JOB, self.OP)
            self.coordinator.fence((self.OWNER, self.JOB, self.OP))
            self.pool.load_gate.set()
            return await self.events(execution)

        events = run(scenario())
        terminal = events[-1][1]
        self.assertEqual((terminal.state, terminal.cancel_requested), ("not_started", True))
        self.assertIsNotNone(self.entry.engine)  # the load completed, never cancelled
        self.assertEqual((self.pool.releases, self.entry.in_use), (["m"], 0))

    def test_a_drain_during_the_load_ends_not_started(self):
        self.pool.load_gate = asyncio.Event()

        async def scenario():
            execution = self.start()
            await self.pool.loading.wait()
            drain = asyncio.create_task(self.coordinator.drain(window=5))
            await asyncio.sleep(0)
            self.pool.load_gate.set()
            events = await self.events(execution)
            await asyncio.wait_for(drain, 5)
            return events

        events = run(scenario())
        self.assertEqual(events[-1][1].state, "not_started")
        self.assertEqual(self.entry.in_use, 0)

    def test_the_reclaim_wait_and_the_finish_retries_share_one_window(self):
        self.pool.load_plan = "fail"
        self.pool.reclaim_seconds = 0.45
        attempts = []

        def failing_finish(*args, **kwargs):
            attempts.append(time.monotonic())
            raise ControlStoreError("control_store_unavailable")

        self.store.finish = failing_finish

        async def scenario():
            return await self.events(self.start())

        with mock.patch.object(prep, "DRAIN_WINDOW_SECONDS", 0.6), \
                mock.patch.object(coord, "DRAIN_WINDOW_SECONDS", 0.6):
            events = run(scenario())
        self.assertEqual(events[-1][0], "unresolved")
        self.assertGreater(len(attempts), 0)
        # Every finish attempt falls inside one window from the refusal; a
        # second window after the 0.45 s reclaim would run to about 0.8 s.
        self.assertLessEqual(max(attempts) - self.pool.failed_at, 0.6 + 0.05)

    def test_a_cancelled_coordinator_still_finishes_then_stays_cancelled(self):
        self.pool.load_gate = asyncio.Event()

        async def scenario():
            execution = self.start()
            await self.pool.loading.wait()
            execution.task.cancel()
            await asyncio.sleep(0)
            execution.task.cancel()
            self.pool.load_gate.set()
            events = await self.events(execution)
            try:
                await execution.task
            except asyncio.CancelledError:
                pass
            return events, execution

        events, execution = run(scenario())
        self.assertEqual(events[-1][1].state, "not_started")
        self.assertTrue(execution.task.cancelled())
        self.assertEqual((self.pool.releases, self.entry.in_use), (["m"], 0))


if __name__ == "__main__":
    unittest.main()
