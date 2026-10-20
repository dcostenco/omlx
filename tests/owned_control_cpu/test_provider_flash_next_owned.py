# SPDX-License-Identifier: Apache-2.0
"""Slice 28 CPU proof: owned requests on Flash-Next (qwen4_exp).

What runs for real: the admission allowlist and the pre-load refusal; the
runtime signature with the real Qwen4 PLE offload status and transform (only
the residency estimator and the MoE admission estimate are stubbed); owned
acquisition on cold and reused engines; the loaded-instance check over fake
model trees with the real MoE tree walkers; every scheduler retry site, each
with an owned no-retry request beside an ordinary one; the counted-submission
termination; and source guards over the retry and submission sites. MLX, real
loads and real memory pressure are not observable here (fork unit tests and
the live run cover them).
"""

import ast
import asyncio
from collections import deque
import contextlib
from concurrent.futures import Future, ThreadPoolExecutor
import copy
from dataclasses import dataclass, replace
import json
import logging
from pathlib import Path
import sys
import tempfile
import threading
import time
import types
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import Mock, patch

import test_provider_owned_preparation as prep_fixture
import test_provider_running_cleanup as running_fixture
from omlx import host_submit
from omlx import owned_preparation as prep
from omlx import owned_qwen4
from omlx.model_settings import ModelSettings, ane_prefill_backend, ane_prefill_fraction
from omlx.owned_admission import OwnedAdmission
from omlx.request import RequestOutput, RequestStatus

ROOT = prep_fixture.ROOT
FUTURE = prep_fixture.FUTURE

# ---------------------------------------------------------------------------
# The real runtime signature and Qwen4 PLE offload status.

SIGNATURE_METHODS = {
    "_engine_runtime_signature",
    "_canonical_signature_value",
    "_qwen4_ple_offload_status",
    "_effective_qwen4_model_settings",
    "_deepseek_v41_engram_offload_status",
    "_log_offload_decision_once",
    "_resident_leaves_no_prompt_room",
    "_raise_owned_refusal",
}
signature_methods = [
    n
    for n in prep_fixture.POOL_CLASS.body
    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in SIGNATURE_METHODS
]
assert {m.name for m in signature_methods} == SIGNATURE_METHODS
YARN_TREE = ast.parse((ROOT / "omlx/patches/qwen35_yarn.py").read_text())
yarn_nodes = [
    n
    for n in YARN_TREE.body
    if (isinstance(n, ast.FunctionDef) and n.name == "yarn_factor")
    or (
        isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "YARN_MODEL_TYPES" for t in n.targets)
    )
]
assert len(yarn_nodes) == 2
signature_ns = dict(
    prep_fixture.namespace,
    json=json,
    copy=copy,
    replace=replace,
    Any=Any,
    ane_prefill_backend=ane_prefill_backend,
    ane_prefill_fraction=ane_prefill_fraction,
    bundled_dflash_draft=lambda *args, **kwargs: None,
    logger=logging.getLogger("flash-next-owned-test"),
    # The status methods import their estimators relative to the package.
    __name__="omlx.engine_pool_slice28",
    __package__="omlx",
)
exec(
    compile(
        ast.fix_missing_locations(
            ast.Module(
                body=[*yarn_nodes, prep_fixture.class_node("SignaturePool", signature_methods)],
                type_ignores=[],
            )
        ),
        str(ROOT / "omlx/engine_pool.py"),
        "exec",
        flags=FUTURE,
        dont_inherit=True,
    ),
    signature_ns,
)


# Forced-offload warnings are expected in these cases.
logging.getLogger("flash-next-owned-test").setLevel(logging.ERROR)


class SignatureStubs:
    """Pool state the real signature methods read."""

    def __init__(self, ceiling=100):
        self._entries = {}
        self.settings = {}
        self._settings_manager = SimpleNamespace(
            get_settings=lambda model_id: self.settings.get(model_id)
        )
        self._offload_warn_state = {}
        self.ceiling = ceiling

    def _residency_ceiling(self):
        return self.ceiling

    def _fallback_admission_ceiling(self):
        return 0

    def _current_ceiling(self):
        return 0

    def _entry_is_diffusion_model(self, entry):
        return False

    def _distributed_deployment_for_entry(self, entry):
        return None


class SignaturePool(signature_ns["SignaturePool"], SignatureStubs):
    pass


@dataclass(frozen=True)
class Estimate:
    """A Qwen4-Exp residency estimate: forced to SSD below ``force_below``."""

    checkpoint_bytes: int = 100
    resident_bytes: int = 90
    mmap_bytes: int = 30
    supported: bool = True
    force_below: int = 0

    def force_ssd_offload(self, ceiling):
        return ceiling < self.force_below


@contextlib.contextmanager
def estimator(estimate=None, raises=None):
    """Stub the two lazily imported estimators for one test only."""
    residency = types.ModuleType("omlx.patches.mlx_vlm_qwen4_exp_compat.residency")

    def qwen4_exp_residency_estimate(path):
        if raises is not None:
            raise raises
        return estimate or Estimate()

    residency.qwen4_exp_residency_estimate = qwen4_exp_residency_estimate
    compat = types.ModuleType("omlx.patches.mlx_vlm_qwen4_exp_compat")
    compat.__path__ = []
    compat.residency = residency
    moe = types.ModuleType("omlx.patches.moe_expert_offload")
    moe.estimate_offload_admission_bytes = (
        lambda path, checkpoint, fraction, mtp_resident=False: checkpoint // 2
    )
    with patch.dict(
        sys.modules, {compat.__name__: compat, residency.__name__: residency, moe.__name__: moe}
    ):
        yield


def trial_settings(**overrides):
    """The trial's persisted Flash-Next settings, with gather-ahead off."""
    settings = ModelSettings(
        qwen4_ple_ssd_offload=True,
        qwen4_ple_gather_ahead=False,
        moe_expert_offload_enabled=True,
        moe_expert_offload_resident_fraction=0.03,
        max_context_window=65536,
    )
    for key, value in overrides.items():
        setattr(settings, key, value)
    return settings


def flash_entry(model_id="flash", family="qwen4_exp"):
    entry = prep_fixture.EngineEntry(
        model_id=model_id,
        model_path=f"/models/{model_id}",
        model_type="vlm",
        engine_type="vlm",
        estimated_size=1,
    )
    entry.config_model_type = family
    entry.model_context_length = 262144
    return entry


def signature_pool(settings=None, family="qwen4_exp", ceiling=100):
    pool = SignaturePool(ceiling)
    entry = flash_entry(family=family)
    pool._entries[entry.model_id] = entry
    pool.settings[entry.model_id] = settings if settings is not None else trial_settings()
    return pool, entry


# The exact signature the trial's Flash-Next settings record (canonical repr).
TRIAL_SIGNATURE = (
    ("trust_remote_code", "False"),
    ("external_engine", "None"),
    ("index_cache_freq", "None"),
    ("embedding_audio_enabled", "False"),
    ("mtp_enabled", "False"),
    ("qwen4_ple_ssd_offload", "True"),
    ("qwen4_ple_gather_ahead", "False"),
    ("qwen4_gdn_decode_wide_proj", "False"),
    ("deepseek_v41_engram_ssd_offload", "False"),
    ("deepseek_v41_ced_prefill_enabled", "False"),
    ("turboquant_kv_enabled", "False"),
    ("qwen35_oq_a8_enabled", "False"),
    ("qwen35_ane_prefill_enabled", "False"),
    ("moe_expert_offload_enabled", "True"),
    ("moe_expert_offload_resident_fraction", "0.03"),
    ("specprefill_enabled", "False"),
    ("dflash_enabled", "False"),
    ("vlm_mtp_enabled", "False"),
)


def with_value(signature, key, value):
    return tuple((k, value if k == key else v) for k, v in signature)


def entry_refusal(entry, settings, signature=None, submit_failures=0):
    calls = []

    def lazy():
        calls.append(1)
        return signature

    reason = prep.owned_entry_refusal(
        entry,
        settings,
        distributed=False,
        diffusion=False,
        signature=lazy,
        submit_failures=submit_failures,
    )
    return reason, len(calls)


class AllowlistTests(unittest.TestCase):
    def test_the_trial_settings_record_exactly_the_reviewed_signature(self):
        with estimator():
            pool, entry = signature_pool()
            signature = pool._engine_runtime_signature("flash", None)
            self.assertEqual(signature, TRIAL_SIGNATURE)
            self.assertIsNone(prep.owned_signature_refusal("qwen4_exp", signature))
            self.assertEqual(entry_refusal(entry, trial_settings(), signature), (None, 1))
            self.assertIs(pool._raise_owned_refusal("flash"), entry)

    def test_each_reviewed_key_with_another_value_is_refused(self):
        other = {"True": "False", "False": "True", "None": "'splash'"}
        for key, value in TRIAL_SIGNATURE:
            values = (
                ["0.0", "1.5", "-0.03", "'x'", "None"]
                if key == "moe_expert_offload_resident_fraction"
                else [other[value]]
            )
            for bad in values:
                with self.subTest(key=key, value=bad):
                    self.assertEqual(
                        prep.owned_signature_refusal("qwen4_exp", with_value(TRIAL_SIGNATURE, key, bad)),
                        "variant_unreviewed",
                    )
        for fraction in ("1.0", "0.25", "0.001"):
            signature = with_value(TRIAL_SIGNATURE, "moe_expert_offload_resident_fraction", fraction)
            self.assertIsNone(prep.owned_signature_refusal("qwen4_exp", signature))

    def test_a_missing_duplicate_or_unlisted_key_is_refused(self):
        cases = {f"missing {k}": tuple(kv for kv in TRIAL_SIGNATURE if kv[0] != k) for k, _ in TRIAL_SIGNATURE}
        cases["duplicate"] = TRIAL_SIGNATURE + (TRIAL_SIGNATURE[0],)
        cases["yarn_factor"] = TRIAL_SIGNATURE + (("yarn_factor", "8.0"),)
        cases["a key added later"] = TRIAL_SIGNATURE + (("future_variant", "False"),)
        cases["none"] = None
        for name, signature in cases.items():
            with self.subTest(name):
                expected = "settings_unavailable" if signature is None else "variant_unreviewed"
                self.assertEqual(prep.owned_signature_refusal("qwen4_exp", signature), expected)

    def test_turning_a_variant_on_in_the_settings_is_refused_through_the_real_signature(self):
        cases = {
            "turboquant": dict(turboquant_kv_enabled=True),
            "oq_a8": dict(qwen35_oq_a8_enabled=True),
            "gather_ahead": dict(qwen4_ple_gather_ahead=True),
            "gdn_wide_proj": dict(qwen4_gdn_decode_wide_proj=True),
            "index_cache": dict(index_cache_freq=4),
            "yarn": dict(max_context_window=524288),
            "audio": dict(embedding_audio_enabled=True),
            "trust_remote_code": dict(trust_remote_code=True),
            "ple_resident": dict(qwen4_ple_ssd_offload=False),
            "moe_resident": dict(moe_expert_offload_enabled=False),
        }
        for name, overrides in cases.items():
            with self.subTest(name), estimator():
                pool, entry = signature_pool(trial_settings(**overrides))
                signature = pool._engine_runtime_signature("flash", None)
                self.assertNotEqual(signature, TRIAL_SIGNATURE)
                self.assertEqual(prep.owned_signature_refusal("qwen4_exp", signature), "variant_unreviewed")
                with self.assertRaises(prep.OwnedPreparationRefused) as caught:
                    pool._raise_owned_refusal("flash")
                self.assertEqual(caught.exception.reason, "variant_unreviewed")

    def test_oq_a8_is_refused_before_any_load_whatever_its_patch_would_do(self):
        # Its patch can return False or raise while the signature records it on;
        # owned work refuses before the load, so the patch never runs.
        for behaviour in ("returns_false", "raises"):
            with self.subTest(behaviour), estimator():
                patch_module = types.ModuleType("omlx.patches.qwen35_oq_a8")
                calls = []

                def apply(*args, **kwargs):
                    calls.append(args)
                    if behaviour == "raises":
                        raise RuntimeError("synthetic A8 install failure")
                    return False

                patch_module.apply_qwen35_oq_a8_patch = apply
                with patch.dict(sys.modules, {patch_module.__name__: patch_module}):
                    pool, entry = signature_pool(trial_settings(qwen35_oq_a8_enabled=True))
                    with self.assertRaises(prep.OwnedPreparationRefused) as caught:
                        pool._raise_owned_refusal("flash")
                self.assertEqual(caught.exception.reason, "variant_unreviewed")
                self.assertEqual(calls, [])

    def test_retained_refusals_keep_their_reason_on_pinned_qwen4(self):
        for key, reason in prep.OWNED_SETTINGS_REFUSALS:
            with self.subTest(key):
                value = "splash" if key == "external_engine" else True
                refusal, signature_calls = entry_refusal(
                    flash_entry(), trial_settings(**{key: value}), TRIAL_SIGNATURE
                )
                self.assertEqual((refusal, signature_calls), (reason, 0))

    def test_deepseek_stays_refused_with_and_without_its_flags(self):
        for flags in ({}, dict(deepseek_v41_engram_ssd_offload=True, deepseek_v41_ced_prefill_enabled=True)):
            with self.subTest(flags):
                refusal = entry_refusal(flash_entry(family="deepseek_v41"), trial_settings(**flags), TRIAL_SIGNATURE)
                self.assertEqual(refusal, ("pressure_override", 0))
        self.assertNotIn("deepseek_v41", prep.OWNED_PRESSURE_SIGNATURES)

    def test_an_unmapped_pressure_family_is_refused(self):
        families = prep.PRESSURE_OVERRIDE_FAMILIES | {"qwen9_exp"}
        with patch.object(prep, "PRESSURE_OVERRIDE_FAMILIES", families):
            self.assertEqual(
                entry_refusal(flash_entry(family="qwen9-exp"), trial_settings(), TRIAL_SIGNATURE),
                ("pressure_override", 0),
            )

    def test_reviewed_families_are_pressure_families(self):
        self.assertLessEqual(set(prep.OWNED_PRESSURE_SIGNATURES), set(prep.PRESSURE_OVERRIDE_FAMILIES))
        self.assertEqual(set(prep.OWNED_PRESSURE_SIGNATURES), {"qwen4_exp"})
        self.assertEqual(
            set(prep.OWNED_PRESSURE_SIGNATURES["qwen4_exp"]), {key for key, _ in TRIAL_SIGNATURE}
        )

    def test_a_failed_host_submission_refuses_every_family_first(self):
        llm = flash_entry(family="llama")
        for entry in (flash_entry(), llm):
            with self.subTest(entry.config_model_type):
                self.assertEqual(
                    entry_refusal(entry, trial_settings(), TRIAL_SIGNATURE, submit_failures=1),
                    ("host_read_ambiguous", 0),
                )
        with estimator(), patch.dict(signature_ns, submission_failures=lambda: 2):
            pool, _ = signature_pool()
            with self.assertRaises(prep.OwnedPreparationRefused) as caught:
                pool._raise_owned_refusal("flash")
        self.assertEqual(caught.exception.reason, "host_read_ambiguous")


class RealSignatureTests(unittest.TestCase):
    """The real transforms: pinned offload is recorded true unless estimation fails."""

    def offload(self, signature):
        return dict(signature)["qwen4_ple_ssd_offload"]

    def test_offload_is_recorded_true_whether_or_not_pressure_forces_it(self):
        for forced in (False, True):
            with self.subTest(forced=forced), estimator(Estimate(force_below=10**9 if forced else 0)):
                pool, entry = signature_pool()
                settings = pool.settings["flash"]
                enabled, was_forced, _ = pool._qwen4_ple_offload_status(entry, settings)
                self.assertEqual((enabled, was_forced), (True, forced))
                effective = pool._effective_qwen4_model_settings(entry, settings)
                if forced:
                    self.assertIsNot(effective, settings)  # an equivalent copy
                else:
                    self.assertIs(effective, settings)
                self.assertEqual(effective.to_dict(), settings.to_dict())
                self.assertEqual(pool._engine_runtime_signature("flash", None), TRIAL_SIGNATURE)
                self.assertEqual(pool._engine_runtime_signature("flash", effective), TRIAL_SIGNATURE)

    def test_a_ceiling_change_between_admission_and_load_keeps_the_signature(self):
        with estimator(Estimate(force_below=50)):
            pool, entry = signature_pool(ceiling=100)
            settings = pool.settings["flash"]
            admitted = pool._effective_qwen4_model_settings(entry, settings, ceiling=10)  # forced
            before = pool._engine_runtime_signature("flash", admitted)
            pool.ceiling = 10  # the load sees a lower ceiling: forced now
            loaded = pool._effective_qwen4_model_settings(entry, admitted)
            after = pool._engine_runtime_signature("flash", loaded)
            pool.ceiling = 1000  # and a higher one: not forced
            relaxed = pool._engine_runtime_signature("flash", loaded)
        self.assertEqual((before, after, relaxed), (TRIAL_SIGNATURE,) * 3)

    def test_an_estimator_failure_records_offload_false_and_is_refused(self):
        for error in (OSError("unreadable"), TypeError("shape"), ValueError("config")):
            with self.subTest(type(error).__name__), estimator(raises=error):
                pool, entry = signature_pool()
                signature = pool._engine_runtime_signature("flash", None)
                self.assertEqual(self.offload(signature), "False")
                self.assertEqual(signature, with_value(TRIAL_SIGNATURE, "qwen4_ple_ssd_offload", "False"))
                with self.assertRaises(prep.OwnedPreparationRefused) as caught:
                    pool._raise_owned_refusal("flash")
                self.assertEqual(caught.exception.reason, "variant_unreviewed")


# ---------------------------------------------------------------------------
# Owned acquisition on cold and reused engines.


class FlashPool(prep_fixture.Pool):
    """The real acquisition; the stub signature is the settings' ``sig``."""

    during_load = None

    def _engine_runtime_signature(self, model_id, runtime_settings=None):
        settings = runtime_settings if runtime_settings is not None else self.settings.get(model_id)
        return None if settings is None else tuple(settings.sig)

    async def _load_engine(self, model_id, force_lm=False, runtime_settings=None):
        await prep_fixture.Pool._load_engine(self, model_id, force_lm, runtime_settings)
        if self.during_load is not None:
            self.during_load(self)


def flash_pool(family="qwen4_exp", sig=TRIAL_SIGNATURE):
    pool = FlashPool()
    entry = pool.add("flash", engine_type="vlm", model_type="vlm", sig=sig)
    entry.config_model_type = family
    return pool, entry


def preload(pool, entry, recorded=TRIAL_SIGNATURE, instance_refusal=None):
    entry.engine = prep_fixture.VLMBatchedEngine("flash")
    entry.engine.instance_refusal = instance_refusal
    entry.runtime_settings_signature = recorded
    return entry.engine


class AcquisitionTests(unittest.TestCase):
    def acquire(self, pool):
        return prep_fixture.run(pool.acquire_owned("flash", fence=prep_fixture.Fence(pool)))

    def test_a_cold_load_of_the_reviewed_variant_is_leased(self):
        pool, entry = flash_pool()
        lease = self.acquire(pool)
        self.assertIs(lease.engine, entry.engine)
        self.assertEqual((len(pool.loads), entry.in_use, entry.runtime_settings_signature), (1, 1, TRIAL_SIGNATURE))

    def test_a_reused_engine_of_the_reviewed_variant_is_leased(self):
        pool, entry = flash_pool()
        engine = preload(pool, entry)
        lease = self.acquire(pool)
        self.assertIs(lease.engine, engine)
        self.assertEqual((pool.loads, entry.in_use), ([], 1))

    def test_a_loaded_engine_of_another_variant_is_refused_and_released(self):
        pool, entry = flash_pool()
        preload(pool, entry, recorded=with_value(TRIAL_SIGNATURE, "qwen4_ple_gather_ahead", "True"))
        error = prep_fixture.refused(self, pool.acquire_owned("flash", fence=prep_fixture.Fence(pool)))
        # Owned work never unloads another variant to load its own.
        self.assertEqual(error.reason, "variant_loaded")
        self.assertEqual((pool.loads, pool.unloads, entry.in_use), ([], [], 0))

    def test_a_settings_change_during_the_load_fails_the_identity_check(self):
        pool, entry = flash_pool()
        changed = with_value(TRIAL_SIGNATURE, "turboquant_kv_enabled", "True")
        pool.during_load = lambda p: setattr(p.settings["flash"], "sig", changed)
        error = prep_fixture.refused(self, pool.acquire_owned("flash", fence=prep_fixture.Fence(pool)))
        self.assertEqual(error.reason, "fallback_engine")
        self.assertEqual(entry.in_use, 0)

    def test_an_unreviewed_variant_built_by_the_load_is_refused_from_the_recorded_signature(self):
        # The load itself built (and recorded) another variant the settings now
        # also describe: identity holds, the recorded-signature check refuses.
        pool, entry = flash_pool()
        built = with_value(TRIAL_SIGNATURE, "qwen4_ple_ssd_offload", "False")

        def rebuild(p):
            p.settings["flash"].sig = built
            p._entries["flash"].runtime_settings_signature = built

        pool.during_load = rebuild
        error = prep_fixture.refused(self, pool.acquire_owned("flash", fence=prep_fixture.Fence(pool)))
        self.assertEqual(error.reason, "variant_unreviewed")
        self.assertEqual((entry.in_use, pool.releases), (0, ["flash"]))

    def test_a_loaded_instance_mismatch_is_refused_and_released(self):
        for loaded in (False, True):
            with self.subTest(loaded=loaded):
                pool, entry = flash_pool()
                if loaded:
                    preload(pool, entry, instance_refusal="ple_resident")
                else:
                    pool.during_load = lambda p: setattr(p._entries["flash"].engine, "instance_refusal", "moe_capacity")
                error = prep_fixture.refused(self, pool.acquire_owned("flash", fence=prep_fixture.Fence(pool)))
                self.assertEqual(error.reason, "model_entangled")
                self.assertEqual((entry.in_use, pool.releases), (0, ["flash"]))

    def test_other_families_never_run_the_instance_check(self):
        pool, entry = flash_pool(family="llama", sig=(("variant", "v1"),))
        preload(pool, entry, recorded=(("variant", "v1"),), instance_refusal="would_refuse")
        self.assertIs(self.acquire(pool).engine, entry.engine)


# ---------------------------------------------------------------------------
# The loaded-instance check, with the real MoE tree walkers.

MOE_TREE = ast.parse((ROOT / "omlx/patches/moe_expert_offload.py").read_text())
MOE_NAMES = {"_iter_switch_glus", "_is_mtp_path", "_is_stock_switch_glu", "moe_offload_caches", "_minimum_experts"}
moe_nodes = [n for n in MOE_TREE.body if isinstance(n, ast.FunctionDef) and n.name in MOE_NAMES]
assert {n.name for n in moe_nodes} == MOE_NAMES
moe_ns = dict(Path=Path, json=json)
exec(
    compile(ast.Module(body=moe_nodes, type_ignores=[]), str(ROOT / "omlx/patches/moe_expert_offload.py"), "exec"),
    moe_ns,
)


class Node(dict):
    """mlx.nn.Module's shape: children are dict items, also read as attributes."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


class SwitchGLU(Node):
    """A stock (unwrapped) SwitchGLU, matched by class name and activation."""

    activation = staticmethod(lambda x: x)


class OffloadSwitchGLU(Node):
    pass


class DiskTable:
    def __init__(self, root, gather_ahead=False, submissions=0, files=("model-00001.safetensors",)):
        self.gather_ahead = gather_ahead
        self.prefetch_submissions = submissions
        self._readers = {name: SimpleNamespace(path=Path(root) / name) for name in files}


class ResidentTable:
    gather_ahead = True
    prefetch_submissions = 0
    _readers = {}


def expert_cache(n=256, full=8, capacity=None):
    return SimpleNamespace(moe_offload_cache=True, n_experts=n, full_capacity=full, capacity=full if capacity is None else capacity)


def model_tree(root, layers=3, table=None, cache=None, mlp=None, mtp_switch=False):
    inner = Node(layers=[])
    for index in range(layers):
        glu = OffloadSwitchGLU()
        glu.cache = cache(index) if cache else expert_cache()
        layer = Node(
            mlp=(mlp(index) if mlp else glu),
            ple=Node(ple_embedding=Node(ngram_embedding=table(index) if table else DiskTable(root))),
        )
        inner["layers"].append(layer)
    language = Node(model=inner)
    if mtp_switch:
        language["mtp"] = Node(layers=[Node(mlp=SwitchGLU())])
    return Node(language_model=language)


class InstanceCheckTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        (self.root / "config.json").write_text(json.dumps({"text_config": {"num_experts_per_tok": 8}}))

    def tearDown(self):
        self.tmp.cleanup()

    def check(self, tree, fraction=0.03, wrapped=None, table_type=DiskTable):
        caches = moe_ns["moe_offload_caches"](tree)
        return owned_qwen4.qwen4_instance_refusal(
            tree,
            model_dir=self.root,
            fraction=fraction,
            wrapped=len(caches) if wrapped is None else wrapped,
            disk_table_type=table_type,
            iter_switch_glus=moe_ns["_iter_switch_glus"],
            is_mtp_path=moe_ns["_is_mtp_path"],
            offload_caches=moe_ns["moe_offload_caches"],
            minimum_experts=moe_ns["_minimum_experts"],
        )

    def test_the_built_flash_next_shape_is_admitted(self):
        self.assertIsNone(self.check(model_tree(self.root)))
        # The MTP head's experts stay stock; they are never wrapped.
        self.assertIsNone(self.check(model_tree(self.root, mtp_switch=True)))
        # Slots a prefill shrank to the floor still pass: full_capacity is compared.
        self.assertIsNone(self.check(model_tree(self.root, cache=lambda i: expert_cache(capacity=1))))

    def test_each_mismatch_is_refused(self):
        other = self.root.parent / f"{self.root.name}-other-model"  # a sibling, not inside
        cases = {
            "ple_resident": dict(tree=model_tree(self.root, table=lambda i: ResidentTable() if i == 1 else DiskTable(self.root))),
            "ple_gather_ahead": dict(tree=model_tree(self.root, table=lambda i: DiskTable(self.root, gather_ahead=True))),
            "ple_gather_ahead ": dict(tree=model_tree(self.root, table=lambda i: DiskTable(self.root, submissions=1 if i == 2 else 0))),
            "ple_foreign_path": dict(tree=model_tree(self.root, table=lambda i: DiskTable(other))),
            "ple_foreign_path ": dict(tree=model_tree(self.root, table=lambda i: DiskTable(self.root, files=()))),
            "ple_unobserved": dict(tree=Node(language_model=Node(model=Node(layers=[])))),
            "moe_unwrapped": dict(tree=model_tree(self.root, mlp=lambda i: SwitchGLU() if i == 0 else None)),
            "moe_coverage": dict(tree=model_tree(self.root), wrapped=4),
            "moe_capacity": dict(tree=model_tree(self.root, cache=lambda i: expert_cache(full=9 if i == 1 else 8))),
            "moe_capacity ": dict(tree=model_tree(self.root), fraction=0.25),
        }
        for reason, case in cases.items():
            with self.subTest(reason):
                tree = case.pop("tree")
                if reason.strip() == "moe_unwrapped":
                    # Layer 0 holds a stock SwitchGLU; the others keep their offload wrapper.
                    for index, layer in enumerate(tree.language_model.model.layers):
                        if layer["mlp"] is None:
                            glu = OffloadSwitchGLU()
                            glu.cache = expert_cache()
                            layer["mlp"] = glu
                self.assertEqual(self.check(tree, **case), reason.strip())

    def test_shards_in_nested_folders_are_inside_the_model(self):
        nested = model_tree(self.root, table=lambda i: DiskTable(self.root, files=("ple/model-00001.safetensors",)))
        self.assertIsNone(self.check(nested))
        escaping = model_tree(self.root, table=lambda i: DiskTable(self.root, files=("../elsewhere/model.safetensors",)))
        self.assertEqual(self.check(escaping), "ple_foreign_path")
        itself = model_tree(self.root, table=lambda i: DiskTable(self.root, files=(".",)))
        self.assertEqual(self.check(itself), "ple_foreign_path")

    def test_no_moe_offload_at_all_is_refused(self):
        tree = model_tree(self.root)
        for layer in tree.language_model.model.layers:
            layer["mlp"] = Node()  # dense, nothing wrapped
        self.assertEqual(self.check(tree, wrapped=0), "moe_coverage")

    def test_the_check_never_reads_the_ple_construction_globals(self):
        names = {
            n.id if isinstance(n, ast.Name) else n.attr
            for n in ast.walk(ast.parse(Path(owned_qwen4.__file__).read_text()))
            if isinstance(n, (ast.Name, ast.Attribute))
        }
        self.assertFalse({name for name in names if name.startswith("_PLE_RUNTIME")})
        self.assertNotIn("get_ple_runtime_mode", names)

    def test_an_engine_without_a_vlm_model_is_unavailable(self):
        self.assertEqual(owned_qwen4.qwen4_instance_refusal_for_engine(SimpleNamespace()), "engine_unavailable")


# ---------------------------------------------------------------------------
# Every retry site, real, with an owned no-retry request beside others.

RETRY_METHODS = {
    "_drop_from_prefill_queues",
    "_reschedule_generation_overflow_requests",
    "_requeue_or_fail_prefill",
    "_fail_packed_prefill",
    "_process_pending_memory_requeue",
    "_advance_chunked_prefills",
    "_publish_owned_no_retry_outputs",
    "_pause_for_prefill_eviction",
    "_teardown_prefill_request",
    "_end_owned_without_retry",
    "_drop_boundary_snapshots_for_request",
    "_probe_owned_presence",
    "_retire_owned_leftovers",
}
RETRY_CONSTANTS = {
    "_MAX_GENERATION_OVERFLOW_RETRIES",
    "_MAX_PREFILL_OOM_RETRIES",
    "_MEMORY_REQUEUE_PENDING_WRITES",
}
SCHEDULER_CLASS = running_fixture.cls
retry_methods = [n for n in SCHEDULER_CLASS.body if isinstance(n, ast.FunctionDef) and n.name in RETRY_METHODS]
assert {m.name for m in retry_methods} == RETRY_METHODS
retry_constants = [
    n
    for n in SCHEDULER_CLASS.body
    if isinstance(n, (ast.Assign, ast.AnnAssign))
    and any(
        isinstance(t, ast.Name) and t.id in RETRY_CONSTANTS
        for t in (n.targets if isinstance(n, ast.Assign) else [n.target])
    )
]
assert len(retry_constants) == len(RETRY_CONSTANTS)


class EvictionNeeded(Exception):
    def __init__(self, request):
        super().__init__("synthetic eviction")
        self.request = request


class PrefillMemoryExceeded(Exception):
    """The typed capacity refusal; the real output builder reads these fields."""

    estimated_bytes = 2 * 1024**3
    limit_bytes = 1024**3


class PrefillAborted(Exception):
    pass


class PackedPrefillError(Exception):
    pass


SCHEDULER_STUB = SimpleNamespace(
    _owned_no_retry=running_fixture.RunningScheduler._owned_no_retry,
    _clear_cache=lambda scheduler: None,
    _packed_prefill_ready=lambda scheduler: False,
    _packing_fits_contended_cap=lambda scheduler: False,
    _contended_prefill_cap=lambda scheduler: 0,
)
EXCEPTIONS = dict(
    _PrefillEvictionNeeded=EvictionNeeded,
    PrefillMemoryExceededError=PrefillMemoryExceeded,
    _PrefillAbortedError=PrefillAborted,
    _PackedPrefillError=PackedPrefillError,
)
retry_ns = dict(
    running_fixture.namespace,
    **EXCEPTIONS,
    Scheduler=SCHEDULER_STUB,
    _cache_base_sizes=lambda cache: 0,
    _prefill_memory_exception_output=lambda rid, error: RequestOutput(
        request_id=rid, finished=True, finish_reason="error", error=str(error)
    ),
    _mtp_priming=SimpleNamespace(_eligible_host=lambda model: None, activate_request=lambda model, rid: None),
    logger=logging.getLogger("flash-next-owned-test"),
)
exec(
    compile(
        ast.fix_missing_locations(
            ast.Module(
                body=[prep_fixture.class_node("RetryMethods", [*retry_constants, *retry_methods])],
                type_ignores=[],
            )
        ),
        str(running_fixture.SOURCE),
        "exec",
        flags=FUTURE,
        dont_inherit=True,
    ),
    retry_ns,
)


class RetryScheduler(retry_ns["RetryMethods"], running_fixture.RunningScheduler):
    pass


def bind_owned_retirement(scheduler):
    """The stage fixture's scheduler gains slice 28's real retirement method."""
    scheduler._retire_owned_leftovers = types.MethodType(RetryScheduler._retire_owned_leftovers, scheduler)
    scheduler._owned_retirement_failures = set()


def step_output():
    return SimpleNamespace(outputs=[], finished_request_ids=set(), has_work=False)


class RetrySiteTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.base = running_fixture.RunningCleanupTests()
        await self.base.asyncSetUp()
        self.core = self.base.core
        s = RetryScheduler()
        s.__dict__.update(self.base.s.__dict__)
        self.core.scheduler = self.s = s
        s._owned_no_retry_outputs = []
        s._owned_retirement_failures = set()
        s._memory_requeue_ids = set()
        s._packed_prefill_stats = {"failures": 0}
        s._packed_prefill_disabled = None
        s._reclaim_prefill_headroom = Mock()
        s.config = SimpleNamespace(max_num_seqs=1, model_name="flash")
        # The real snapshot release, not the running fixture's stub.
        s.__dict__.pop("_drop_boundary_snapshots_for_request", None)
        # Row registrations the abort cleanup must retire.
        self.unregistered = []
        self.rows = patch.dict(
            running_fixture.namespace,
            _unregister_uid_row=lambda model, uid: self.unregistered.append(uid),
        )
        self.rows.start()

    async def asyncTearDown(self):
        self.rows.stop()
        await self.base.asyncTearDown()

    async def owned(self, rid, retry=False):
        handle = OwnedAdmission(self.core, rid, time.monotonic() + 3, retry=retry)
        await self.core.add_request("synthetic", request_id=rid, _owned_admission=handle)
        return handle.request

    async def ordinary(self, rid):
        await self.core.add_request("synthetic", request_id=rid)
        return self.s.requests[rid]

    async def three(self):
        """An owned no-retry request, an ordinary one, and owned work on another family."""
        return (
            await self.owned("synthetic-flash"),
            await self.ordinary("synthetic-plain"),
            await self.owned("synthetic-other-family", retry=True),
        )

    def holdings(self, request):
        """What a live request holds: prefix blocks, a boundary snapshot, a cache."""
        self.base.fetch(request)
        self.s._boundary_cache_snapshots[request.request_id] = object()
        request.prompt_cache = [object()]

    def decoding(self, request, uid):
        s = self.s
        s.waiting.remove(request)
        request.status = RequestStatus.RUNNING
        request.batch_uid = uid
        s.running[request.request_id] = request
        s.request_id_to_uid[request.request_id] = uid
        s.uid_to_request_id[uid] = request.request_id
        self.holdings(request)

    def prefilling(self, request):
        s = self.s
        s.waiting.remove(request)
        state = running_fixture.PrefillState(
            request=request, cache=[object()], tokens_remaining=None, last_token=[0],
            tokens_processed=4, base_size=0, emitted_boundaries={}, boundary_enabled=False,
            block_size=4, total_length=12,
        )
        s._prefill_states[request.request_id] = state
        s.prefilling.append(request)
        self.holdings(request)
        return state

    def assert_ended(self, request, prefilling_rebuilt_by_caller=False):
        s, rid = self.s, request.request_id
        self.assertNotIn(rid, s.requests)
        self.assertNotIn(rid, s.running)
        # The chunked loop rebuilds prefilling from the states left.
        self.assertNotIn(rid, s._prefill_states)
        self.assertFalse(any(r is request for r in s.waiting))
        if not prefilling_rebuilt_by_caller:
            self.assertFalse(any(r is request for r in s.prefilling))

    def assert_retired(self, request, uid=None):
        """Ended through the abort cleanup: nothing of the request is left."""
        s, rid = self.s, request.request_id
        self.assert_ended(request)
        self.assertNotIn(rid, s.request_id_to_uid)
        if uid is not None:
            self.assertNotIn(uid, s.uid_to_request_id)
            self.assertIn(uid, self.unregistered)
        self.assertNotIn(rid, s._boundary_cache_snapshots)
        s._boundary_snapshot_store.cleanup_request.assert_any_call(rid)
        self.assertIsNone(request.prompt_cache)
        # The abort releases the block refs and clears the block-aware entry
        # (which drops the table in production; the fixture's cache records it).
        self.assertIn(rid, s.block_aware_cache.cleared)
        self.assertEqual(request.status, RequestStatus.FINISHED_ABORTED)
        self.assertIn(rid, tracker_removed())

    def assert_retried(self, *requests):
        for request in requests:
            self.assertTrue(any(r is request for r in self.s.waiting), request.request_id)

    def assert_error(self, outputs, request):
        ended = [o for o in outputs if o.request_id == request.request_id]
        self.assertEqual(len(ended), 1)
        self.assertTrue(ended[0].finished)
        self.assertEqual(ended[0].finish_reason, "error")

    def published(self, request):
        output = step_output()
        self.s._publish_owned_no_retry_outputs(output)
        self.assert_error(output.outputs, request)
        self.assertIn(request.request_id, output.finished_request_ids)
        self.assertTrue(output.has_work)
        self.assertEqual(self.s._owned_no_retry_outputs, [])

    async def test_the_running_reschedule_retires_owned_work_and_publishes_its_error_next_step(self):
        flash, plain, other = await self.three()
        for uid, request in enumerate((flash, plain, other), start=3):
            self.decoding(request, uid)
        before = self.base.refs()
        self.s.batch_generator = None  # what every caller does first
        self.assertEqual(self.s._reschedule_running_requests(), [])
        self.assert_retired(flash, uid=3)
        self.assert_retried(plain, other)
        # The ended request's uid is free again: a new owned request can take it.
        self.assertNotIn(3, self.s.uid_to_request_id)
        self.assertEqual(self.s.request_id_to_uid["synthetic-plain"], 4)
        self.assertEqual(self.base.refs(), [r - 1 for r in before])  # only its refs went
        self.published(flash)
        quiet = step_output()
        self.s._publish_owned_no_retry_outputs(quiet)
        self.assertEqual((quiet.outputs, quiet.has_work), ([], False))

    async def test_the_corruption_reschedule_fails_owned_work_through_its_failed_ids(self):
        flash, plain, other = await self.three()
        for uid, request in enumerate((flash, plain, other), start=3):
            self.decoding(request, uid)
        # _recover_from_cache_error ran first: maps and generator are gone.
        self.s.batch_generator = None
        self.s.request_id_to_uid.clear()
        self.s.uid_to_request_id.clear()
        self.s._collect_corruption_retry_requests = lambda: list(self.s.running.values())
        self.s._drop_boundary_snapshots_for_request = Mock(
            side_effect=lambda rid: self.s._boundary_cache_snapshots.pop(rid, None)
        )
        self.assertEqual(self.s._reschedule_running_requests(is_corruption=True), ["synthetic-flash"])
        self.assert_ended(flash)
        self.assertNotIn("synthetic-flash", self.s._boundary_cache_snapshots)
        self.s._release_paged_cache_for_request.assert_any_call("synthetic-flash")
        self.assert_retried(plain, other)
        self.assertEqual(self.s._owned_no_retry_outputs, [])  # the caller publishes failed_ids

    async def test_the_generation_overflow_reschedule_fails_owned_work(self):
        flash, plain, other = await self.three()
        for uid, request in enumerate((flash, plain, other), start=3):
            self.decoding(request, uid)
        self.assertEqual(self.s._reschedule_generation_overflow_requests(), ["synthetic-flash"])
        self.assert_ended(flash)
        self.assertNotIn("synthetic-flash", self.s._boundary_cache_snapshots)
        self.s._release_paged_cache_for_request.assert_any_call("synthetic-flash")
        self.assert_retried(plain, other)
        self.assertEqual(self.s._generation_overflow_recovery_ids, {"synthetic-plain", "synthetic-other-family"})

    async def test_the_prefill_requeue_fails_owned_work(self):
        flash, plain, other = await self.three()
        for request in (flash, plain, other):
            self.s.waiting.remove(request)
            self.s.requests.pop(request.request_id)  # the catch sites tear down first
        error = RuntimeError("Memory limit exceeded")
        self.assertFalse(self.s._requeue_or_fail_prefill(flash, error))
        self.assertTrue(self.s._requeue_or_fail_prefill(plain, error))
        self.assertTrue(self.s._requeue_or_fail_prefill(other, error))
        self.assert_ended(flash)
        self.assert_retried(plain, other)

    async def test_a_failed_packed_forward_rejects_owned_work_and_requeues_the_rest(self):
        for error in (RuntimeError("kernel"), MemoryError("Memory limit exceeded")):
            with self.subTest(type(error).__name__):
                await self.asyncTearDown(); await self.asyncSetUp()
                flash, plain, other = await self.three()
                plans = [SimpleNamespace(state=self.prefilling(r)) for r in (flash, plain, other)]
                rejected = []
                self.s._fail_packed_prefill(plans, error, rejected)
                self.assert_ended(flash, prefilling_rebuilt_by_caller=True)
                self.assert_error(rejected, flash)
                self.assertEqual([o.request_id for o in rejected], ["synthetic-flash"])
                self.assert_retried(plain, other)

    async def test_the_memory_enforcer_batch_requeue_retires_owned_work_and_requeues_the_rest(self):
        flash, plain, other = await self.three()
        for uid, request in enumerate((flash, plain, other), start=3):
            self.decoding(request, uid)
        s = self.s
        s._pending_memory_requeue = True
        s._memory_requeue_storage_ready = Mock(return_value=True)
        s._memory_requeue_blocker = Mock(return_value=None)
        s._memory_requeue_row = Mock(return_value=("row", 0))
        s.paged_ssd_cache_manager = SimpleNamespace(bounded_pending_writes=lambda n: contextlib.nullcontext())
        s._store_for_memory_requeue = Mock(return_value=3)
        s._save_requeue_row_state = Mock()

        def detach(request):
            s.running.pop(request.request_id, None)

        s._detach_for_memory_requeue = Mock(side_effect=detach)
        s._process_pending_memory_requeue()
        self.assert_retired(flash, uid=3)
        self.assert_retried(plain, other)
        self.assertEqual(s._memory_requeue_ids, {"synthetic-plain", "synthetic-other-family"})
        for mock_call in (s._store_for_memory_requeue, s._save_requeue_row_state, s._detach_for_memory_requeue):
            self.assertEqual(
                {c.args[0].request_id for c in mock_call.call_args_list},
                {"synthetic-plain", "synthetic-other-family"},
            )
        self.published(flash)

    def schedule_until(self, entry_point, raises):
        """Drive the real _schedule_waiting to one of its three eviction catches."""
        s = self.s
        s._prefill_gate_open.return_value = True
        s._prepare_prefix_cache_for_request = Mock(return_value=None)
        s._build_sampler_and_processors = Mock(return_value=(None, []))
        s._build_state_machine = Mock()
        s._decode_fairness = False
        s._is_harmony_model = False
        s._detect_needs_think_prefix = Mock(return_value=False)
        s._preflight_memory_check = Mock(return_value=None)
        s._do_external_prefill = Mock(side_effect=AssertionError("unexpected entry"))
        s._begin_prefill = Mock(side_effect=AssertionError("unexpected entry"))
        chunked = entry_point == "chunked"
        s.config = SimpleNamespace(
            max_num_seqs=1, chunked_prefill=chunked, prefill_step_size=0 if chunked else 2048, model_name="flash"
        )
        if entry_point == "preflight":
            s._preflight_memory_check = Mock(side_effect=raises)
        elif entry_point == "external":
            s._do_external_prefill = Mock(side_effect=raises)
        else:
            s._begin_prefill = Mock(return_value=SimpleNamespace())
            s._step_prefill_chunk = Mock(side_effect=raises)
        with patch.dict(running_fixture.namespace, dict(EXCEPTIONS, Scheduler=SCHEDULER_STUB)):
            return s._schedule_waiting()

    async def test_the_three_scheduling_eviction_catches_retire_owned_work_and_pause_the_rest(self):
        for entry_point in ("preflight", "external", "chunked"):
            for kind in ("flash", "plain", "other"):
                with self.subTest(entry_point=entry_point, request=kind):
                    await self.asyncTearDown(); await self.asyncSetUp()
                    if kind == "flash":
                        request = await self.owned("synthetic-flash")
                    elif kind == "plain":
                        request = await self.ordinary("synthetic-plain")
                    else:
                        request = await self.owned("synthetic-other-family", retry=True)
                    self.base.fetch(request)  # prefix blocks found by the lookup
                    eviction = SimpleNamespace(request_id=request.request_id, reason="synthetic")

                    def evict(*args, **kwargs):
                        if entry_point == "external":
                            # An earlier chunk captured a boundary snapshot, and
                            # _do_external_prefill kept its cache for a resume.
                            self.s._boundary_cache_snapshots[request.request_id] = object()
                            request.prompt_cache = [object()]
                        raise EvictionNeeded(eviction)

                    _, rejected = self.schedule_until(entry_point, evict)
                    self.assertEqual(rejected, [])
                    if kind == "flash":
                        self.assert_retired(request)
                        self.assertIsNone(self.s._pending_prefill_eviction_request)
                        self.published(request)
                    else:
                        self.assertIs(self.s.waiting[0], request)
                        self.assertIs(self.s._pending_prefill_eviction_request, eviction)
                        self.assertEqual(self.s._owned_no_retry_outputs, [])

    async def test_the_chunked_prefill_eviction_hold_retires_owned_work_and_holds_the_rest(self):
        for kind in ("flash", "plain", "other"):
            with self.subTest(request=kind):
                await self.asyncTearDown(); await self.asyncSetUp()
                if kind == "flash":
                    request = await self.owned("synthetic-flash")
                elif kind == "plain":
                    request = await self.ordinary("synthetic-plain")
                else:
                    request = await self.owned("synthetic-other-family", retry=True)
                self.prefilling(request)
                eviction = SimpleNamespace(request_id=request.request_id, reason="synthetic")
                self.s._prefill_gate_open.return_value = True
                self.s._step_prefill_chunk = Mock(side_effect=EvictionNeeded(eviction))
                rejected = []
                self.s._advance_chunked_prefills([], rejected)
                self.assertEqual(rejected, [])
                if kind == "flash":
                    self.assert_retired(request)
                    self.assertIsNone(self.s._pending_prefill_eviction_request)
                    self.published(request)
                else:
                    self.assertIs(self.s.prefilling[0], request)
                    self.assertIs(self.s._pending_prefill_eviction_request, eviction)

    def assert_failed_without_leftovers(self, request, rejected):
        """A terminal prefill failure of owned work: error, nothing left behind."""
        s, rid = self.s, request.request_id
        self.assert_ended(request, prefilling_rebuilt_by_caller=True)
        self.assert_error(rejected, request)
        self.assertNotIn(rid, s._boundary_cache_snapshots)
        # Strictly: the default form only logs a failed delete.
        s._boundary_snapshot_store.cleanup_request.assert_any_call(rid, strict=True)
        self.assertIsNone(request.prompt_cache)
        self.assertIsNone(request._extracted_cache)

    async def test_terminal_prefill_failures_release_snapshots_and_caches(self):
        """The hard-cap RuntimeError at every catch site that asks for a requeue."""
        hard_cap = RuntimeError("Memory limit exceeded during prefill")
        for entry_point in ("external", "chunked", "chunked_loop", "packed"):
            for kind in ("flash", "plain"):
                with self.subTest(entry_point=entry_point, request=kind):
                    await self.asyncTearDown(); await self.asyncSetUp()
                    request = await (self.owned("synthetic-flash") if kind == "flash" else self.ordinary("synthetic-plain"))
                    rid = request.request_id

                    def capture_then_fail(*args, **kwargs):
                        # A chunk crossed a cache boundary and kept its cache.
                        self.s._boundary_cache_snapshots[rid] = object()
                        request.prompt_cache = [object()]
                        request._extracted_cache = object()
                        raise hard_cap

                    if entry_point in ("external", "chunked"):
                        _, rejected = self.schedule_until(entry_point, capture_then_fail)
                    elif entry_point == "chunked_loop":
                        self.prefilling(request)
                        self.s._prefill_gate_open.return_value = True
                        self.s._step_prefill_chunk = Mock(side_effect=capture_then_fail)
                        rejected = []
                        with patch.dict(retry_ns, Scheduler=SCHEDULER_STUB):
                            self.s._advance_chunked_prefills([], rejected)
                    else:
                        plan = SimpleNamespace(state=self.prefilling(request))
                        with self.assertRaises(RuntimeError):
                            capture_then_fail()
                        rejected = []
                        self.s._fail_packed_prefill([plan], MemoryError("Memory limit exceeded"), rejected)
                    if kind == "flash":
                        self.assert_failed_without_leftovers(request, rejected)
                        self.assertEqual(request.prefill_oom_retries, 0)  # never retried
                    else:
                        # Requeued for a retry (the same pass may pick it up again).
                        self.assertGreaterEqual(request.prefill_oom_retries, 1)

    async def test_typed_capacity_refusals_are_absent_only_once_retired(self):
        """PrefillMemoryExceededError, the typed refusal, at its three catch sites."""
        for entry_point in ("external", "chunked", "chunked_loop"):
            for store_fails in (False, True):
                with self.subTest(entry_point=entry_point, store_fails=store_fails):
                    await self.asyncTearDown(); await self.asyncSetUp()
                    request = await self.owned("synthetic-flash")
                    handle, rid = request._prism_owned_admission, request.request_id

                    def capture_then_refuse(*args, **kwargs):
                        # An earlier chunk crossed a boundary; then the cap refuses.
                        self.s._boundary_cache_snapshots[rid] = object()
                        request.prompt_cache = [object()]
                        raise PrefillMemoryExceeded("minimum chunk exceeds the safety cap")

                    if store_fails:
                        self.s._boundary_snapshot_store.cleanup_request = Mock(side_effect=OSError("synthetic disk error"))
                    if entry_point == "chunked_loop":
                        self.prefilling(request)
                        self.s._prefill_gate_open.return_value = True
                        self.s._step_prefill_chunk = Mock(side_effect=capture_then_refuse)
                        rejected = []
                        self.s._advance_chunked_prefills([], rejected)
                    else:
                        _, rejected = self.schedule_until(entry_point, capture_then_refuse)
                    self.assert_error(rejected, request)
                    self.assertNotIn(rid, self.s.requests)
                    self.assertEqual(request.prefill_oom_retries, 0)
                    outcome = self.s._probe_owned_presence(handle)
                    if store_fails:
                        self.assertEqual(outcome, "unknown")
                        self.assertIn(rid, self.s._owned_retirement_failures)
                    else:
                        self.assertEqual(outcome, "absent")
                        self.assertNotIn(rid, self.s._boundary_cache_snapshots)
                        self.assertIsNone(request.prompt_cache)
                        self.s._boundary_snapshot_store.cleanup_request.assert_any_call(rid, strict=True)

    async def test_the_probe_never_retires_other_requests(self):
        request = await self.owned("synthetic-other-family", retry=True)
        handle, rid = request._prism_owned_admission, request.request_id
        self.s.waiting.remove(request)
        self.s.requests.pop(rid)
        self.s._boundary_cache_snapshots[rid] = snapshot = object()
        self.assertEqual(self.s._probe_owned_presence(handle), "absent")  # unchanged
        self.assertIs(self.s._boundary_cache_snapshots[rid], snapshot)
        self.s._boundary_snapshot_store.cleanup_request.assert_not_called()

    async def test_a_failed_snapshot_cleanup_is_never_reported_absent(self):
        request = await self.owned("synthetic-flash")
        handle = request._prism_owned_admission

        def capture_then_fail(*args, **kwargs):
            self.s._boundary_cache_snapshots[request.request_id] = object()
            raise RuntimeError("Memory limit exceeded during prefill")

        self.s._boundary_snapshot_store.cleanup_request = Mock(side_effect=OSError("synthetic disk error"))
        _, rejected = self.schedule_until("external", capture_then_fail)
        self.assert_error(rejected, request)  # the request still ends with its error
        self.assertNotIn(request.request_id, self.s.requests)
        self.assertEqual(self.s._owned_retirement_failures, {"synthetic-flash"})
        # Gone from the queues, but not proved retired: never "absent".
        self.assertEqual(self.s._probe_owned_presence(handle), "unknown")

    async def test_a_successful_terminal_cleanup_is_absent(self):
        request = await self.owned("synthetic-flash")
        handle = request._prism_owned_admission
        _, rejected = self.schedule_until("external", RuntimeError("Memory limit exceeded during prefill"))
        self.assert_error(rejected, request)
        self.assertEqual(self.s._owned_retirement_failures, set())
        self.assertEqual(self.s._probe_owned_presence(handle), "absent")

    async def test_a_failed_abort_cleanup_is_never_reported_absent(self):
        request = await self.owned("synthetic-flash")
        handle = request._prism_owned_admission
        self.s._do_abort_request = Mock(side_effect=OSError("synthetic cleanup failure"))
        with self.assertRaises(OSError):
            self.s._end_owned_without_retry(request, "reschedule")
        self.assertEqual(self.s._owned_retirement_failures, {"synthetic-flash"})
        self.assertEqual(self.s._owned_no_retry_outputs, [])
        self.s.requests.pop("synthetic-flash")  # whatever the failed cleanup got to
        self.assertEqual(self.s._probe_owned_presence(handle), "unknown")

    async def test_an_unmarked_request_is_left_alone_by_the_end_helper(self):
        plain = await self.ordinary("synthetic-plain")
        other = await self.owned("synthetic-other-family", retry=True)
        for request in (plain, other):
            self.assertFalse(self.s._end_owned_without_retry(request, "reschedule"))
            self.assertIs(self.s.requests[request.request_id], request)
        self.assertEqual(self.s._owned_no_retry_outputs, [])


def tracker_removed():
    return running_fixture.tracker.removed



# ---------------------------------------------------------------------------
# Counted host-read submissions: termination.

from omlx.owned_admission import RunningCleaned, StreamSynchronized, WaitingCleaned  # noqa: E402
import test_provider_owned_admission_stage as stage_fixture  # noqa: E402
import test_provider_scheduler_cleanup as cleanup_fixture  # noqa: E402


class FakeCore:
    _closed = False
    _closing = False

    def __init__(self):
        self.cleaned = []

    async def probe_owned_absence(self, handle, deadline):
        return "absent"

    async def _owned_stream_barrier(self, handle, deadline):
        return StreamSynchronized(handle, None, "generation", "synthetic")

    def _cleanup_request(self, request_id):
        self.cleaned.append(request_id)


class CountedTerminationTests(unittest.IsolatedAsyncioTestCase):
    def execution(self, fact=None):
        task = None
        if fact is not None:
            task = asyncio.get_running_loop().create_future()
            task.set_result(fact)
        return SimpleNamespace(
            drain_deadline=time.monotonic() + 3,
            claim=SimpleNamespace(request_id="synthetic"),
            cleanup_task=task,
            cancellations=[],
        )

    def cases(self):
        handle = object()
        return {
            "running_cleaned": (
                RunningCleaned(handle, None, "generation", "synthetic", "decoding"), None,
                ("cancelled", ("running_cleaned", "stream_synchronized")),
            ),
            "waiting_cleaned": (
                WaitingCleaned(handle, None, "generation", "synthetic"), None,
                ("cancelled", ("waiting_cleaned", "stream_synchronized")),
            ),
            "completed": (
                None, RequestOutput(request_id="synthetic", finished=True, finish_reason="stop"),
                ("completed", ("finished_output", "absent_proved", "stream_synchronized")),
            ),
            "failed": (
                None, RequestOutput(request_id="synthetic", finished=True, finish_reason="error", error="x"),
                ("failed", ("error_output", "absent_proved", "stream_synchronized")),
            ),
        }

    async def test_an_unchanged_count_keeps_every_outcome(self):
        for name, (fact, final, expected) in self.cases().items():
            with self.subTest(name), patch.object(prep, "submission_failures", lambda: 3):
                outcome = await prep._terminate(self.execution(fact), FakeCore(), object(), final, 3)
                self.assertEqual(outcome, expected)

    async def test_a_moved_count_never_publishes_any_outcome(self):
        for name, (fact, final, _) in self.cases().items():
            with self.subTest(name), patch.object(prep, "submission_failures", lambda: 4):
                with self.assertRaises(prep.ExecutionUnresolved) as caught:
                    await prep._terminate(self.execution(fact), FakeCore(), object(), final, 3)
                self.assertEqual(str(caught.exception), "host_read_ambiguous")


class CountedStageTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.base = stage_fixture.AdmissionStageTests()
        await self.base.asyncSetUp()
        bind_owned_retirement(self.base.s)
        self.count = 0
        self.patch = patch.object(prep, "submission_failures", lambda: self.count)
        self.patch.start()

    async def asyncTearDown(self):
        self.patch.stop()
        await self.base.asyncTearDown()

    async def test_a_count_moved_during_an_admitted_execution_leaves_it_unresolved(self):
        ex = self.base.execution()
        task = asyncio.create_task(self.base.stage(ex))
        await self.base.until(lambda: ex.admitted)
        self.count += 1  # an ambiguous submission while the request runs
        collector = self.base.core._output_collectors["synthetic-owned"]
        await self.base.feed(collector, stage_fixture.out("ok", finished=True))
        await self.base.until(lambda: ex.final_output is not None)
        self.base.finish_in_scheduler("synthetic-owned")
        with self.assertRaises(prep.ExecutionUnresolved) as caught:
            await asyncio.wait_for(task, 5)
        self.assertEqual(str(caught.exception), "host_read_ambiguous")
        self.assertEqual(self.base.pool.releases, ["m"])  # the lease is still released

    async def test_a_pressure_family_admission_is_marked_no_retry(self):
        for family, retry in (("qwen4_exp", False), ("qwen4-exp", False), ("deepseek_v41", False), ("llama", True), (None, True)):
            with self.subTest(family=family):
                self.base.pool.entry.config_model_type = family
                ex = self.base.execution(rid=f"synthetic-{family}")
                task = asyncio.create_task(self.base.stage(ex))
                await self.base.until(lambda: ex.admitted)
                self.assertIs(self.base.engine.last_admission.retry, retry)
                collector = self.base.core._output_collectors[f"synthetic-{family}"]
                await self.base.feed(collector, stage_fixture.out("ok", finished=True))
                await self.base.until(lambda: ex.final_output is not None)
                self.base.finish_in_scheduler(f"synthetic-{family}")
                self.assertEqual(await asyncio.wait_for(task, 5), stage_fixture.COMPLETED)

    async def test_proven_non_admission_is_exempt_from_the_count(self):
        def refuse(*args, **kwargs):
            self.count += 1  # an ordinary request's failed submission, meanwhile
            raise RuntimeError("synthetic scheduler preflight")

        self.base.s.preflight_or_raise = Mock(side_effect=refuse)
        outcome = await self.base.stage(self.base.execution(rid="synthetic-refused"))
        self.assertEqual(outcome, ("not_started", ("admission_refused", "stream_synchronized")))
        self.assertEqual(self.count, 1)


class EnqueueThenRaise:
    """ThreadPoolExecutor.submit in order: enqueue, then a failed worker start.

    The read runs later on the one worker, held until ``release``; the caller
    gets the exception, not the future."""

    def __init__(self):
        self.inner = ThreadPoolExecutor(max_workers=1)
        self.release = threading.Event()

    def submit(self, fn, /, *args, **kwargs):
        def hidden():
            self.release.wait(10)
            return fn(*args, **kwargs)

        self.inner.submit(hidden)
        raise RuntimeError("can't start new thread")


class HostReadIntegrationTests(unittest.IsolatedAsyncioTestCase):
    """The real counter and a real counted submission, around a real owned execution."""

    async def asyncSetUp(self):
        self.base = stage_fixture.AdmissionStageTests()
        await self.base.asyncSetUp()
        bind_owned_retirement(self.base.s)
        # The count never decreases in a process; restore it for later tests.
        self.restore = patch.object(host_submit, "_failures", host_submit._failures)
        self.restore.start()
        self.executor = EnqueueThenRaise()
        self.read_ran = threading.Event()

    async def asyncTearDown(self):
        self.executor.release.set()
        self.executor.inner.shutdown(wait=True)
        self.restore.stop()
        await self.base.asyncTearDown()

    def fail_a_submission(self):
        before = host_submit.submission_failures()
        with self.assertRaises(RuntimeError):
            host_submit.submit_tracked(self.executor, self.read_ran.set)
        self.assertEqual(host_submit.submission_failures(), before + 1)

    async def test_a_hidden_read_during_an_owned_execution_leaves_it_unresolved(self):
        ex = self.base.execution()
        task = asyncio.create_task(self.base.stage(ex))
        await self.base.until(lambda: ex.admitted)
        self.fail_a_submission()  # e.g. an ordinary request's MoE read, meanwhile
        collector = self.base.core._output_collectors["synthetic-owned"]
        await self.base.feed(collector, stage_fixture.out("ok", finished=True))
        await self.base.until(lambda: ex.final_output is not None)
        self.base.finish_in_scheduler("synthetic-owned")
        with self.assertRaises(prep.ExecutionUnresolved) as caught:
            await asyncio.wait_for(task, 5)
        self.assertEqual(str(caught.exception), "host_read_ambiguous")
        # The read had not run when the execution ended; it runs anyway.
        self.assertFalse(self.read_ran.is_set())
        self.executor.release.set()
        self.assertTrue(await asyncio.to_thread(self.read_ran.wait, 10))
        self.assertEqual(self.base.pool.releases, ["m"])

    async def test_a_failure_after_preparation_admits_nothing(self):
        self.base.engine.before_admit = lambda handle: self.fail_a_submission()
        ex = self.base.execution(rid="synthetic-late")
        with self.assertRaises(coord_unavailable()):
            await asyncio.wait_for(self.base.stage(ex), 5)
        handle = self.base.engine.last_admission
        self.assertFalse(handle._attempted)  # the executor dispatch inserted nothing
        self.assertNotIn("synthetic-late", self.base.s.requests)
        self.assertFalse(ex.admitted)
        self.assertEqual(self.base.pool.releases, ["m"])

    async def test_an_unmoved_count_admits_and_completes(self):
        ex = self.base.execution(rid="synthetic-clean")
        task = asyncio.create_task(self.base.stage(ex))
        await self.base.until(lambda: ex.admitted)
        self.assertTrue(self.base.engine.last_admission._attempted)
        await self.base.feed(self.base.core._output_collectors["synthetic-clean"], stage_fixture.out("ok", finished=True))
        await self.base.until(lambda: ex.final_output is not None)
        self.base.finish_in_scheduler("synthetic-clean")
        self.assertEqual(await asyncio.wait_for(task, 5), stage_fixture.COMPLETED)


class RetirementFailureStageTests(unittest.IsolatedAsyncioTestCase):
    """Owned termination through the real presence probe."""

    async def asyncSetUp(self):
        self.base = stage_fixture.AdmissionStageTests()
        await self.base.asyncSetUp()
        bind_owned_retirement(self.base.s)

    async def asyncTearDown(self):
        await self.base.asyncTearDown()

    async def test_a_failed_store_cleanup_at_termination_ends_unresolved(self):
        # Flash-Next work (retry=False) whose request a terminal path removed:
        # the real probe retires its snapshots strictly before "absent".
        self.base.pool.entry.config_model_type = "qwen4_exp"
        self.base.s._boundary_snapshot_store.cleanup_request = Mock(side_effect=TimeoutError("writer busy"))
        ex = self.base.execution(rid="synthetic-flash-term")
        task = asyncio.create_task(self.base.stage(ex))
        await self.base.until(lambda: ex.admitted)
        self.base.s._boundary_cache_snapshots["synthetic-flash-term"] = object()
        await self.base.feed(
            self.base.core._output_collectors["synthetic-flash-term"],
            stage_fixture.err("error", "synthetic capacity refusal"),
        )
        await self.base.until(lambda: ex.final_output is not None)
        self.base.finish_in_scheduler("synthetic-flash-term")
        with self.assertRaises(prep.ExecutionUnresolved) as caught:
            await asyncio.wait_for(task, 5)
        self.assertEqual(str(caught.exception), "probe_unknown")
        self.assertIn("synthetic-flash-term", self.base.s._owned_retirement_failures)

    async def test_a_retired_terminal_ends_failed_absent_proved(self):
        self.base.pool.entry.config_model_type = "qwen4_exp"
        ex = self.base.execution(rid="synthetic-flash-ok")
        task = asyncio.create_task(self.base.stage(ex))
        await self.base.until(lambda: ex.admitted)
        self.base.s._boundary_cache_snapshots["synthetic-flash-ok"] = object()
        await self.base.feed(
            self.base.core._output_collectors["synthetic-flash-ok"],
            stage_fixture.err("error", "synthetic capacity refusal"),
        )
        await self.base.until(lambda: ex.final_output is not None)
        self.base.finish_in_scheduler("synthetic-flash-ok")
        self.assertEqual(
            await asyncio.wait_for(task, 5),
            ("failed", ("error_output", "absent_proved", "stream_synchronized")),
        )
        self.assertNotIn("synthetic-flash-ok", self.base.s._boundary_cache_snapshots)
        self.base.s._boundary_snapshot_store.cleanup_request.assert_any_call("synthetic-flash-ok", strict=True)

    async def test_an_unproved_retirement_ends_unresolved_not_absent_proved(self):
        ex = self.base.execution()
        task = asyncio.create_task(self.base.stage(ex))
        await self.base.until(lambda: ex.admitted)
        await self.base.feed(
            self.base.core._output_collectors["synthetic-owned"],
            stage_fixture.err("error", "synthetic prefill failure"),
        )
        await self.base.until(lambda: ex.final_output is not None)
        self.base.s._owned_retirement_failures = {"synthetic-owned"}
        self.base.finish_in_scheduler("synthetic-owned")
        with self.assertRaises(prep.ExecutionUnresolved) as caught:
            await asyncio.wait_for(task, 5)
        self.assertEqual(str(caught.exception), "probe_unknown")
        self.assertEqual(self.base.pool.releases, ["m"])

    async def test_a_failed_deferred_row_removal_ends_unresolved(self):
        # A finished Flash-Next request whose cache store deferred its batch
        # row removal; the real drain's removal fails, yet it drops the request.
        self.base.pool.entry.config_model_type = "qwen4_exp"
        s = self.base.s
        rid = "synthetic-flash-row"
        ex = self.base.execution(rid=rid)
        task = asyncio.create_task(self.base.stage(ex))
        await self.base.until(lambda: ex.admitted)
        self.assertIs(self.base.engine.last_admission.retry, False)
        await self.base.feed(self.base.core._output_collectors[rid], stage_fixture.out("ok", finished=True))
        await self.base.until(lambda: ex.final_output is not None)
        drain_onto(s, failing_uid=7)
        store = Future()
        store.set_result(None)

        def finish_with_deferred_remove():
            s.waiting.remove(s.requests[rid])
            s._pending_async_removes.append((7, rid, store))
            s._inflight_store_futures[rid] = store
            s.uid_to_request_id[7] = rid
            s.request_id_to_uid[rid] = 7
            with self.assertLogs(level="WARNING"):
                s._drain_pending_async_removes()

        # On the MLX executor, where the step and the probe run.
        await asyncio.get_running_loop().run_in_executor(self.base.core._mlx_executor, finish_with_deferred_remove)
        s._remove_uid_from_active_batch.assert_called_once_with(7)
        self.assertNotIn(rid, s.requests)  # the inherited drain still drops it
        with self.assertRaises(prep.ExecutionUnresolved) as caught:
            await asyncio.wait_for(task, 5)
        self.assertEqual(str(caught.exception), "probe_unknown")
        self.assertIn(rid, s._owned_retirement_failures)
        self.assertEqual(self.base.pool.releases, ["m"])


def drain_onto(s, failing_uid):
    """The real deferred drain on a stage scheduler, its row removal failing."""

    def remove(uid):
        if uid == failing_uid:
            raise RuntimeError("synthetic remove failure")

    s._drain_pending_async_removes = types.MethodType(
        cleanup_fixture.TestedScheduler._drain_pending_async_removes, s
    )
    s._owned_no_retry = running_fixture.RunningScheduler._owned_no_retry
    s._remove_uid_from_active_batch = Mock(side_effect=remove)
    s._tracked_deferred_cleanups = {}
    s._inflight_store_info = {}
    s._clear_request_admission_bookkeeping = Mock()
    s._store_cache_gate = None
    s._schedule_deferred_metal_clear = Mock()
    s._note_memory_admission_progress = Mock()


def owned_handle(rid, retry):
    core = SimpleNamespace(
        _prism_control_generation_id=1, _engine_id="e", scheduler=object(),
        _mlx_executor=object(), _mlx_stream=object(),
    )
    return OwnedAdmission(core, rid, time.monotonic() + 30, retry=retry)


class DeferredRemoveRecordTests(unittest.TestCase):
    """The real deferred drain: a failed row removal of owned no-retry work is
    recorded; every other request keeps the inherited behavior."""

    def drain(self, retry, *, fails=True):
        case = cleanup_fixture.SchedulerCleanupTests("test_unfinished_store_is_retained_without_cleanup")
        case.setUp()
        s = case.s
        req, future, _ = case.add("owned", 21)
        if retry is not None:
            req._prism_owned_admission = owned_handle("owned", retry)
        future.set_result(None)
        if fails:
            s._remove_uid_from_active_batch.side_effect = RuntimeError("synthetic remove failure")
        with self.assertLogs(level="WARNING") if fails else contextlib.nullcontext():
            s._drain_pending_async_removes()
        s._remove_uid_from_active_batch.assert_any_call(21)
        self.assertNotIn("owned", s.requests)  # inherited: the drain still drops it
        self.assertNotIn(21, s.uid_to_request_id)
        return getattr(s, "_owned_retirement_failures", set())

    def test_a_failed_removal_of_no_retry_work_is_recorded(self):
        self.assertEqual(self.drain(retry=False), {"owned"})

    def test_retried_owned_and_ordinary_work_keep_the_inherited_drain(self):
        for retry in (True, None):
            with self.subTest(retry=retry):
                self.assertEqual(self.drain(retry=retry), set())

    def test_a_successful_removal_records_nothing(self):
        self.assertEqual(self.drain(retry=False, fails=False), set())


def coord_unavailable():
    from omlx.api.prism_control_coordinator import EngineStageUnavailable

    return EngineStageUnavailable


# ---------------------------------------------------------------------------
# Source guards: retry sites and host-read submissions.

SCHEDULER_SOURCE = running_fixture.SOURCE.read_text()


def enclosing_functions(tree):
    owners = {}
    for function in ast.walk(tree):
        if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for node in ast.walk(function):
                owners.setdefault(id(node), function.name)
    return owners


class SourceGuardTests(unittest.TestCase):
    def setUp(self):
        self.tree = ast.parse(SCHEDULER_SOURCE)
        self.owners = enclosing_functions(self.tree)
        self.functions = {
            n.name: n for n in ast.walk(self.tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

    def calls(self, attribute, node=None):
        return [
            n for n in ast.walk(node or self.tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == attribute
        ]

    def mentions(self, node, name):
        return any(
            isinstance(n, ast.Attribute) and n.attr == name for n in ast.walk(node)
        )

    def test_every_waiting_insert_is_listed(self):
        inserts = {}
        for call in ast.walk(self.tree):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr in ("append", "appendleft", "extend", "extendleft", "insert")
                and isinstance(call.func.value, ast.Attribute)
                and call.func.value.attr == "waiting"
            ):
                owner = self.owners[id(call)]
                inserts[owner] = inserts.get(owner, 0) + 1
        self.assertEqual(
            inserts,
            {
                # Admission and deferrals before compute in this attempt.
                "add_request": 1,
                "_schedule_waiting": 8,
                # Retries: each consults the owned no-retry mark (below).
                "_reschedule_running_requests": 1,
                "_reschedule_generation_overflow_requests": 1,
                "_requeue_prefill_retry": 1,
                "_process_pending_memory_requeue": 1,
                "_pause_for_prefill_eviction": 1,
            },
        )
        for name in (
            "_reschedule_running_requests",
            "_reschedule_generation_overflow_requests",
            "_process_pending_memory_requeue",
            "_requeue_or_fail_prefill",
            "_fail_packed_prefill",
        ):
            with self.subTest(name):
                function = self.functions[name]
                self.assertTrue(
                    self.mentions(function, "_owned_no_retry")
                    or self.mentions(function, "_end_owned_without_retry")
                )

    def test_every_requeue_and_eviction_pause_is_guarded_by_the_owned_mark(self):
        for call in self.calls("_requeue_prefill_retry"):
            owner = self.owners[id(call)]
            with self.subTest(requeue_in=owner):
                self.assertIn(owner, {"_requeue_or_fail_prefill", "_fail_packed_prefill"})
                self.assertTrue(self.mentions(self.functions[owner], "_owned_no_retry"))
        handlers = [
            h for h in ast.walk(self.tree)
            if isinstance(h, ast.ExceptHandler) and isinstance(h.type, ast.Name) and h.type.id == "_PrefillEvictionNeeded"
        ]
        # One re-raises from inside external prefill to its guarded caller.
        passthrough = [h for h in handlers if self.owners[id(h)] == "_do_external_prefill"]
        self.assertEqual(len(passthrough), 1)
        self.assertIsInstance(passthrough[0].body[-1], ast.Raise)
        self.assertIsNone(passthrough[0].body[-1].exc)
        self.assertFalse(self.calls("_pause_for_prefill_eviction", passthrough[0]))
        self.assertFalse(self.mentions(passthrough[0], "_pending_prefill_eviction_request"))
        handlers = [h for h in handlers if h not in passthrough]
        self.assertEqual(len(handlers), 4)
        for handler in handlers:
            with self.subTest(handler=self.owners[id(handler)], line=handler.lineno):
                guard = next(
                    (
                        s for s in handler.body
                        if isinstance(s, ast.If) and self.mentions(s.test, "_end_owned_without_retry")
                    ),
                    None,
                )
                self.assertIsNotNone(guard)
                index = handler.body.index(guard)
                before = handler.body[:index]
                # Nothing before the guard pauses, holds or publishes the eviction.
                for statement in before:
                    self.assertFalse(self.calls("_pause_for_prefill_eviction", statement))
                    self.assertFalse(self.mentions(statement, "_pending_prefill_eviction_request"))
                    self.assertFalse(self.calls("append", statement) and self.mentions(statement, "still_prefilling"))
                self.assertIsInstance(guard.body[-1], ast.Continue)
                self.assertFalse(self.calls("_pause_for_prefill_eviction", guard))
                self.assertFalse(self.mentions(guard, "_pending_prefill_eviction_request"))
        pauses = self.calls("_pause_for_prefill_eviction")
        self.assertEqual(len(pauses), 3)
        self.assertTrue(all(any(p in ast.walk(h) for h in handlers) for p in pauses))

    def test_the_step_publishes_ended_owned_work_once_before_returning(self):
        step = self.functions["_step_unguarded"]
        calls = self.calls("_publish_owned_no_retry_outputs", step)
        self.assertEqual(len(calls), 1)
        self.assertIsInstance(step.body[-1], ast.Return)
        self.assertTrue(self.calls("_publish_owned_no_retry_outputs", step.body[-2]))

    def test_every_host_read_submission_is_counted(self):
        files = (
            ROOT / "omlx/patches/moe_expert_offload.py",
            ROOT / "omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/language.py",
        )
        for path in files:
            with self.subTest(path.name):
                tree = ast.parse(path.read_text())
                bare = [
                    n.lineno for n in ast.walk(tree)
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "submit"
                ]
                self.assertEqual(bare, [])
                tracked = [
                    n for n in ast.walk(tree)
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "submit_tracked"
                ]
                self.assertEqual(len(tracked), 1 if path.name == "moe_expert_offload.py" else 2)

    def test_the_loader_passes_the_persisted_gather_ahead_setting(self):
        tree = ast.parse((ROOT / "omlx/utils/model_loading.py").read_text())
        calls = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "configure_qwen4_exp_runtime"
        ]
        self.assertEqual(len(calls), 1)
        keyword = next((k for k in calls[0].keywords if k.arg == "ple_gather_ahead"), None)
        self.assertIsNotNone(keyword)
        self.assertIn("getattr(model_settings, 'qwen4_ple_gather_ahead', True)", ast.unparse(keyword.value))

    def test_the_gather_ahead_gate_precedes_the_only_executor_submission(self):
        path = ROOT / "omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/language.py"
        tree = ast.parse(path.read_text())
        table = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DiskBackedShardedEmbedding")
        prefetch = next(n for n in table.body if isinstance(n, ast.FunctionDef) and n.name == "prefetch")
        first = prefetch.body[1] if isinstance(prefetch.body[0], ast.Expr) else prefetch.body[0]
        self.assertIsInstance(first, ast.If)
        self.assertEqual(ast.unparse(first.test), "not self.gather_ahead")
        self.assertIsInstance(first.body[0], ast.Return)
        executor_submissions = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "submit_tracked"
            and ast.unparse(n.args[0]) == "self._prefetch_executor"
        ]
        self.assertEqual(len(executor_submissions), 1)
        self.assertTrue(any(executor_submissions[0] in list(ast.walk(prefetch)) for _ in [0]))
