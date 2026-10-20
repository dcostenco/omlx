# SPDX-License-Identifier: Apache-2.0
"""CPU-only ASGI control-status tests; never import the inference server."""

import ast
from hashlib import sha256
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(os.environ["OMLX_CONTROL_ROOT"])


class ForbidEngines(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (
            fullname == "mlx"
            or fullname.startswith("mlx.")
            or fullname == "omlx.server"
        ):
            raise RuntimeError("engine_import_forbidden")


sys.meta_path.insert(0, ForbidEngines())
sys.path.insert(0, str(ROOT))
from fastapi import FastAPI
from fastapi.testclient import TestClient
from omlx.api.prism_control_routes import create_control_router, configure_control_store

# Load the exact existing auth utility without admin/__init__ importing its UI.
# Keep the CPU no-engine boundary: load the real network utility without
# executing utils/__init__.py, which imports MLX hardware readers.
from types import ModuleType
utils_package = ModuleType("omlx.utils")
utils_package.__path__ = [str(ROOT / "omlx/utils")]
sys.modules.setdefault("omlx.utils", utils_package)
network_spec = importlib.util.spec_from_file_location(
    "omlx.utils.network", ROOT / "omlx/utils/network.py"
)
network_module = importlib.util.module_from_spec(network_spec)
sys.modules[network_spec.name] = network_module
network_spec.loader.exec_module(network_module)

spec = importlib.util.spec_from_file_location(
    "omlx.admin.tested_control_auth", ROOT / "omlx/admin/auth.py"
)
auth = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = auth
spec.loader.exec_module(auth)
spec = importlib.util.spec_from_file_location(
    "tested_control_network", ROOT / "omlx/utils/network.py"
)
network = importlib.util.module_from_spec(spec)
spec.loader.exec_module(network)
is_loopback_bind = network.is_loopback_bind

KEY = "synthetic-main-control-key"
SUB = "synthetic-sub-control-key"
JOB = "11111111-1111-4111-8111-111111111111"
OP = "22222222-2222-4222-8222-222222222222"
PATH = f"/prism-control/v1/jobs/{JOB}/operations/{OP}"
OWNER = sha256(KEY.encode()).hexdigest()
REGISTRATION = dict(
    owner_id=OWNER,
    job_id=JOB,
    operation_id=OP,
    model_id="synthetic-model",
    request_sha256="c" * 64,
    settings_sha256="d" * 64,
)


class ControlRouteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="prism-control-route-")
        self.addCleanup(self.temp.cleanup)
        settings = SimpleNamespace(
            auth=SimpleNamespace(
                sub_keys=[SimpleNamespace(key=SUB)],
                skip_api_key_verification=True,
                allow_unauthenticated_inference=True,
            )
        )
        self.state = SimpleNamespace(
            api_key=KEY,
            bind_host="127.0.0.1",
            global_settings=settings,
            prism_control_store=None,
        )
        configure_control_store(self.state, Path(self.temp.name), is_loopback_bind)
        self.record = self.state.prism_control_store.register(**REGISTRATION)
        app = FastAPI()
        app.include_router(
            create_control_router(
                lambda: self.state, auth.verify_any_api_key, is_loopback_bind
            )
        )
        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def get(self, **options):
        return self.client.get(
            PATH, headers={"Authorization": "Bearer " + KEY}, **options
        )

    def test_exact_authenticated_record_is_delivered_without_engine_authority(self):
        result = self.get()
        self.assertEqual(result.status_code, 200)
        body = result.json()
        self.assertEqual(body["job_id"], JOB)
        self.assertEqual(body["store_generation_id"], self.record.generation_id)
        self.assertEqual(body["state"], "registered")
        for field in [
            "backend_stopped",
            "engine_generation_verified",
            "provider_qualified",
            "production_enabled",
        ]:
            self.assertIs(body[field], False)
        self.assertNotIn("owner_id", body)
        self.assertNotIn(KEY, result.text)
        self.assertNotIn(SUB, result.text)

    def test_no_auth_or_wrong_auth_is_refused_even_with_inference_bypasses(self):
        for headers in [
            {},
            {"Authorization": "Bearer wrong-key"},
            {"x-api-key": "wrong-key"},
        ]:
            with self.subTest(headers=headers):
                self.assertEqual(
                    self.client.get(PATH, headers=headers).status_code, 401
                )
        self.assertEqual(
            self.client.get(PATH, headers={"x-api-key": KEY}).status_code, 200
        )

    def test_missing_configured_main_key_never_allows_unauthenticated_control(self):
        for value in [None, ""]:
            self.state.api_key = value
            self.assertEqual(self.get().status_code, 401)

    def test_public_unknown_or_mixed_bind_is_refused_before_store_access(self):
        for host in [None, "0.0.0.0", "::", "127.0.0.1,0.0.0.0"]:
            self.state.bind_host = host
            self.assertEqual(self.get().status_code, 403)

    def test_sub_key_cannot_read_main_namespace_and_revocation_is_immediate(self):
        headers = {"Authorization": "Bearer " + SUB}
        self.assertEqual(self.client.get(PATH, headers=headers).status_code, 404)
        self.state.prism_control_store.register(
            **(REGISTRATION | dict(owner_id=sha256(SUB.encode()).hexdigest()))
        )
        self.assertEqual(self.client.get(PATH, headers=headers).status_code, 200)
        self.state.global_settings.auth.sub_keys = []
        self.assertEqual(self.client.get(PATH, headers=headers).status_code, 401)

    def test_query_parameters_cannot_override_authenticated_owner_or_stop_status(self):
        result = self.get(
            params={
                "owner_id": "b" * 64,
                "backend_stopped": "true",
                "production_enabled": "true",
            }
        )
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["model_id"], REGISTRATION["model_id"])
        self.assertFalse(result.json()["backend_stopped"])
        self.assertFalse(result.json()["production_enabled"])

    def test_get_preserves_database_bytes_and_cannot_execute(self):
        path = self.state.prism_control_store.path
        before = path.read_bytes()
        self.assertEqual(self.get().status_code, 200)
        self.assertEqual(path.read_bytes(), before)
        headers = {"Authorization": "Bearer " + KEY}
        for method, url in [("PUT", PATH), ("POST", PATH)]:
            self.assertIn(
                self.client.request(method, url, headers=headers).status_code,
                [404, 405],
            )
        self.assertEqual(path.read_bytes(), before)

    def test_invalid_identity_and_unavailable_state_return_opaque_errors(self):
        headers = {"Authorization": "Bearer " + KEY}
        result = self.client.get(PATH.replace(JOB, "invalid-identity"), headers=headers)
        self.assertEqual(result.status_code, 400)
        self.assertNotIn("invalid-identity", result.text)
        self.state.prism_control_store = None
        self.assertEqual(self.get().status_code, 503)

    def test_status_does_not_import_engines_or_installed_state(self):
        self.assertNotIn("mlx.core", sys.modules)
        self.assertNotIn("omlx.server", sys.modules)
        self.assertTrue(
            self.state.prism_control_store.path.is_relative_to(Path(self.temp.name))
        )

    def test_store_setup_failure_preserves_existing_inference_state(self):
        backend = object()
        self.state.engine_pool = backend
        unavailable = Path(self.temp.name) / "bad-base"
        unavailable.mkdir()
        (unavailable / "prism-control").symlink_to(
            Path(self.temp.name) / "prism-control", target_is_directory=True
        )
        configure_control_store(self.state, unavailable, is_loopback_bind)
        self.assertIsNone(self.state.prism_control_store)
        self.assertIs(self.state.engine_pool, backend)
        self.assertEqual(self.state.api_key, KEY)
        self.assertEqual(self.get().status_code, 503)

    def test_oversized_credentials_are_refused_without_reflecting_or_reading_state(
        self,
    ):
        before = self.state.prism_control_store.path.read_bytes()
        oversized = "x" * 5000
        for headers in [
            {"Authorization": "Bearer " + oversized},
            {"x-api-key": oversized},
        ]:
            result = self.client.get(PATH, headers=headers)
            self.assertEqual(result.status_code, 401)
            self.assertNotIn(oversized, result.text)
        self.assertEqual(self.state.prism_control_store.path.read_bytes(), before)

    def test_source_server_declares_router_and_startup_store_delivery(self):
        # Static wiring is separate evidence from this exercised ASGI router.
        tree = ast.parse((ROOT / "omlx/server.py").read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        self.assertTrue(
            any(
                isinstance(node.func, ast.Name)
                and node.func.id == "configure_control_store"
                for node in calls
            )
        )
        self.assertTrue(
            any(
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "include_router"
                and node.args
                and isinstance(node.args[0], ast.Call)
                and isinstance(node.args[0].func, ast.Name)
                and node.args[0].func.id == "create_control_router"
                for node in calls
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
