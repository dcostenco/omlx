# SPDX-License-Identifier: Apache-2.0
"""Registration transport only: CPU ASGI, no engine admission or installed state."""

from copy import deepcopy
import ast
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
import uuid
from unittest import mock

import test_provider_control_routes as support
from omlx.api.prism_control_routes import create_control_router, configure_control_store
from omlx.api import prism_control_routes as routes
from fastapi import FastAPI
from fastapi.testclient import TestClient

REQUEST = {
    "model": "synthetic-model",
    "input": "synthetic-private-input-marker",
    "max_output_tokens": 32,
    "stream": True,
    "store": True,
    "truncation": "disabled",
}


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="prism-control-register-")
        self.addCleanup(self.temp.cleanup)
        settings = SimpleNamespace(
            auth=SimpleNamespace(
                sub_keys=[],
                skip_api_key_verification=True,
                allow_unauthenticated_inference=True,
            )
        )
        self.state = SimpleNamespace(
            api_key=support.KEY,
            bind_host="127.0.0.1",
            global_settings=settings,
            prism_control_store=None,
        )
        configure_control_store(
            self.state, Path(self.temp.name), support.is_loopback_bind
        )
        self.config = {
            "snapshot_version": 1,
            "model_settings": {"thinking_budget": 256},
            "sampling_defaults": {"temperature": 1.0},
        }
        self.snapshots = 0

        def snapshot(model):
            self.snapshots += 1
            if model != REQUEST["model"]:
                raise support.auth.HTTPException(
                    status_code=404, detail="Control model unknown"
                )
            return deepcopy(self.config)

        app = FastAPI()
        app.include_router(
            create_control_router(
                lambda: self.state,
                support.auth.verify_any_api_key,
                support.is_loopback_bind,
                snapshot,
            )
        )
        self.client = TestClient(app)
        self.addCleanup(self.client.close)
        self.headers = {"Authorization": "Bearer " + support.KEY}

    def put(self, request=None, path=support.PATH):
        return self.client.put(
            path,
            json={"request": REQUEST if request is None else request},
            headers=self.headers,
        )

    def test_registration_is_durable_idempotent_and_has_no_execution_authority(self):
        first = self.put()
        self.assertEqual(first.status_code, 200)
        self.assertEqual(self.put().json(), first.json())
        self.assertEqual(
            self.client.get(support.PATH, headers=self.headers).json(), first.json()
        )
        for field in [
            "backend_stopped",
            "engine_generation_verified",
            "provider_qualified",
            "production_enabled",
        ]:
            self.assertIs(first.json()[field], False)
        stored = self.state.prism_control_store.path.read_bytes()
        self.assertNotIn(REQUEST["input"].encode(), stored)
        self.assertNotIn(support.KEY, first.text)

    def test_reordered_request_fields_and_explicit_defaults_keep_the_same_digest(self):
        first = self.put().json()
        equivalent = dict(reversed(list(REQUEST.items()))) | {"temperature": None}
        self.assertEqual(self.put(equivalent).json(), first)

    def test_changes_to_input_reasoning_or_sampling_cannot_rebind_an_identity(self):
        first = self.put().json()
        for change in [
            {"input": "other-input"},
            {"temperature": 0.4},
            {"reasoning": {"effort": "high"}},
            {"max_output_tokens": 64},
            {"instructions": "different-instruction"},
        ]:
            with self.subTest(change=change):
                self.assertEqual(self.put(REQUEST | change).status_code, 409)
        self.assertEqual(
            self.client.get(support.PATH, headers=self.headers).json(), first
        )

    def test_server_settings_changes_cannot_silently_rebind_same_input(self):
        first = self.put().json()
        self.config["model_settings"]["thinking_budget"] = 1024
        self.assertEqual(self.put().status_code, 409)
        self.assertEqual(
            self.client.get(support.PATH, headers=self.headers).json(), first
        )
        new_path = support.PATH.replace(support.OP, str(uuid.uuid4()))
        second = self.put(path=new_path).json()
        self.assertEqual(second["request_sha256"], first["request_sha256"])
        self.assertNotEqual(second["settings_sha256"], first["settings_sha256"])

    def test_caller_hash_or_authority_fields_cannot_replace_server_computation(self):
        result = self.client.put(
            support.PATH,
            json={"request": REQUEST, "settings_sha256": "a" * 64},
            headers=self.headers,
        )
        self.assertEqual(result.status_code, 400)
        result = self.put(
            REQUEST | {"provider_qualified": True, "backend_stopped": True}
        )
        self.assertEqual(result.status_code, 200)
        self.assertFalse(result.json()["provider_qualified"])
        self.assertFalse(result.json()["backend_stopped"])

    def test_output_cap_and_server_memory_references_are_rejected_before_snapshot(self):
        for change in [
            {"max_output_tokens": None},
            {"max_output_tokens": True},
            {"max_output_tokens": 0},
            {"max_output_tokens": 8193},
            {"previous_response_id": "resp_old"},
            {"conversation": "old-memory"},
            {"background": True},
        ]:
            with self.subTest(change=change):
                self.assertEqual(self.put(REQUEST | change).status_code, 400)
        self.assertEqual(self.snapshots, 0)

    def test_malformed_nonfinite_deep_or_oversized_json_never_registers(self):
        bodies = [
            b"{",
            b'{"request":{"temperature":NaN}}',
            b"[" * 1200 + b"0" + b"]" * 1200,
            b"x" * (256 * 1024 + 1),
        ]
        for body in bodies:
            result = self.client.put(support.PATH, content=body, headers=self.headers)
            self.assertIn(result.status_code, [400, 413])
        self.assertEqual(self.snapshots, 0)
        self.assertIsNone(
            self.state.prism_control_store.lookup(
                support.OWNER, support.JOB, support.OP
            )
        )

    def test_authentication_and_invalid_identity_precede_body_and_snapshot(self):
        self.assertEqual(
            self.client.put(support.PATH, json={"request": REQUEST}).status_code, 401
        )
        invalid = support.PATH.replace(support.JOB, "invalid-job")
        self.assertEqual(self.put(path=invalid).status_code, 400)
        self.assertEqual(self.snapshots, 0)

    def test_unknown_model_or_unavailable_snapshot_never_creates_a_record(self):
        self.assertEqual(
            self.put(REQUEST | {"model": "unknown-model"}).status_code, 404
        )
        self.config["bad_float"] = float("nan")
        self.assertEqual(self.put().status_code, 503)
        self.assertIsNone(
            self.state.prism_control_store.lookup(
                support.OWNER, support.JOB, support.OP
            )
        )

    def test_registration_does_not_create_execution_routes(self):
        self.assertEqual(self.put().status_code, 200)
        self.assertIn(
            self.client.post(support.PATH, headers=self.headers).status_code, [404, 405]
        )
        # Slice 27.2 adds the execute route; without an execution coordinator
        # it refuses before reading the body and writes nothing.
        result = self.client.post(support.PATH + "/execute", headers=self.headers)
        self.assertEqual(
            (result.status_code, result.json()),
            (503, {"detail": "Control execution unavailable"}),
        )
        record = self.state.prism_control_store.lookup_status(
            support.OWNER, support.JOB, support.OP
        )
        self.assertEqual((record.state, record.claim_state), ("registered", "none"))

    def test_blocking_body_receive_cannot_register_after_its_deadline(self):
        def body():
            time.sleep(0.05)
            yield json.dumps({"request": REQUEST}).encode()

        with mock.patch.object(routes, "BODY_TIMEOUT_SECONDS", 0.01):
            result = self.client.put(support.PATH, content=body(), headers=self.headers)
        self.assertEqual(result.status_code, 408)
        self.assertEqual(self.snapshots, 0)
        self.assertIsNone(
            self.state.prism_control_store.lookup(
                support.OWNER, support.JOB, support.OP
            )
        )

    def test_source_snapshot_callback_reads_cold_inventory_and_selected_configuration(
        self,
    ):
        tree = ast.parse((support.ROOT / "omlx/server.py").read_text())
        nodes = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "get_control_settings_snapshot"
            or isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "CONTROL_SETTINGS_SECTIONS"
                for target in node.targets
            )
        ]
        self.assertEqual(len(nodes), 2)

        @dataclass
        class Config:
            value: int

        sections = (
            "server",
            "memory",
            "cache",
            "scheduler",
            "sampling",
            "idle_timeout",
        )
        globals_config = SimpleNamespace(
            **{name: Config(index) for index, name in enumerate(sections)},
            auth=SimpleNamespace(api_key="must-not-enter-snapshot")
        )
        seen = []

        def inventory(model):
            seen.append(model)
            return object() if model == REQUEST["model"] else None

        state = SimpleNamespace(
            engine_pool=SimpleNamespace(get_entry=inventory),
            global_settings=globals_config,
            sampling=Config(12),
        )
        scope = {
            "_server_state": state,
            "asdict": asdict,
            "HTTPException": support.auth.HTTPException,
            "get_model_settings_for_request": lambda model: SimpleNamespace(
                to_dict=lambda: {"thinking_budget": 256}
            ),
        }
        exec(
            compile(
                ast.Module(body=nodes, type_ignores=[]),
                str(support.ROOT / "omlx/server.py"),
                "exec",
            ),
            scope,
        )
        snapshot = scope["get_control_settings_snapshot"](REQUEST["model"])
        self.assertEqual(seen, [REQUEST["model"]])
        self.assertEqual(set(snapshot["global_settings"]), set(sections))
        self.assertEqual(snapshot["model_settings"], {"thinking_budget": 256})
        self.assertNotIn("must-not-enter-snapshot", json.dumps(snapshot))
        with self.assertRaises(support.auth.HTTPException):
            scope["get_control_settings_snapshot"]("unknown-model")


if __name__ == "__main__":
    unittest.main(verbosity=2)
