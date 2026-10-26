import secrets
import unittest
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import test_provider_control_routes as support
from omlx.api.prism_control_routes import create_control_router

auth = support.auth
is_loopback_bind = support.is_loopback_bind


class ImageCapabilityRouteTests(unittest.TestCase):
    def setUp(self):
        self.key = secrets.token_urlsafe(24)
        self.model = "vision-model"
        self.snapshot = {
            "owned_image_input": {
                "schema": "owned-png-v1",
                "model_id": self.model,
                "enabled": True,
            }
        }
        self.calls = []
        state = SimpleNamespace(
            api_key=self.key,
            bind_host="127.0.0.1",
            global_settings=SimpleNamespace(auth=SimpleNamespace(sub_keys=[])),
        )
        app = FastAPI()
        app.include_router(
            create_control_router(
                lambda: state,
                auth.verify_any_api_key,
                is_loopback_bind,
                self.read_snapshot,
            )
        )
        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def read_snapshot(self, model):
        self.calls.append(model)
        if model != self.model:
            raise KeyError(model)
        return self.snapshot

    def get(self, model=None, key=None):
        return self.client.get(
            f"/prism-control/v1/models/{model or self.model}/image-input",
            headers={"Authorization": "Bearer " + (self.key if key is None else key)},
        )

    def test_auth_and_bind_guard_before_snapshot(self):
        self.assertEqual(self.get(key="bad").status_code, 401)
        self.assertEqual(len(self.calls), 0)

    def test_authenticated_capability_contains_exact_model_and_digest(self):
        response = self.get()
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["model_id"], self.model)
        self.assertEqual(body["enabled"], True)
        self.assertEqual(len(body["settings_sha256"]), 64)

    def test_disabled_and_settings_flip_change_authority_digest(self):
        first = self.get().json()
        self.snapshot["owned_image_input"]["enabled"] = False
        second = self.get().json()
        self.assertFalse(second["enabled"])
        self.assertNotEqual(first["settings_sha256"], second["settings_sha256"])
        forged = self.client.get(
            f"/prism-control/v1/models/{self.model}/image-input?enabled=true",
            headers={"Authorization": "Bearer " + self.key},
        ).json()
        self.assertFalse(forged["enabled"])

    def test_public_bind_refuses_before_snapshot(self):
        self.client.close()
        state = SimpleNamespace(
            api_key=self.key,
            bind_host="0.0.0.0",
            global_settings=SimpleNamespace(auth=SimpleNamespace(sub_keys=[])),
        )
        app = FastAPI()
        app.include_router(
            create_control_router(
                lambda: state,
                auth.verify_any_api_key,
                is_loopback_bind,
                self.read_snapshot,
            )
        )
        with TestClient(app) as client:
            self.assertEqual(
                client.get(
                    f"/prism-control/v1/models/{self.model}/image-input",
                    headers={"Authorization": "Bearer " + self.key},
                ).status_code,
                403,
            )
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
