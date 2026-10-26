import unittest

from omlx.owned_image_policy import (
    OWNED_IMAGE_SCHEMA,
    OwnedImagePolicy,
    capture_image_policy,
    image_input_capability,
)


class OwnedImagePolicyTests(unittest.TestCase):
    def test_capability_is_strict_and_does_not_accept_truthy_values(self):
        capability = image_input_capability("vision", enabled=True, context_tokens=4096)
        self.assertTrue(capability["enabled"])
        self.assertEqual(capability["schema"], OWNED_IMAGE_SCHEMA)
        self.assertFalse(
            image_input_capability("vision", enabled=1, context_tokens=4096)["enabled"]
        )
        self.assertFalse(
            image_input_capability("vision", enabled=True, context_tokens=True)[
                "enabled"
            ]
        )
        self.assertFalse(
            image_input_capability("vision", enabled=True, context_tokens=0)["enabled"]
        )

    def test_capture_ignores_request_forged_flags_and_binds_digest(self):
        current = {"vision": "digest-a"}
        policy = capture_image_policy(
            "vision",
            {
                "owned_image_input": {
                    "schema": OWNED_IMAGE_SCHEMA,
                    "model_id": "vision",
                    "enabled": True,
                }
            },
            "digest-a",
            lambda model: current[model],
        )
        self.assertTrue(policy.permits("vision", "digest-a"))
        self.assertFalse(policy.permits("vision", "digest-b"))
        current["vision"] = "digest-b"
        self.assertFalse(policy.permits("vision", "digest-a"))
        forged = capture_image_policy(
            "vision",
            {"allow_images": True, "owned_image_input": {"enabled": True}},
            "digest-a",
            lambda _: "digest-a",
        )
        self.assertFalse(forged.permits("vision", "digest-a"))

    def test_digest_failure_fails_closed(self):
        policy = OwnedImagePolicy(
            "vision", "digest", True, lambda _: (_ for _ in ()).throw(RuntimeError())
        )
        self.assertFalse(policy.permits("vision", "digest"))


if __name__ == "__main__":
    unittest.main()
