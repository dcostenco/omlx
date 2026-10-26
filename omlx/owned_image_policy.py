# SPDX-License-Identifier: Apache-2.0
"""Server-captured authority for owned image input; never a request field."""

from dataclasses import dataclass, field
from typing import Callable

from .owned_images import (
    MAX_IMAGES,
    MAX_IMAGE_BYTES,
    MAX_TOTAL_IMAGE_BYTES,
    MAX_IMAGE_SIDE,
    MAX_IMAGE_PIXELS,
    MAX_TOTAL_IMAGE_PIXELS,
)

OWNED_IMAGE_SCHEMA = "owned-png-v1"


def image_input_capability(model_id, *, enabled, context_tokens):
    return {
        "schema": OWNED_IMAGE_SCHEMA,
        "model_id": model_id,
        "enabled": enabled is True
        and type(context_tokens) is int
        and context_tokens > 0,
        "context_tokens": context_tokens,
        "limits": {
            "images": MAX_IMAGES,
            "image_bytes": MAX_IMAGE_BYTES,
            "total_image_bytes": MAX_TOTAL_IMAGE_BYTES,
            "image_side": MAX_IMAGE_SIDE,
            "image_pixels": MAX_IMAGE_PIXELS,
            "total_image_pixels": MAX_TOTAL_IMAGE_PIXELS,
        },
        "evidence_scope": "server_input_capability_not_model_quality",
    }


@dataclass(frozen=True)
class OwnedImagePolicy:
    model_id: str
    settings_sha256: str
    enabled: bool
    _current_digest: Callable[[str], str] = field(repr=False, compare=False)

    def permits(self, model_id, settings_sha256):
        if (
            self.enabled is not True
            or self.model_id != model_id
            or self.settings_sha256 != settings_sha256
        ):
            return False
        try:
            return self._current_digest(model_id) == self.settings_sha256
        except Exception:
            return False


def capture_image_policy(model_id, snapshot, settings_sha256, current_digest):
    capability = snapshot.get("owned_image_input", {})
    enabled = (
        isinstance(capability, dict)
        and capability.get("schema") == OWNED_IMAGE_SCHEMA
        and capability.get("model_id") == model_id
        and capability.get("enabled") is True
    )
    return OwnedImagePolicy(model_id, settings_sha256, enabled, current_digest)
