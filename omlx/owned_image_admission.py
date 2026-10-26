# SPDX-License-Identifier: Apache-2.0
"""Strict owned image representation checks; authority comes from the caller."""

from .owned_images import validate_owned_images

_MESSAGE_ROLES = frozenset({"user", "assistant", "system", "developer"})
_TEXT_PARTS = frozenset({"input_text", "text", "output_text"})
_LANGUAGE_SLOTS = ("_position_ids", "_rope_deltas")
_ADAPTER_SLOTS = (
    "_batch_rope_deltas",
    "_qwen4_text_prefill_positions",
    "_qwen4_step_text_positions",
)
_UNSUPPLIED = object()


def owned_image_sources(request, *, plain_only=False):
    """Validated, ordered user image sources from the parsed request.

    Plain-only is the internal diagnostic subset. The default also lets the
    existing text stage carry non-message input when there are no images.
    """
    value = request.model_dump(mode="json", exclude_none=True).get("input")
    count = validate_owned_images(value)
    if isinstance(value, str):
        return ()
    if not count and not plain_only:
        return ()
    if not isinstance(value, list):
        raise ValueError("owned_image_message_required")
    sources = []
    for message in value:
        if (
            not isinstance(message, dict)
            or message.get("type") not in (None, "message")
            or message.get("role") not in _MESSAGE_ROLES
        ):
            raise ValueError("owned_image_message_required")
        content = message.get("content")
        if isinstance(content, str):
            continue
        if not isinstance(content, list):
            raise ValueError("owned_image_message_required")
        for part in content:
            if not isinstance(part, dict):
                raise ValueError("owned_image_part_invalid")
            if part.get("type") == "input_image":
                if message["role"] != "user":
                    raise ValueError("owned_image_user_required")
                sources.append(part["image_url"])
            elif part.get("type") not in _TEXT_PARTS:
                raise ValueError("owned_image_part_invalid")
    if len(sources) != count:
        raise ValueError("owned_image_count_mismatch")
    return tuple(sources)


def prepared_image_sources(messages):
    """Canonical forms emitted by Responses conversion, before any decoding."""
    sources = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                raise ValueError("owned_image_part_invalid")
            kind = part.get("type")
            if kind in ("input_image", "image_url"):
                source = part.get("image_url")
                if isinstance(source, dict) and set(source) <= {"url", "detail"}:
                    source = source.get("url")
                if message.get("role") != "user" or not isinstance(source, str):
                    raise ValueError("owned_image_part_invalid")
                sources.append(source)
            elif kind not in _TEXT_PARTS:
                raise ValueError("owned_image_part_invalid")
    return tuple(sources)


def require_owned_image_quiescence(admission, core, *, language_model=_UNSUPPLIED):
    """Observe only, on the retained executor; no exclusive lease is claimed."""
    admission.validate(core, admission.binding.request_id, deadline=True)
    scheduler = core.scheduler
    model = scheduler.model
    lm = model._language_model
    if lm is None or (language_model is not _UNSUPPLIED and language_model is not lm):
        raise ValueError("owned_image_language_model_mismatch")
    if (
        core._output_collectors
        or scheduler._step_failed
        or scheduler._owned_model_refusal() is not None
        or scheduler._owned_vision_refusal(
            None, admission.binding.request_id, retired=True
        )
        is not None
    ):
        raise ValueError("owned_image_engine_not_quiescent")
    return (
        lm,
        *(getattr(lm, name) for name in _LANGUAGE_SLOTS),
        *(getattr(model, name) for name in _ADAPTER_SLOTS),
    )
