# SPDX-License-Identifier: Apache-2.0
"""Owned work must not borrow the adapter's unrelated pending vision payload."""

from types import SimpleNamespace

import pytest

from omlx.models.vlm import VLMModelAdapter
from omlx.scheduler import Scheduler


def make_adapter():
    return VLMModelAdapter(
        SimpleNamespace(
            language_model=SimpleNamespace(),
            config=SimpleNamespace(model_type="synthetic", text_config=None),
        )
    )


@pytest.mark.parametrize("payload", [object(), False, 0, [], {}])
def test_pending_payload_refuses_without_retiring_shared_state(payload):
    adapter = make_adapter()
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.model = adapter
    positions = object()
    extra = {"position_ids": positions}
    adapter.set_pending_embeddings(payload, extra, start_offset=3)
    adapter._uid_rope_deltas[7] = 2.5

    assert scheduler._owned_model_refusal() == "pending_embeddings"
    assert adapter._pending_embeds is payload
    assert adapter._pending_kwargs is extra
    assert adapter._pending_kwargs["position_ids"] is positions
    assert adapter._embed_offset == 3
    assert adapter._uid_rope_deltas == {7: 2.5}


@pytest.mark.parametrize("model", ["empty_adapter", "text_model"])
def test_no_pending_payload_keeps_existing_model_support(model):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.model = make_adapter() if model == "empty_adapter" else SimpleNamespace()
    assert scheduler._owned_model_refusal() is None


def test_earlier_model_refusal_keeps_its_priority():
    adapter = make_adapter()
    adapter.set_pending_embeddings(object())
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.model = adapter
    scheduler._vlm_mtp_drafter = object()
    assert scheduler._owned_model_refusal() == "vlm_mtp"


@pytest.mark.parametrize("failure", [RuntimeError, AttributeError])
def test_pending_observation_failure_is_not_converted_to_clear_state(failure):
    class Unobservable:
        @property
        def has_pending_embeddings(self):
            raise failure("synthetic unavailable pending state")

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.model = Unobservable()
    with pytest.raises(failure, match="unavailable pending state"):
        scheduler._owned_model_refusal()


def test_missing_adapter_storage_is_not_a_clear_observation():
    adapter = VLMModelAdapter.__new__(VLMModelAdapter)
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.model = adapter
    with pytest.raises(AttributeError):
        scheduler._owned_model_refusal()
