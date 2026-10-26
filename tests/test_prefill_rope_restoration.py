# SPDX-License-Identifier: Apache-2.0
"""Cached VLM prefill restores saved None and the original model owner."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import pytest

import omlx.scheduler as scheduler_module
from omlx.request import Request, SamplingParams
from omlx.scheduler import Scheduler, SchedulerConfig, _PrefillAbortedError

assert Path(scheduler_module.__file__).resolve() == (
    Path(__file__).resolve().parents[1] / "omlx/scheduler.py"
)


@pytest.mark.parametrize("end", ["complete", "abort", "retry"])
@pytest.mark.parametrize("previous_is_none", [True, False])
@pytest.mark.parametrize("replace_owner", [False, True])
@pytest.mark.parametrize("cached", [True, False])
def test_prefill_preserves_saved_delta_and_owner(
    mock_model, mock_tokenizer, end, previous_is_none, replace_owner, cached
):
    previous = None if previous_is_none else object()
    changed, replacement_delta = object(), object()
    original_owner = SimpleNamespace(_rope_deltas=previous)
    replacement_owner = SimpleNamespace(_rope_deltas=replacement_delta)
    calls = []
    failure = RuntimeError("synthetic forward failure for retry")

    class ChangesDelta(type(mock_model)):
        def __call__(self, input_ids, **kwargs):
            calls.append(kwargs)
            # Only a cached VLM prefix temporarily clears its prior delta.
            assert original_owner._rope_deltas is (None if cached else previous)
            original_owner._rope_deltas = changed
            if replace_owner:
                self._language_model = replacement_owner
            if end == "retry":
                raise failure
            return super().__call__(input_ids, **kwargs)

    model = ChangesDelta()
    model._language_model = original_owner
    scheduler = Scheduler(
        model=model,
        tokenizer=mock_tokenizer,
        config=SchedulerConfig(prefill_step_size=8),
    )
    request = Request(
        request_id="synthetic-cached-rope",
        prompt=[1, 2, 3],
        sampling_params=SamplingParams(),
    )
    request.prompt_token_ids = [1, 2, 3]
    request.num_prompt_tokens = 3
    embeds = mx.zeros((1, 4, 2))
    extra = {"position_ids": mx.zeros((3, 1, 4), dtype=mx.int32)}
    request.vlm_inputs_embeds = embeds
    request.vlm_extra_kwargs = extra
    uid = 42
    scheduler.request_id_to_uid[request.request_id] = uid
    scheduler.uid_to_request_id[uid] = request.request_id
    if end == "abort":
        scheduler._pending_abort_ids.add(request.request_id)
    cache = [SimpleNamespace(state=mx.array([0]), offset=0)]

    with patch.object(scheduler_module, "_sync_and_clear_cache"):
        if end == "retry":
            with pytest.raises(RuntimeError) as caught:
                scheduler._do_external_prefill(
                    request,
                    request.prompt_token_ids,
                    cache,
                    (embeds, extra, 1 if cached else 0),
                )
            assert caught.value is failure
            scheduler._requeue_prefill_retry(request)
            assert scheduler.waiting[0] is request
            assert request.vlm_inputs_embeds is embeds
            assert request.vlm_extra_kwargs is extra
        elif end == "abort":
            with pytest.raises(_PrefillAbortedError):
                scheduler._do_external_prefill(
                    request,
                    request.prompt_token_ids,
                    cache,
                    (embeds, extra, 1 if cached else 0),
                )
        else:
            returned_cache, last = scheduler._do_external_prefill(
                request,
                request.prompt_token_ids,
                cache,
                (embeds, extra, 1 if cached else 0),
            )
            assert returned_cache is cache
            assert last == [3]

    assert len(calls) == 1
    assert original_owner._rope_deltas is (previous if cached else changed)
    assert replacement_owner._rope_deltas is replacement_delta
    assert request._prefill_saved_rope_deltas is None


@pytest.mark.parametrize("pause_after_chunks", [0, 1])
@pytest.mark.parametrize("previous_is_none", [True, False])
@pytest.mark.parametrize("abort_after_resume", [False, True])
def test_eviction_pause_restores_before_preserving_progress_and_resuming(
    mock_model, mock_tokenizer, pause_after_chunks, previous_is_none, abort_after_resume
):
    previous = None if previous_is_none else object()
    owner = SimpleNamespace(_rope_deltas=previous)
    forwards = []

    class ChangesDelta(type(mock_model)):
        def __call__(self, input_ids, **kwargs):
            forwards.append(input_ids.shape[1])
            owner._rope_deltas = object()
            return super().__call__(input_ids, **kwargs)

    model = ChangesDelta()
    model._language_model = owner
    scheduler = Scheduler(
        model=model,
        tokenizer=mock_tokenizer,
        config=SchedulerConfig(prefill_step_size=1),
    )
    request = Request(
        request_id="synthetic-eviction-rope",
        prompt=[1, 2, 3, 4],
        sampling_params=SamplingParams(),
    )
    request.prompt_token_ids = [1, 2, 3, 4]
    request.num_prompt_tokens = 4
    request.cached_tokens = 1
    embeds = mx.zeros((1, 5, 2))
    extra = {"position_ids": mx.zeros((3, 1, 5), dtype=mx.int32)}
    cache = [SimpleNamespace(state=mx.array([0]), offset=0)]
    uid = 43
    scheduler.request_id_to_uid[request.request_id] = uid
    scheduler.uid_to_request_id[uid] = request.request_id
    route = Scheduler._guard_prefill_chunk_route
    paused = False

    def pause_once(instance, *args, **kwargs):
        nonlocal paused
        if not paused and len(forwards) == pause_after_chunks:
            paused = True
            raise scheduler_module._PrefillEvictionNeeded(
                SimpleNamespace(reason="synthetic eviction pause")
            )
        return route(instance, *args, **kwargs)

    with (
        patch.object(scheduler_module, "_sync_and_clear_cache"),
        patch.object(
            Scheduler,
            "_guard_prefill_chunk_route",
            side_effect=pause_once,
            autospec=True,
        ),
    ):
        with pytest.raises(scheduler_module._PrefillEvictionNeeded):
            scheduler._do_external_prefill(
                request, request.prompt_token_ids, cache, (embeds, extra, 1)
            )
        assert owner._rope_deltas is previous
        assert request._prefill_saved_rope_deltas is None
        assert request.cached_tokens == 1 + pause_after_chunks
        assert request.prefill_resumed_tokens == pause_after_chunks
        assert request._prefill_resumed is True
        if abort_after_resume:
            scheduler._pending_abort_ids.add(request.request_id)
        remaining = request.prompt_token_ids[pause_after_chunks:]
        if abort_after_resume:
            with pytest.raises(_PrefillAbortedError):
                scheduler._do_external_prefill(
                    request,
                    remaining,
                    cache,
                    (embeds, extra, 1 + pause_after_chunks),
                )
        else:
            _, last = scheduler._do_external_prefill(
                request,
                remaining,
                cache,
                (embeds, extra, 1 + pause_after_chunks),
            )
            assert last == [4]
    assert owner._rope_deltas is previous
    assert request._prefill_saved_rope_deltas is None
