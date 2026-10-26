# SPDX-License-Identifier: Apache-2.0
"""Owned VLM worker boundary; no model, stream or stop proof is fabricated."""

from .owned_admission import OwnedAdmission
from .owned_image_cache import without_image_cache
from .owned_image_admission import require_owned_image_quiescence

_POSITION_SLOTS = ("_position_ids", "_rope_deltas")
_ABSENT = object()


def prepare_owned_vision(
    admission,
    core,
    prepare,
    *args,
    dispatch_executor,
    stream_context,
    synchronize,
    language_model=None,
):
    """Synchronize the captured stream even when its preparation raises.

    Called only on the retained engine executor. An unsuccessful boundary is
    left pending on the handle so the stage retains its lease and reports no
    terminal outcome. Device behavior still requires separate GPU acceptance.
    """
    if type(admission) is not OwnedAdmission:
        raise ValueError("owned_admission_handle_required")
    if admission._image_admission:
        # Same executor job as prepare(): queued scheduler work cannot slip
        # between this observation and the image encoder.
        require_owned_image_quiescence(admission, core, language_model=language_model)
    admission.begin_preparation(core, dispatch_executor)
    stream = admission.binding.stream
    failure = None
    # run_in_executor does not inherit task ContextVars: establish the policy
    # inside the retained worker, including its stream cleanup boundary.
    with without_image_cache(), stream_context(stream):
        # These bindings belong to earlier work on the shared model. The
        # returned per-request kwargs retain the newly captured positions.
        # Preserve references only, not mutations to their object contents.
        previous = (
            [(name, getattr(language_model, name, _ABSENT)) for name in _POSITION_SLOTS]
            if language_model is not None
            else []
        )
        try:
            result = prepare(*args)
        except BaseException as error:
            failure = error
        synchronize(stream)
        # A failed wait must not restore state or certify preparation ended.
        # A failed restoration likewise leaves the boundary unresolved.
        for name, value in previous:
            if value is _ABSENT:
                if hasattr(language_model, name):
                    delattr(language_model, name)
            else:
                setattr(language_model, name, value)
    admission.finish_preparation(core)
    if failure is not None:
        raise failure
    return result
