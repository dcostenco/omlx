# SPDX-License-Identifier: Apache-2.0
"""Task/thread-local bypass for oMLX's image caches; never changes settings."""

from contextlib import contextmanager
from contextvars import ContextVar

_image_cache_allowed = ContextVar("omlx_image_cache_allowed", default=True)


def image_cache_allowed() -> bool:
    return _image_cache_allowed.get()


@contextmanager
def without_image_cache():
    token = _image_cache_allowed.set(False)
    try:
        yield
    finally:
        _image_cache_allowed.reset(token)
