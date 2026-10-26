# SPDX-License-Identifier: Apache-2.0
"""Strict inline image validation for owned control requests; no image cache."""

import base64
import binascii
from io import BytesIO

from PIL import Image, ImageFile

MAX_IMAGES = 4
MAX_IMAGE_BYTES = 128 * 1024
MAX_TOTAL_IMAGE_BYTES = 192 * 1024
MAX_IMAGE_SIDE = 2048
MAX_IMAGE_PIXELS = 4 * 1024 * 1024
MAX_TOTAL_IMAGE_PIXELS = 8 * 1024 * 1024
IMAGE_ERROR = "Invalid owned image input"
IMAGE_PREFIXES = {"data:image/png;base64": "PNG"}
IMAGE_FIELDS = frozenset({"type", "image_url", "detail"})
IMAGE_DETAILS = frozenset({"auto", "low", "high"})
IMAGE_TYPES = ("input_image", "image_url", "image", "image_file")
IMAGE_KEYS = frozenset({"image_url", "image", "image_file"})


def validate_owned_images(value):
    """Return the validated image count without changing input or admission.

    Count and raw byte budgets precede Pillow. All image headers pass the
    per-image and aggregate pixel budgets before any explicit pixel load.
    Only canonical Responses input_image parts are accepted by this helper.
    JPEG is refused until strict scan validation exists. Other media types
    remain subject to the separate owned refusal guard.
    """
    parts = []
    pending = [(value, ())]
    while pending:
        item, path = pending.pop()
        if isinstance(item, list):
            pending.extend(
                (child, (*path, i)) for i, child in reversed(list(enumerate(item)))
            )
        elif isinstance(item, dict):
            image_position = (
                len(path) == 3
                and type(path[0]) is int
                and path[1] in ("content", "output")
                and type(path[2]) is int
            )
            if item.get("type") in IMAGE_TYPES or (
                image_position and IMAGE_KEYS.intersection(item)
            ):
                if (
                    not image_position
                    or item.get("type") != "input_image"
                    or not {"type", "image_url"} <= item.keys() <= IMAGE_FIELDS
                    or not isinstance(item.get("image_url"), str)
                    or not isinstance(item.get("detail", "auto"), str)
                    or item.get("detail", "auto") not in IMAGE_DETAILS
                ):
                    raise ValueError(IMAGE_ERROR)
                parts.append(item["image_url"])
                if len(parts) > MAX_IMAGES:
                    raise ValueError(IMAGE_ERROR)
            else:
                pending.extend(
                    (child, (*path, key)) for key, child in reversed(list(item.items()))
                )

    # Do not change this process-wide flag; another request may depend on it.
    if parts and ImageFile.LOAD_TRUNCATED_IMAGES:
        raise ValueError(IMAGE_ERROR)
    payloads = []
    total_bytes = 0
    for source in parts:
        prefix, separator, encoded = source.partition(",")
        if (
            not separator
            or prefix not in IMAGE_PREFIXES
            or len(encoded) > 4 * ((MAX_IMAGE_BYTES + 2) // 3)
        ):
            raise ValueError(IMAGE_ERROR)
        try:
            data = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error):
            raise ValueError(IMAGE_ERROR) from None
        total_bytes += len(data)
        if (
            not data
            or len(data) > MAX_IMAGE_BYTES
            or total_bytes > MAX_TOTAL_IMAGE_BYTES
            or base64.b64encode(data).decode("ascii") != encoded
        ):
            raise ValueError(IMAGE_ERROR)
        payloads.append((data, IMAGE_PREFIXES[prefix]))

    try:
        total_pixels = 0
        for data, expected_format in payloads:
            with (
                BytesIO(data) as stream,
                Image.open(stream, formats=["PNG"]) as picture,
            ):
                width, height = picture.size
                total_pixels += width * height
                if (
                    picture.format != expected_format
                    or not 1 <= width <= MAX_IMAGE_SIDE
                    or not 1 <= height <= MAX_IMAGE_SIDE
                    or width * height > MAX_IMAGE_PIXELS
                    or total_pixels > MAX_TOTAL_IMAGE_PIXELS
                ):
                    raise ValueError(IMAGE_ERROR)
        for data, expected_format in payloads:
            with (
                BytesIO(data) as stream,
                Image.open(stream, formats=["PNG"]) as picture,
            ):
                if getattr(picture, "n_frames", 1) != 1:
                    raise ValueError(IMAGE_ERROR)
                picture.verify()
            with (
                BytesIO(data) as stream,
                Image.open(stream, formats=["PNG"]) as picture,
            ):
                picture.load()
    except Exception:
        # Pillow plugins raise more than OSError for malformed containers
        # (e.g. IndexError or struct.error). Keep codec failures in this
        # request-validation boundary; never expose the payload or traceback.
        raise ValueError(IMAGE_ERROR) from None
    return len(parts)
