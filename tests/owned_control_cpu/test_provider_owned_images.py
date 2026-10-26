# SPDX-License-Identifier: Apache-2.0
"""Owned image input limits, without MLX, a model, or installed state."""

import base64
import asyncio
from copy import deepcopy
from io import BytesIO
import struct
import unittest
from unittest import mock
import warnings
import zlib

import test_provider_control_registration as registration
import test_provider_control_execute as execute
from PIL import Image, ImageFile
from omlx import owned_images as images
from omlx.api.prism_control_routes import canonical_digest, parse_control_request
from omlx.owned_preparation import owned_request_refusal


def encoded_image(fmt="PNG", size=(8, 8), **options):
    with BytesIO() as output:
        with Image.new("RGB", size, "green") as picture:
            picture.save(output, format=fmt, **options)
        return output.getvalue()


def uri(data, mime="png"):
    return "data:image/" + mime + ";base64," + base64.b64encode(data).decode("ascii")


def request_with(*values):
    return registration.REQUEST | {
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "compare in order"},
                    *[{"type": "input_image", "image_url": value} for value in values],
                ],
            }
        ]
    }


def malformed_pngs():
    def chunk(kind, payload):
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload))
        )

    output = BytesIO()
    with Image.new("L", (1, 1)) as picture:
        picture.save(output, format="PNG")
    png = output.getvalue()
    header = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0))
    return [
        png[:8] + header + chunk(b"IEND", b""),
        png[:-12] + chunk(b"tRNS", b"") + png[-12:],
    ]


class ImageValidationTests(unittest.TestCase):
    def test_malformed_png_codec_errors_are_sanitized(self):
        for data in malformed_pngs():
            with self.subTest(length=len(data)):
                with self.assertRaisesRegex(ValueError, "^Invalid owned image input$"):
                    parse_control_request(request_with(uri(data)))

    def test_common_noncanonical_image_aliases_are_rejected(self):
        parts = [
            {"type": "image", "image": "https://example.test/x.png"},
            {"type": "image", "url": "file:///private.png"},
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"},
            },
            {"type": "image_file", "image_file": {"file_id": "synthetic"}},
            {"image": "https://example.test/x.png"},
        ]
        for part in parts:
            for field in ["content", "output"]:
                raw = registration.REQUEST | {
                    "input": [{"role": "user", field: [part]}]
                }
                with self.subTest(field=field, part=part):
                    with self.assertRaisesRegex(
                        ValueError, "^Invalid owned image input$"
                    ):
                        parse_control_request(raw)

    def test_text_and_valid_images_keep_exact_input_order_and_digest(self):
        for raw in [
            registration.REQUEST,
            request_with(uri(encoded_image()), uri(encoded_image(size=(9, 9)))),
        ]:
            before = deepcopy(raw)
            parsed, digest = parse_control_request(raw)
            self.assertEqual(raw, before)
            self.assertEqual(
                parsed.model_dump(mode="json")["input"],
                registration.routes.ResponsesRequest.model_validate(before).model_dump(
                    mode="json"
                )["input"],
            )
            self.assertEqual(digest, canonical_digest(parsed.model_dump(mode="json")))
        self.assertEqual(owned_request_refusal(parsed), "media_input")

    def test_bad_references_mime_and_base64_are_rejected_without_payload_errors(self):
        valid = uri(encoded_image())
        for value in [
            "https://example.test/private.png",
            "file:///private.png",
            "/private.png",
            "data:image/png;base64,!!!!",
            valid + " ",
            valid.replace("image/png", "image/jpeg"),
            uri(encoded_image("GIF"), "gif"),
            {"url": valid},
            None,
        ]:
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaises(ValueError) as caught:
                    parse_control_request(request_with(value))
                self.assertEqual(str(caught.exception), "Invalid owned image input")

    def test_image_aliases_cannot_bypass_validation_in_nested_input_or_tool_output(
        self,
    ):
        for part in [
            {"type": "image_url", "image_url": {"url": "https://example.test/x"}},
            {"image_url": "https://example.test/x"},
            {"type": "input_image", "file_id": "file_remote"},
        ]:
            for raw in [
                registration.REQUEST | {"input": [{"role": "user", "content": [part]}]},
                registration.REQUEST
                | {
                    "input": [
                        {
                            "type": "function_call_output",
                            "output": [part],
                            "call_id": "synthetic",
                        }
                    ]
                },
            ]:
                with self.subTest(part=part):
                    with self.assertRaises(ValueError):
                        parse_control_request(raw)

    def test_count_and_encoded_byte_limits_precede_image_open(self):
        valid = uri(encoded_image())
        too_large = uri(b"x" * (images.MAX_IMAGE_BYTES + 1))
        with mock.patch.object(
            images.Image, "open", side_effect=AssertionError("decoded")
        ):
            for raw in [
                request_with(*([valid] * (images.MAX_IMAGES + 1))),
                request_with(too_large),
            ]:
                with self.assertRaises(ValueError):
                    parse_control_request(raw)
            with mock.patch.object(images, "MAX_TOTAL_IMAGE_BYTES", 1):
                with self.assertRaises(ValueError):
                    parse_control_request(request_with(valid))

    def test_dimensions_pixels_and_total_pixels_precede_pixel_load(self):
        valid = uri(encoded_image(size=(16, 16)))
        with mock.patch.object(
            images.Image.Image, "load", side_effect=AssertionError("pixels decoded")
        ):
            for constant, limit in [
                ("MAX_IMAGE_SIDE", 8),
                ("MAX_IMAGE_PIXELS", 128),
                ("MAX_TOTAL_IMAGE_PIXELS", 300),
            ]:
                with (
                    self.subTest(constant=constant),
                    mock.patch.object(images, constant, limit),
                ):
                    with self.assertRaises(ValueError):
                        parse_control_request(request_with(valid, valid))

    def test_header_bomb_is_rejected_without_pixel_load(self):
        data = bytearray(encoded_image())
        data[16:24] = struct.pack(">II", 60000, 60000)
        data[29:33] = struct.pack(">I", zlib.crc32(data[12:29]))
        with mock.patch.object(
            images.Image.Image, "load", side_effect=AssertionError("pixels decoded")
        ):
            with self.assertRaises(ValueError):
                parse_control_request(request_with(uri(data)))

    def test_header_bomb_warning_as_error_is_still_sanitized(self):
        data = bytearray(encoded_image())
        data[16:24] = struct.pack(">II", 10000, 10000)
        data[29:33] = struct.pack(">I", zlib.crc32(data[12:29]))
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with self.assertRaisesRegex(ValueError, "^Invalid owned image input$"):
                parse_control_request(request_with(uri(data)))

    def test_corruption_truncation_and_animation_are_rejected(self):
        png, jpeg = encoded_image(), encoded_image("JPEG")
        output = BytesIO()
        with (
            Image.new("RGB", (8, 8), "red") as a,
            Image.new("RGB", (8, 8), "blue") as b,
        ):
            a.save(output, format="PNG", save_all=True, append_images=[b], duration=100)
        for data, mime in [
            (png[:-12], "png"),
            (png[:45] + b"broken" + png[51:], "png"),
            (jpeg[:-2], "jpeg"),
            (output.getvalue(), "png"),
        ]:
            with self.subTest(mime=mime, length=len(data)):
                with self.assertRaises(ValueError):
                    parse_control_request(request_with(uri(data, mime)))

    def test_validated_images_do_not_use_generic_decode_cache(self):
        with mock.patch.dict("sys.modules", {"omlx.utils.image": None}):
            parse_control_request(request_with(uri(encoded_image())))

    def test_jpeg_is_refused_until_a_strict_decoder_is_qualified(self):
        with mock.patch.object(
            images.Image, "open", side_effect=AssertionError("opened")
        ):
            with self.assertRaises(ValueError):
                parse_control_request(request_with(uri(encoded_image("JPEG"), "jpeg")))

    def test_permissive_pillow_mode_refuses_images_without_changing_global_settings(
        self,
    ):
        value = uri(encoded_image())
        with mock.patch.object(ImageFile, "LOAD_TRUNCATED_IMAGES", True):
            with mock.patch.object(
                images.Image, "open", side_effect=AssertionError("opened")
            ):
                with self.assertRaises(ValueError):
                    parse_control_request(request_with(value))
            self.assertIs(ImageFile.LOAD_TRUNCATED_IMAGES, True)
            parse_control_request(registration.REQUEST)

    def test_untyped_dict_tool_output_image_url_remains_ordinary_text(self):
        raw = registration.REQUEST | {
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "synthetic",
                    "output": {
                        "image_url": "https://example.test/x.png",
                        "status": "ok",
                    },
                }
            ]
        }
        expected = registration.routes.ResponsesRequest.model_validate(raw).model_dump(
            mode="json"
        )
        parsed, digest = parse_control_request(raw)
        self.assertEqual(parsed.model_dump(mode="json"), expected)
        self.assertEqual(digest, canonical_digest(expected))
        self.assertIsNone(owned_request_refusal(parsed))

    def test_images_in_dict_tool_outputs_cannot_be_serialized_past_media_refusal(self):
        part = {"type": "input_image", "image_url": uri(encoded_image())}
        raw = registration.REQUEST | {
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "synthetic",
                    "output": {"attachment": part},
                }
            ]
        }
        with self.assertRaises(ValueError):
            parse_control_request(raw)

    def test_canonical_tool_output_list_preserves_image_for_media_refusal(self):
        part = {"type": "input_image", "image_url": uri(encoded_image())}
        raw = registration.REQUEST | {
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "synthetic",
                    "output": [part],
                }
            ]
        }
        parsed, _ = parse_control_request(raw)
        self.assertEqual(parsed.input[0].output, [part])
        self.assertEqual(owned_request_refusal(parsed), "media_input")


class ImageRegistrationTests(unittest.TestCase):
    setUp = registration.RegistrationTests.setUp
    put = registration.RegistrationTests.put

    def test_malformed_png_codec_errors_never_register_or_reach_settings(self):
        for data in malformed_pngs():
            result = self.put(request_with(uri(data)))
            self.assertEqual(
                (result.status_code, result.json()),
                (400, {"detail": "Invalid control request"}),
            )
        self.assertEqual(self.snapshots, 0)
        self.assertIsNone(
            self.state.prism_control_store.lookup(
                registration.support.OWNER,
                registration.support.JOB,
                registration.support.OP,
            )
        )

    def test_invalid_image_never_reaches_settings_or_registration(self):
        result = self.put(request_with("https://example.test/private.png"))
        self.assertEqual(
            (result.status_code, result.json()),
            (400, {"detail": "Invalid control request"}),
        )
        self.assertEqual(self.snapshots, 0)
        self.assertIsNone(
            self.state.prism_control_store.lookup(
                registration.support.OWNER,
                registration.support.JOB,
                registration.support.OP,
            )
        )

    def test_valid_image_registration_is_digest_only_and_still_not_inference(self):
        value = uri(encoded_image())
        result = self.put(request_with(value))
        self.assertEqual(result.status_code, 200)
        self.assertNotIn(value, result.text)
        self.assertNotIn(
            value.encode(), self.state.prism_control_store.path.read_bytes()
        )
        self.assertFalse(result.json()["provider_qualified"])
        self.assertFalse(result.json()["backend_stopped"])
        self.assertEqual(
            owned_request_refusal(parse_control_request(request_with(value))[0]),
            "media_input",
        )


class ImageExecuteTests(unittest.TestCase):
    make = execute.ExecuteTests.make
    register = execute.ExecuteTests.register

    def test_invalid_execution_image_cannot_claim_or_start_a_registered_operation(self):
        async def scenario():
            harness = self.make()
            before = await self.register(harness)
            harness.coordinator.start = mock.Mock(side_effect=AssertionError("started"))
            # Rejection must happen before the coordinator, which owns claiming.
            requests = [
                request_with("https://example.test/private.png"),
                *[request_with(uri(data)) for data in malformed_pngs()],
                registration.REQUEST
                | {
                    "input": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "url": "file:///private.png"}
                            ],
                        }
                    ]
                },
            ]
            for raw in requests:
                result = await execute.asgi(
                    harness.app,
                    "POST",
                    execute.path(suffix="/execute"),
                    execute.execute_body(raw),
                )
                self.assertEqual(
                    (result.status, result.json()),
                    (400, {"detail": "Invalid control request"}),
                )
            harness.coordinator.start.assert_not_called()
            self.assertNotIn("claim", harness.store.calls)
            after = await execute.asgi(harness.app, "GET", execute.path())
            self.assertEqual(after.json(), before)

        asyncio.run(asyncio.wait_for(scenario(), 5))
