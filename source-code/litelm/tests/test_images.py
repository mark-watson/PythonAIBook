"""Image generation: the native ``predict`` request and its decoded images."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import pytest

import litelm
from tests.conftest import FakeTransport

PNG = b"\x89PNG\r\n\x1a\nfake-png-bytes"
JPEG = b"\xff\xd8\xfffake-jpeg-bytes"

MODEL = "gemini/imagen-4.0-fast-generate-001"


def image_body(*images: tuple[bytes, str]) -> dict[str, Any]:
    return {
        "predictions": [
            {
                "bytesBase64Encoded": base64.b64encode(data).decode("ascii"),
                "mimeType": mime,
            }
            for data, mime in images
        ]
    }


class TestGenerateImageRequest:
    def test_the_request_reaches_the_native_predict_endpoint(
        self, fake_transport: FakeTransport
    ) -> None:
        fake_transport.json_results = [image_body((PNG, "image/png"))]
        litelm.generate_image(MODEL, "a serene landscape", api_key="g-key")
        request = fake_transport.last_request
        assert request["url"] == (
            "https://generativelanguage.googleapis.com/v1beta"
            "/models/imagen-4.0-fast-generate-001:predict"
        )
        assert request["headers"]["x-goog-api-key"] == "g-key"
        assert "Authorization" not in request["headers"]
        assert fake_transport.last_payload == {
            "instances": [{"prompt": "a serene landscape"}],
            "parameters": {"sampleCount": 1},
        }

    def test_count_and_aspect_ratio_land_in_parameters(
        self, fake_transport: FakeTransport
    ) -> None:
        fake_transport.json_results = [image_body((PNG, "image/png"))]
        litelm.generate_image(
            MODEL, "p", number_of_images=3, aspect_ratio="16:9", api_key="k"
        )
        assert fake_transport.last_payload["parameters"] == {
            "sampleCount": 3,
            "aspectRatio": "16:9",
        }

    def test_extra_is_merged_into_parameters_last(
        self, fake_transport: FakeTransport
    ) -> None:
        fake_transport.json_results = [image_body((PNG, "image/png"))]
        litelm.generate_image(
            MODEL,
            "p",
            api_key="k",
            extra={"sampleCount": 2, "negativePrompt": "blurry"},
        )
        parameters = fake_transport.last_payload["parameters"]
        assert parameters["sampleCount"] == 2
        assert parameters["negativePrompt"] == "blurry"

    def test_api_base_overrides_the_native_url(
        self, fake_transport: FakeTransport
    ) -> None:
        fake_transport.json_results = [image_body((PNG, "image/png"))]
        litelm.generate_image(MODEL, "p", api_key="k", api_base="http://box:9000/v1/")
        assert fake_transport.last_request["url"] == (
            "http://box:9000/v1/models/imagen-4.0-fast-generate-001:predict"
        )

    def test_a_provider_without_a_native_api_raises(self) -> None:
        with pytest.raises(litelm.LitelmError, match="no native API"):
            litelm.generate_image("openai/gpt-image-1", "p", api_key="k")


class TestGenerateImageParsing:
    def test_images_are_decoded_in_order(self, fake_transport: FakeTransport) -> None:
        fake_transport.json_results = [
            image_body((PNG, "image/png"), (JPEG, "image/jpeg"))
        ]
        images = litelm.generate_image(MODEL, "p", api_key="k")
        assert [image.data for image in images] == [PNG, JPEG]
        assert [image.mime_type for image in images] == ["image/png", "image/jpeg"]
        assert [image.suffix for image in images] == ["png", "jpg"]
        assert len(images[0]) == len(PNG)

    def test_the_nested_image_shape_is_accepted(
        self, fake_transport: FakeTransport
    ) -> None:
        fake_transport.json_results = [
            {
                "predictions": [
                    {
                        "image": {
                            "imageBytes": base64.b64encode(PNG).decode("ascii"),
                            "mimeType": "image/webp",
                        }
                    }
                ]
            }
        ]
        (image,) = litelm.generate_image(MODEL, "p", api_key="k")
        assert image.data == PNG
        assert image.suffix == "webp"

    def test_a_missing_mime_type_defaults_to_png(
        self, fake_transport: FakeTransport
    ) -> None:
        fake_transport.json_results = [
            {"predictions": [{"bytesBase64Encoded": base64.b64encode(PNG).decode()}]}
        ]
        (image,) = litelm.generate_image(MODEL, "p", api_key="k")
        assert image.mime_type == "image/png"

    def test_malformed_base64_raises(self, fake_transport: FakeTransport) -> None:
        fake_transport.json_results = [
            {
                "predictions": [
                    {"bytesBase64Encoded": "not base64!!", "mimeType": "image/png"}
                ]
            }
        ]
        with pytest.raises(litelm.LitelmError, match="malformed base64"):
            litelm.generate_image(MODEL, "p", api_key="k")

    def test_no_predictions_raises(self, fake_transport: FakeTransport) -> None:
        fake_transport.json_results = [{"predictions": []}]
        with pytest.raises(litelm.LitelmError, match="no predictions"):
            litelm.generate_image(MODEL, "p", api_key="k")

    def test_predictions_without_decodable_data_raise(
        self, fake_transport: FakeTransport
    ) -> None:
        fake_transport.json_results = [{"predictions": [{"mimeType": "image/png"}]}]
        with pytest.raises(litelm.LitelmError, match="no decodable images"):
            litelm.generate_image(MODEL, "p", api_key="k")

    def test_an_error_field_raises(self, fake_transport: FakeTransport) -> None:
        fake_transport.json_results = [{"error": {"message": "quota exceeded"}}]
        with pytest.raises(litelm.LitelmError, match="quota exceeded"):
            litelm.generate_image(MODEL, "p", api_key="k")

    def test_save_writes_the_bytes(self, tmp_path: Path) -> None:
        image = litelm.GeneratedImage(data=PNG, mime_type="image/png")
        target = image.save(tmp_path / f"out.{image.suffix}")
        assert target.read_bytes() == PNG
        assert target.name == "out.png"
