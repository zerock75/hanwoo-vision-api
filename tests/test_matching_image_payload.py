from __future__ import annotations

import base64
import io

from PIL import Image

from hanwoo.core.image_payload import encode_image_payload
from hanwoo.core.image_payload import attach_image_payload


def decode_payload_image(payload: dict) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(payload["image_base64"])))


def test_encode_image_payload_preserves_saved_rgba_png(tmp_path) -> None:
    image_path = tmp_path / "match.png"
    image = Image.new("RGBA", (2, 1))
    image.putpixel((0, 0), (255, 255, 255, 0))
    image.putpixel((1, 0), (32, 64, 96, 255))
    image.save(image_path)

    payload = encode_image_payload(image_path)
    decoded = decode_payload_image(payload)

    assert payload["image_mime_type"] == "image/png"
    assert payload["image_size_bytes"] > 0
    assert decoded.mode == "RGBA"
    assert decoded.getpixel((0, 0)) == (255, 255, 255, 0)
    assert decoded.getpixel((1, 0)) == (32, 64, 96, 255)


def test_attach_match_image_adds_payload_to_one_match(tmp_path) -> None:
    image_path = tmp_path / "match.png"
    Image.new("RGBA", (1, 1), (0, 0, 0, 255)).save(image_path)
    match = {"rank": 1, "image_path": str(image_path)}

    result = attach_image_payload(match)
    decoded = decode_payload_image(result)

    assert result["rank"] == 1
    assert result["image_path"] == str(image_path)
    assert result["image_mime_type"] == "image/png"
    assert result["image_size_bytes"] > 0
    assert decoded.mode == "RGBA"
