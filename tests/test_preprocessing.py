from __future__ import annotations

import pytest
from PIL import Image

pytest.importorskip("cv2")
pytest.importorskip("numpy")

from hanwoo.core.preprocessing import fill_mask_holes
from hanwoo.core.preprocessing import apply_mask_to_rgb


def test_fill_mask_holes_keeps_outer_background() -> None:
    mask = Image.new("L", (5, 5), 0)
    for y in range(1, 4):
        for x in range(1, 4):
            mask.putpixel((x, y), 255)
    mask.putpixel((2, 2), 0)

    result = fill_mask_holes(mask)

    assert result.getpixel((2, 2)) == 255
    assert result.getpixel((0, 0)) == 0


def test_filled_mask_hole_becomes_white_in_rgb() -> None:
    image = Image.new("RGBA", (3, 3), (30, 30, 30, 255))
    image.putpixel((1, 1), (0, 0, 0, 0))
    mask = Image.new("L", (3, 3), 255)

    result = apply_mask_to_rgb(image, mask, rembg_output=image).convert("RGB")

    assert result.getpixel((1, 1)) == (255, 255, 255)
