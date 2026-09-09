from __future__ import annotations

import base64
import mimetypes
from pathlib import Path


def encode_image_payload(path: Path) -> dict[str, str | int]:
    data = path.read_bytes()
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return {
        "image_mime_type": mime_type,
        "image_size_bytes": len(data),
        "image_base64": base64.b64encode(data).decode("ascii"),
    }


def attach_image_payload(match: dict) -> dict:
    image_path = Path(str(match["image_path"]))
    return {**match, **encode_image_payload(image_path)}
