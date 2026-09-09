from __future__ import annotations

import tempfile
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

from fastapi import HTTPException, UploadFile


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@contextmanager
def extracted_zip(upload: UploadFile) -> Iterator[Path]:
    """Extract an uploaded ZIP to a temp dir that is removed on exit.

    ZipFile reads straight from the spooled upload and extractall streams member
    by member, so a multi-GB benchmark zip never lands in RAM whole. extractall
    also drops absolute paths and '..' parts, so a hostile zip stays inside.
    """
    with tempfile.TemporaryDirectory(prefix="validate-") as tmp:
        upload.file.seek(0)
        try:
            with zipfile.ZipFile(upload.file) as archive:
                archive.extractall(tmp)
        except zipfile.BadZipFile as exc:
            raise HTTPException(status_code=400, detail=f"Invalid ZIP: {exc}") from exc
        yield Path(tmp)


def images_in(directory: Path, recursive: bool = True) -> list[Path]:
    paths = directory.rglob("*") if recursive else directory.iterdir()
    return sorted(
        (p for p in paths if p.is_file() and p.suffix.lower() in IMAGE_EXTS),
        key=lambda p: p.name,
    )


def _dirs(root: Path) -> list[Path]:
    inner = (p for p in root.rglob("*") if p.is_dir())
    return [root, *sorted(inner, key=lambda p: (len(p.parts), p))]


def find_subtree(root: Path, relative: str) -> Path | None:
    """Locate `relative` (e.g. "test/after") anywhere under root, wrapper dirs and all."""
    parts = PurePosixPath(relative.strip("/")).parts
    for base in _dirs(root):
        candidate = base.joinpath(*parts)
        if candidate.is_dir():
            return candidate
    return None


def find_named_dir(root: Path, name: str) -> Path | None:
    """Shallowest directory called `name` that actually holds images."""
    for path in _dirs(root):
        if path.name == name and images_in(path):
            return path
    return None


def group_by_subdir(folder: Path) -> dict[str, list[Path]]:
    """Images keyed by immediate subdirectory (the date dirs); loose images land under ''."""
    groups = {}
    loose = images_in(folder, recursive=False)
    if loose:
        groups[""] = loose
    for entry in sorted(folder.iterdir()):
        if entry.is_dir():
            images = images_in(entry)
            if images:
                groups[entry.name] = images
    return groups
