import hashlib
import io
import zipfile
from pathlib import Path

from hanwoo.core.sysinfo import system_info, weight_info
from hanwoo.core.zip_dataset import (
    extracted_zip,
    find_named_dir,
    find_subtree,
    group_by_subdir,
    images_in,
)


class _Upload:
    """Minimal stand-in for UploadFile: extracted_zip only touches .file."""

    def __init__(self, blob: bytes):
        self.file = io.BytesIO(blob)


def _zip(names: list[str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        for name in names:
            archive.writestr(name, b"not-a-real-image")
    return buf.getvalue()


def test_zip_layout_discovery():
    upload = _Upload(_zip([
        "bench/test/after/2025-01-01/a.jpg",
        "bench/test/after/2025-01-01/b.jpg",
        "bench/test/before/2025-01-01/a.jpg",
        "bench/test/before/2025-01-02/c.png",
        "bench/test/after/README.txt",
        "bench/test/abnormal/x.jpg",
        "bench/test/good/y.jpg",
        "../escape.jpg",
    ]))
    with extracted_zip(upload) as root:
        assert not (root.parent / "escape.jpg").exists(), "zip traversal escaped the temp dir"

        gallery = find_subtree(root, "test/after")
        query = find_subtree(root, "test/before")
        assert gallery is not None and query is not None
        assert find_subtree(root, "test/missing") is None

        gallery_dates = group_by_subdir(gallery)
        query_dates = group_by_subdir(query)
        assert set(gallery_dates) == {"2025-01-01"}, "README.txt must not create a group"
        assert sorted(set(gallery_dates) & set(query_dates)) == ["2025-01-01"]
        assert [p.name for p in gallery_dates["2025-01-01"]] == ["a.jpg", "b.jpg"]

        abnormal = find_named_dir(root, "abnormal")
        good = find_named_dir(root, "good")
        assert abnormal is not None and good is not None
        assert abnormal.parent == good.parent
        assert len(images_in(abnormal)) == 1
        assert find_named_dir(root, "nope") is None

    assert not root.exists(), "temp dir must be cleaned up"


def test_loose_images_group_under_empty_key():
    upload = _Upload(_zip(["test/after/a.jpg"]))
    with extracted_zip(upload) as root:
        assert set(group_by_subdir(find_subtree(root, "test/after"))) == {""}


def test_weight_checksum_matches_file(tmp_path: Path):
    blob = b"weights" * 100
    path = tmp_path / "best_model.pth"
    path.write_bytes(blob)

    info = weight_info(path)
    assert info["exists"] and info["size_bytes"] == len(blob)
    assert info["sha256"] == hashlib.sha256(blob).hexdigest()
    assert weight_info(tmp_path / "missing.pth") == {
        "path": str(tmp_path / "missing.pth"),
        "exists": False,
    }

    path.write_bytes(blob + b"!")
    assert weight_info(path)["sha256"] == hashlib.sha256(blob + b"!").hexdigest(), (
        "cache must not serve a stale checksum after the checkpoint changes"
    )


def test_system_info_reports_hardware_and_weights(tmp_path: Path):
    path = tmp_path / "encoder.pt"
    path.write_bytes(b"x")
    info = system_info("matching", "cpu", {"matching": path}, model_loaded=True)
    assert info["service"] == "matching"
    assert info["model_loaded"] is True
    assert info["hardware"]["device_type"] == "cpu"
    assert "cuda_available" in info["hardware"]
    assert info["weights"]["matching"]["sha256"]
    assert info["packages"]["torch"]
