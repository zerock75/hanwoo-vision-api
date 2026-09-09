from __future__ import annotations

import hashlib
import os
import platform
import socket
import subprocess
from datetime import datetime, timezone
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch


REPORTED_PACKAGES = (
    "torch",
    "torchvision",
    "timm",
    "transformers",
    "numpy",
    "scipy",
    "scikit-learn",
    "opencv-python-headless",
    "pillow",
    "onnxruntime-gpu",
    "rembg",
    "qdrant-client",
    "fastapi",
)


@lru_cache(maxsize=32)
def _sha256(path: str, size: int, mtime: float) -> str:
    # size/mtime are cache keys only: a replaced checkpoint rehashes, a reused one does not.
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def weight_info(path: Path | str) -> dict:
    path = Path(path)
    if not path.is_file():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "sha256": _sha256(str(path), stat.st_size, stat.st_mtime),
    }


@lru_cache(maxsize=1)
def _nvidia_smi() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return {}
    driver, name, memory = (part.strip() for part in out[0].split(",", 2))
    return {"driver_version": driver, "smi_gpu_name": name, "smi_gpu_memory": memory}


def device_info(device: torch.device | str) -> dict:
    device = torch.device(device)
    info = {
        "device": str(device),
        "device_type": device.type,
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }
    if torch.cuda.is_available():
        index = device.index or 0
        props = torch.cuda.get_device_properties(index)
        info |= {
            "gpu_name": props.name,
            "gpu_index": index,
            "gpu_count": torch.cuda.device_count(),
            "gpu_capability": f"{props.major}.{props.minor}",
            "gpu_total_memory_mb": round(props.total_memory / 1024**2),
            "gpu_memory_allocated_mb": round(torch.cuda.memory_allocated(index) / 1024**2),
            "gpu_memory_reserved_mb": round(torch.cuda.memory_reserved(index) / 1024**2),
            **_nvidia_smi(),
        }
    return info


@lru_cache(maxsize=1)
def package_versions() -> dict:
    versions = {}
    for name in REPORTED_PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            continue
    return versions


def system_info(
    service: str,
    device: torch.device | str,
    weights: dict[str, Path | str],
    **extra,
) -> dict:
    """Everything needed to confirm a service loaded the weights and hardware we expect."""
    return {
        "service": service,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hardware": device_info(device),
        "weights": {name: weight_info(path) for name, path in weights.items()},
        "packages": package_versions(),
        **extra,
    }
