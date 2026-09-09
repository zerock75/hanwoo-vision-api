from __future__ import annotations

import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parents[1]

MODELS_DIR = Path(os.getenv("HANWOO_MODELS_DIR", PROJECT_ROOT / "models"))
STORAGE_DIR = Path(
    os.getenv("HANWOO_STORAGE_DIR", PROJECT_ROOT / "storage" / "matching")
)

MATCHING_MODEL_PATH = Path(
    os.getenv("MATCHING_MODEL_PATH", MODELS_DIR / "matching" / "encoder.pt")
)
U2NET_HOME = Path(os.getenv("U2NET_HOME", MODELS_DIR / "u2net"))

DEVICE = os.getenv("HANWOO_DEVICE", "auto")
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
HANWOO_API_KEY = os.getenv("HANWOO_API_KEY")

GALLERY_DIR = Path(os.getenv("HANWOO_GALLERY_DIR", STORAGE_DIR / "gallery_images"))

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "hanwoo_matching_gallery")

ANOMALY_STORAGE_DIR = Path(
    os.getenv("ANOMALY_STORAGE_DIR", PROJECT_ROOT / "storage" / "anomaly")
)
ANOMALY_MODEL_PATH = Path(
    os.getenv("ANOMALY_MODEL_PATH", MODELS_DIR / "anomaly" / "memory_bank.pth")
)
ANOMALY_THRESHOLD_PATH = Path(
    os.getenv("ANOMALY_THRESHOLD_PATH", MODELS_DIR / "anomaly" / "threshold.json")
)
ANOMALY_DINO_LAYERS: list[int] = [
    int(x.strip())
    for x in os.getenv("ANOMALY_DINO_LAYERS", "10,11").split(",")
    if x.strip()
]
ANOMALY_CORESET_RATIO = float(os.getenv("ANOMALY_CORESET_RATIO", "0.08"))
ANOMALY_K_NEIGHBORS = int(os.getenv("ANOMALY_K_NEIGHBORS", "3"))
ANOMALY_TOP_K_RATIO = float(os.getenv("ANOMALY_TOP_K_RATIO", "0.4"))
ANOMALY_THRESH_PERCENTILE = int(os.getenv("ANOMALY_THRESH_PERCENTILE", "87"))
ANOMALY_IMAGE_SIZE: tuple[int, int] = (672, 672)
ANOMALY_HEATMAP_POW = float(os.getenv("ANOMALY_HEATMAP_POW", "2.5"))

DINOMALY_MODEL_PATH = Path(
    os.getenv(
        "DINOMALY_MODEL_PATH", MODELS_DIR / "dinomaly" / "best_model_dinov3_large.pth"
    )
)
# The threshold is backbone-specific: each checkpoint scores on its own scale, so
# it must move together with DINOMALY_MODEL_PATH. Crossing them fails silently.
# All three are score_mode=roi_topk.
#
#   best_model_dinov3_large.pth  dinov3_vit_large_16   0.085646  (default)
#   best_model_dinov3_base.pth   dinov3_vit_base_16    0.136242
#   best_model.pth               dinov2reg_vit_base_14 0.192822  (DINOv2)
#
# Measured on data/2709test_latest (n=34, roi_topk): large AUROC 0.972 at 155ms,
# base 0.909 at 123ms. Large is ~26% slower per inference, but preprocessing
# (~500ms) dominates either way.
#
# The encoder name and input sizes are read from the checkpoint itself for the
# DINOv3 files; DINOMALY_ENCODER_NAME is only the fallback for the DINOv2 one,
# which is a bare state_dict with no metadata.
DINOMALY_THRESHOLD = float(os.getenv("DINOMALY_THRESHOLD", "0.085646"))
DINOMALY_SCORE_MODE = os.getenv("DINOMALY_SCORE_MODE", "roi_topk")
DINOMALY_TOP_K_RATIO = float(os.getenv("DINOMALY_TOP_K_RATIO", "0.01"))
DINOMALY_ENCODER_NAME = os.getenv("DINOMALY_ENCODER_NAME", "dinov2reg_vit_base_14")
