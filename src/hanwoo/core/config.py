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
HANWOO_API_KEY = os.getenv("HANWOO_API_KEY", "")

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
    os.getenv("DINOMALY_MODEL_PATH", MODELS_DIR / "dinomaly" / "best_model.pth")
)
# Calibrated on the 200-image held-out set (100 good / 100 abnormal):
# roi_topk @0.192822 gives recall 0.95 at specificity 0.96 (AUROC 0.9846),
# vs roi_max @0.206186 which needs 15 false positives for the same recall.
DINOMALY_THRESHOLD = float(os.getenv("DINOMALY_THRESHOLD", "0.192822"))
DINOMALY_SCORE_MODE = os.getenv("DINOMALY_SCORE_MODE", "roi_topk")
DINOMALY_TOP_K_RATIO = float(os.getenv("DINOMALY_TOP_K_RATIO", "0.01"))
DINOMALY_ENCODER_NAME = os.getenv("DINOMALY_ENCODER_NAME", "dinov2reg_vit_base_14")
