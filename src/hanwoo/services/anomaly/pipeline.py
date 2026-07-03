from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from hanwoo.core.config import (
    ANOMALY_HEATMAP_POW,
    ANOMALY_IMAGE_SIZE,
    ANOMALY_MODEL_PATH,
    ANOMALY_THRESHOLD_PATH,
)
from hanwoo.core.encoders.dinov2 import DINOv2Extractor
from hanwoo.core.gpu import choose_device
from hanwoo.core.vectorstore.memory_bank import MemoryBank

_REGION_NAMES: dict[str, str] = {
    f"{r}_{c}": f"{rn} {cn}"
    for r, rn in [("top", "상단"), ("mid", "중단"), ("bot", "하단")]
    for c, cn in [("left", "좌측"), ("center", "중앙"), ("right", "우측")]
}

_TRANSFORM = transforms.Compose([
    transforms.Resize(ANOMALY_IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _upsample_heatmap(hmap: np.ndarray, bg_mask: np.ndarray | None = None) -> np.ndarray:
    up = cv2.resize(hmap, ANOMALY_IMAGE_SIZE[::-1], interpolation=cv2.INTER_CUBIC)
    up = cv2.GaussianBlur(up, (31, 31), 0)
    if bg_mask is not None:
        up[bg_mask] = 0.0
        valid = up[~bg_mask]
    else:
        valid = up.flatten()

    # 유효픽셀 `0` 예외처리  
    if valid.size == 0:
        return np.zeros_like(up)

    vmin, vmax = valid.min(), valid.max()
    up = np.clip((up - vmin) / (vmax - vmin + 1e-8), 0, 1)
    if bg_mask is not None:
        up[bg_mask] = 0.0
    return np.power(up, ANOMALY_HEATMAP_POW).astype(np.float32)


def _locate_regions(hmap: np.ndarray, thr: float = 0.4) -> list[str]:
    H, W = hmap.shape
    found = []
    for ri, r in enumerate(["top", "mid", "bot"]):
        for ci, c in enumerate(["left", "center", "right"]):
            cell = hmap[ri*H//3:(ri+1)*H//3, ci*W//3:(ci+1)*W//3]
            if cell.max() > thr:
                found.append((cell.mean(), _REGION_NAMES[f"{r}_{c}"]))
    found.sort(key=lambda x: x[0], reverse=True)
    return [f[1] for f in found] or ["이상 없음"]


class AnomalyService:
    def __init__(
        self,
        model_path: Path = ANOMALY_MODEL_PATH,
        threshold_path: Path = ANOMALY_THRESHOLD_PATH,
        device_name: str = "auto",
    ) -> None:
        self.model_path = model_path
        self.threshold_path = threshold_path
        self.device = choose_device(device_name)
        self.extractor: DINOv2Extractor | None = None
        self.bank: MemoryBank | None = None

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Anomaly memory bank not found: {self.model_path}\n"
                "Run scripts/build_memory_bank.py to build it first."
            )
        self.extractor = DINOv2Extractor().to(self.device).half()
        self.extractor.eval()
        self.bank = MemoryBank(device=self.device)
        self.bank.load(self.model_path)
        if self.threshold_path.exists():
            with open(self.threshold_path) as f:
                data = json.load(f)
            self.bank.threshold = float(data["threshold"])

    def is_loaded(self) -> bool:
        return self.extractor is not None and self.bank is not None and self.bank.is_loaded()

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict:
        if not self.is_loaded():
            raise RuntimeError("Anomaly service is not loaded.")

        img_rgb = image.convert("RGB")
        img_tensor = _TRANSFORM(img_rgb).unsqueeze(0).to(self.device)

        img_np = np.array(img_rgb.resize(ANOMALY_IMAGE_SIZE[::-1], Image.BILINEAR))
        bg_mask = (img_np.sum(axis=2) == 0)

        patch_feats = self.extractor(img_tensor)
        bg_tensor = torch.from_numpy(bg_mask).unsqueeze(0)
        scores, hmaps = self.bank.predict(patch_feats, bg_masks=bg_tensor)

        score = float(scores[0])
        hmap_up = _upsample_heatmap(hmaps[0], bg_mask)
        regions = _locate_regions(hmap_up)
        heatmap_b64 = self._make_overlay_b64(img_np, hmap_up, bg_mask)

        threshold = self.bank.threshold
        return {
            "anomaly_score": round(score, 4),
            "is_anomaly": bool(threshold is not None and score >= threshold),
            "threshold": round(threshold, 4) if threshold is not None else None,
            "regions": regions,
            "heatmap_b64": heatmap_b64,
        }

    def set_threshold(self, value: float) -> None:
        if self.bank is None:
            raise RuntimeError("Bank not loaded.")
        self.bank.threshold = value
        self.threshold_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.threshold_path, "w") as f:
            json.dump({"threshold": value}, f)

    @staticmethod
    def _make_overlay_b64(img_np, hmap, bg_mask) -> str:
        import base64
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        colored = cv2.applyColorMap((hmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = img_bgr.copy()
        meat = ~bg_mask
        overlay[meat] = cv2.addWeighted(img_bgr, 0.5, colored, 0.5, 0)[meat]
        _, buf = cv2.imencode(".png", overlay)
        return base64.b64encode(buf.tobytes()).decode()
