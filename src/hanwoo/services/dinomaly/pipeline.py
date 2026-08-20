from __future__ import annotations

import sys
import time
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from torchvision import transforms

from hanwoo.core.config import (
    DINOMALY_ENCODER_NAME,
    DINOMALY_MODEL_PATH,
    DINOMALY_SCORE_MODE,
    DINOMALY_THRESHOLD,
    DINOMALY_TOP_K_RATIO,
)
from hanwoo.core.gpu import choose_device
from hanwoo.core.preprocessing import remove_background

_VENDOR_SRC = Path(__file__).resolve().parents[4] / "models" / "dinomaly" / "vendor" / "dinomaly"
if str(_VENDOR_SRC) not in sys.path:
    sys.path.insert(0, str(_VENDOR_SRC))

from dinov1.utils import trunc_normal_
from models import vit_encoder
from models.uad import ViTill
from models.vision_transformer import Block as VitBlock
from models.vision_transformer import LinearAttention2, bMlp

IMAGE_SIZE = 448
CROP_SIZE = 392
TARGET_LAYERS = [2, 3, 4, 5, 6, 7, 8, 9]
FUSE_LAYER_ENCODER = [[0, 1, 2, 3], [4, 5, 6, 7]]
FUSE_LAYER_DECODER = [[0, 1, 2, 3], [4, 5, 6, 7]]
N_DECODER_BLOCKS = 8

_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.CenterCrop(CROP_SIZE),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _binarize_mask(mask: Image.Image) -> np.ndarray:
    mask_np = np.array(mask.convert("L"), dtype=np.uint8)
    return np.where(mask_np > 10, 255, 0).astype(np.uint8)


def _erode_mask(mask: np.ndarray, boundary_erode_px: int) -> np.ndarray:
    if boundary_erode_px <= 0:
        return mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (boundary_erode_px, boundary_erode_px))
    eroded = cv2.erode(mask, kernel, iterations=1)
    return eroded if np.any(eroded) else mask.copy()


# u2net and the cv2 morphology in remove_background cost scales with input pixels
# (~600ms at 2560px), while the ROI mask is downsampled to CROP_SIZE anyway.
SEG_MAX_SIDE = 1024


def segment_beef_masks(image: Image.Image, boundary_erode_px: int = 7) -> tuple[np.ndarray, np.ndarray]:
    image = image.convert("RGB")
    scale = SEG_MAX_SIDE / max(image.size)
    seg_input = (
        image
        if scale >= 1.0
        else image.resize(
            (round(image.width * scale), round(image.height * scale)),
            Image.Resampling.BILINEAR,
        )
    )
    _, raw_mask = remove_background(
        seg_input,
        return_mask=True,
        refine_mask=True,
    )
    if raw_mask.size != image.size:
        raw_mask = raw_mask.resize(image.size, Image.Resampling.BILINEAR)
    mask_full = _binarize_mask(raw_mask)
    inner_mask = _erode_mask(mask_full, boundary_erode_px)
    return mask_full, inner_mask


def colorize_heatmap(anomaly_map: np.ndarray, original: Image.Image) -> Image.Image:
    amap = anomaly_map.copy().astype(np.float32)
    if amap.max() > amap.min():
        amap = (amap - amap.min()) / (amap.max() - amap.min())
    h, w = amap.shape
    r = np.clip(1.5 - np.abs(amap * 4 - 3), 0, 1)
    g = np.clip(1.5 - np.abs(amap * 4 - 2), 0, 1)
    b = np.clip(1.5 - np.abs(amap * 4 - 1), 0, 1)
    alpha = amap * 0.65
    overlay = Image.fromarray(
        (np.stack([r, g, b, alpha], axis=-1) * 255).astype(np.uint8),
        mode="RGBA",
    )
    original_resized = original.resize((w, h)).convert("RGBA")
    return Image.alpha_composite(original_resized, overlay).convert("RGB")


class DinomalyService:
    def __init__(
        self,
        model_path: Path = DINOMALY_MODEL_PATH,
        threshold: float = DINOMALY_THRESHOLD,
        score_mode: str = DINOMALY_SCORE_MODE,
        score_topk_ratio: float = DINOMALY_TOP_K_RATIO,
        encoder_name: str = DINOMALY_ENCODER_NAME,
        device_name: str = "auto",
    ) -> None:
        self.model_path = model_path
        self.threshold = threshold
        self.score_mode = score_mode
        self.score_topk_ratio = score_topk_ratio
        self.encoder_name = encoder_name
        self.device = choose_device(device_name)
        self.model: ViTill | None = None

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Dinomaly checkpoint not found: {self.model_path}"
            )
        encoder = vit_encoder.load(self.encoder_name)
        arch = self.encoder_name.split("_")[-2]
        if arch == "small":
            embed_dim, num_heads = 384, 6
        elif arch == "large":
            embed_dim, num_heads = 1024, 16
        else:
            embed_dim, num_heads = 768, 12

        bottleneck = nn.ModuleList([bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2)])
        decoder = nn.ModuleList([
            VitBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-8),
                attn=LinearAttention2,
            )
            for _ in range(N_DECODER_BLOCKS)
        ])

        self.model = ViTill(
            encoder=encoder,
            bottleneck=bottleneck,
            decoder=decoder,
            target_layers=TARGET_LAYERS,
            mask_neighbor_size=0,
            fuse_layer_encoder=FUSE_LAYER_ENCODER,
            fuse_layer_decoder=FUSE_LAYER_DECODER,
        ).to(self.device)

        for m in nn.ModuleList([bottleneck, decoder]).modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        state = _torch_load(self.model_path, self.device)
        if isinstance(state, dict):
            state = state.get(
                "model_state_dict",
                state.get("state_dict", state.get("model", state)),
            )
        self.model.load_state_dict(state, strict=False)
        self.model.eval()

    def is_loaded(self) -> bool:
        return self.model is not None

    def _prepare_meat_roi_masks(
        self, image: Image.Image
    ) -> tuple[np.ndarray, np.ndarray]:
        mask_full, inner_mask = segment_beef_masks(image)
        if not np.any(mask_full):
            mask_full = np.full((image.height, image.width), 255, dtype=np.uint8)
        if not np.any(inner_mask):
            inner_mask = mask_full.copy()
        mask_image = Image.fromarray(inner_mask.astype(np.uint8), mode="L")
        mask_image = mask_image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST)
        crop_start = (IMAGE_SIZE - CROP_SIZE) // 2
        crop_end = IMAGE_SIZE - crop_start
        mask_image = mask_image.crop((crop_start, crop_start, crop_end, crop_end))
        model_inner_mask = np.array(mask_image, dtype=np.uint8) > 0
        if not np.any(model_inner_mask):
            model_inner_mask = np.ones((CROP_SIZE, CROP_SIZE), dtype=bool)
        return mask_full > 0, model_inner_mask

    def _compute_score_details(
        self, anomaly_map: np.ndarray, model_inner_mask: np.ndarray
    ) -> dict[str, float]:
        full_image_score = float(np.clip(anomaly_map.max(), 0.0, 1.0))
        roi_values = anomaly_map[model_inner_mask]
        if roi_values.size == 0:
            roi_score = full_image_score
            roi_topk_score = full_image_score
        else:
            roi_score = float(np.clip(roi_values.max(), 0.0, 1.0))
            topk_count = max(1, int(np.ceil(roi_values.size * self.score_topk_ratio)))
            if topk_count >= roi_values.size:
                topk_values = roi_values
            else:
                topk_values = np.partition(
                    roi_values, roi_values.size - topk_count
                )[-topk_count:]
            roi_topk_score = float(np.clip(float(topk_values.mean()), 0.0, 1.0))
        return {
            "full_image_score": full_image_score,
            "roi_score": roi_score,
            "roi_topk_score": roi_topk_score,
        }

    def _select_score(self, score_details: dict[str, float]) -> float:
        if self.score_mode == "full":
            return float(score_details["full_image_score"])
        if self.score_mode == "roi_topk":
            return float(score_details["roi_topk_score"])
        return float(score_details["roi_score"])

    @torch.no_grad()
    def _compute_anomaly_map_and_score(
        self, image: Image.Image
    ) -> tuple[np.ndarray, float, np.ndarray, dict[str, float], dict[str, float]]:
        t0 = time.perf_counter()
        full_mask, model_inner_mask = self._prepare_meat_roi_masks(image)
        t_mask = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        img_t = _TRANSFORM(image).unsqueeze(0).to(self.device)
        t_transform = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        en, de = self.model(img_t)
        t_forward = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        anomaly_map = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float32)
        for fs, ft in zip(en, de):
            a_map = 1 - F.cosine_similarity(fs, ft)
            a_map = a_map.unsqueeze(1)
            a_map = F.interpolate(
                a_map,
                size=(CROP_SIZE, CROP_SIZE),
                mode="bilinear",
                align_corners=True,
            )
            anomaly_map += a_map[0, 0].cpu().detach().numpy()
        t_cosine = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        anomaly_map = gaussian_filter(anomaly_map, sigma=4)
        t_gaussian = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        score_details = self._compute_score_details(anomaly_map, model_inner_mask)
        score = self._select_score(score_details)
        t_score = (time.perf_counter() - t0) * 1000

        timings = {
            "mask": round(t_mask, 1),
            "transform": round(t_transform, 1),
            "forward": round(t_forward, 1),
            "cosine_loop": round(t_cosine, 1),
            "gaussian": round(t_gaussian, 1),
            "score": round(t_score, 1),
        }
        return anomaly_map, score, full_mask, score_details, timings

    @torch.no_grad()
    def predict(self, image: Image.Image, return_heatmap: bool = True) -> dict:
        if not self.is_loaded():
            raise RuntimeError("Dinomaly service is not loaded.")

        t0 = time.perf_counter()
        anomaly_map, score, full_mask, score_details, timings = self._compute_anomaly_map_and_score(image)
        t_infer = (time.perf_counter() - t0) * 1000
        timings["_compute_total"] = round(t_infer, 1)

        result = {
            "anomaly_score": round(score, 4),
            "is_anomaly": bool(score >= self.threshold),
            "threshold": round(self.threshold, 4),
            "score_mode": self.score_mode,
            "score_details": {k: round(v, 4) for k, v in score_details.items()},
            "infer_timings_ms": timings,
        }

        if return_heatmap:
            orig_w, orig_h = image.size
            crop_start = (IMAGE_SIZE - CROP_SIZE) // 2
            crop_end = IMAGE_SIZE - crop_start
            x_start = int(crop_start * orig_w / IMAGE_SIZE)
            x_end = int(crop_end * orig_w / IMAGE_SIZE)
            y_start = int(crop_start * orig_h / IMAGE_SIZE)
            y_end = int(crop_end * orig_h / IMAGE_SIZE)

            t0 = time.perf_counter()
            anomaly_map_display = np.zeros((orig_h, orig_w), dtype=np.float32)
            crop_w = x_end - x_start
            crop_h = y_end - y_start
            amap_tensor = torch.from_numpy(anomaly_map).unsqueeze(0).unsqueeze(0)
            amap_cropped_resized = F.interpolate(
                amap_tensor,
                size=(crop_h, crop_w),
                mode="bilinear",
                align_corners=False,
            )
            anomaly_map_display[y_start:y_end, x_start:x_end] = amap_cropped_resized[0, 0].numpy()
            anomaly_map_display = np.where(full_mask, anomaly_map_display, 0.0)
            t_post = (time.perf_counter() - t0) * 1000
            timings["post_process"] = round(t_post, 1)

            t0 = time.perf_counter()
            heatmap_img = colorize_heatmap(anomaly_map_display, image)
            import base64
            import io
            buf = io.BytesIO()
            heatmap_img.save(buf, format="PNG")
            heatmap_b64 = base64.b64encode(buf.getvalue()).decode()
            t_heatmap = (time.perf_counter() - t0) * 1000
            timings["heatmap_b64"] = round(t_heatmap, 1)
            result["heatmap_b64"] = heatmap_b64

        timings["predict_total"] = round(
            t_infer + timings.get("post_process", 0) + timings.get("heatmap_b64", 0), 1
        )
        return result

    def set_threshold(self, value: float) -> None:
        self.threshold = value

    def set_score_mode(self, value: str) -> None:
        self.score_mode = value
