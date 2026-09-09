"""Contract for the DINOv3 backbone path — ported from the training repo's
test_dinov3.py. The DINOv2 path is exercised implicitly: resolve_sizes must keep
returning its 448/392 geometry."""
from functools import partial
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from hanwoo.core.config import MODELS_DIR
from hanwoo.services.dinomaly import pipeline as P
from models import vit_encoder  # noqa: E402  (vendored, added to sys.path by pipeline)
from models.uad import ViTill
from models.vision_transformer import Block as VitBlock
from models.vision_transformer import LinearAttention2, bMlp

BASE_CKPT = MODELS_DIR / "dinomaly" / "best_model_dinov3_base.pth"


def test_resolve_sizes():
    assert vit_encoder.resolve_sizes("dinov3_vit_base_16", None, None) == (512, 448)
    assert vit_encoder.resolve_sizes("dinov2reg_vit_base_14", None, None) == (448, 392)
    # an explicit crop that doesn't tile into whole patches gets trimmed down
    assert vit_encoder.resolve_sizes("dinov3_vit_base_16", 448, 392) == (448, 384)
    assert vit_encoder.resolve_sizes("dinov2reg_vit_base_14", 448, 392) == (448, 392)


def test_dinov3_forward_shapes():
    """The timm DINOv3 encoder has no prepare_tokens and no prefix tokens, so
    ViTill must take its forward_intermediates branch and drop none of the grid."""
    _, crop = vit_encoder.resolve_sizes("dinov3_vit_base_16", None, None)
    dim = 768
    model = ViTill(
        encoder=vit_encoder.load("dinov3_vit_base_16"),
        bottleneck=nn.ModuleList([bMlp(dim, dim * 4, dim, drop=0.2)]),
        decoder=nn.ModuleList([
            VitBlock(dim=dim, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                     norm_layer=partial(nn.LayerNorm, eps=1e-8), attn=LinearAttention2)
            for _ in range(P.N_DECODER_BLOCKS)
        ]),
        target_layers=P.TARGET_LAYERS,
        mask_neighbor_size=0,
        fuse_layer_encoder=P.FUSE_LAYER_ENCODER,
        fuse_layer_decoder=P.FUSE_LAYER_DECODER,
    ).eval()

    with torch.no_grad():
        en, de = model(torch.randn(1, 3, crop, crop))

    side = crop // 16
    assert [tuple(t.shape) for t in en] == [(1, dim, side, side)] * 2
    assert [tuple(t.shape) for t in de] == [(1, dim, side, side)] * 2


@pytest.mark.skipif(not BASE_CKPT.exists(), reason=f"missing {BASE_CKPT}")
def test_service_reads_geometry_from_checkpoint():
    """load() must take the encoder and its sizes from the checkpoint rather than
    from DINOMALY_ENCODER_NAME, and must reject a state_dict that doesn't fit."""
    svc = P.DinomalyService(
        model_path=BASE_CKPT,
        encoder_name="dinov2reg_vit_base_14",  # deliberately wrong; ckpt must win
        device_name="cpu",
    )
    svc.load()  # raises if any key is missing or unexpected

    assert svc.encoder_name == "dinov3_vit_base_16"
    assert (svc.image_size, svc.crop_size) == (512, 448)
    assert svc.transform.transforms[0].size == (512, 512)
    assert svc.transform.transforms[2].size == (448, 448)


def test_rejects_unknown_score_mode():
    with pytest.raises(ValueError, match="DINOMALY_SCORE_MODE"):
        P.DinomalyService(score_mode="roi_topk_typo", device_name="cpu")
