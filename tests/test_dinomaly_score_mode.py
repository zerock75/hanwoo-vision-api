from hanwoo.services.dinomaly.pipeline import DinomalyService


def test_select_score_follows_mode():
    svc = DinomalyService.__new__(DinomalyService)
    details = {"full_image_score": 0.9, "roi_score": 0.7, "roi_topk_score": 0.3}
    for mode, expected in (
        ("full", 0.9),
        ("roi_max", 0.7),
        ("roi_topk", 0.3),
    ):
        svc.set_score_mode(mode)
        assert svc.score_mode == mode
        assert svc._select_score(details) == expected




def test_segment_downscale_matches_full_res():
    """Capping segmentation resolution must not move the ROI mask."""
    import numpy as np
    from PIL import Image, ImageDraw
    from hanwoo.services.dinomaly import pipeline as P

    img = Image.new("RGB", (2048, 1536), (30, 30, 30))
    ImageDraw.Draw(img).ellipse((500, 300, 1500, 1200), fill=(150, 40, 40))

    original = P.SEG_MAX_SIDE
    try:
        P.SEG_MAX_SIDE = 10**9
        _, inner_full = P.segment_beef_masks(img)
        P.SEG_MAX_SIDE = 1024
        _, inner_capped = P.segment_beef_masks(img)
    finally:
        P.SEG_MAX_SIDE = original

    assert inner_capped.shape == inner_full.shape == (1536, 2048)
    iou = np.logical_and(inner_full, inner_capped).sum() / np.logical_or(inner_full, inner_capped).sum()
    assert iou > 0.95, f"ROI mask drifted: IoU {iou:.4f}"


if __name__ == "__main__":
    test_select_score_follows_mode()
    test_segment_downscale_matches_full_res()
    print("ok")
