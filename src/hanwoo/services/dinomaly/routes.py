from __future__ import annotations

import io
import time
from typing import Literal

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

from hanwoo.services.dinomaly.pipeline import DinomalyService

router = APIRouter()
dinomaly_service: DinomalyService | None = None


def set_dinomaly_service(service: DinomalyService) -> None:
    global dinomaly_service
    dinomaly_service = service


def get_dinomaly_service() -> DinomalyService:
    if dinomaly_service is None or not dinomaly_service.is_loaded():
        raise HTTPException(status_code=503, detail="Dinomaly service is not ready.")
    return dinomaly_service


async def _read_image(file: UploadFile) -> Image.Image:
    content = await file.read()
    try:
        return Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc


@router.get("/health")
def health():
    svc = dinomaly_service
    return {
        "status": "healthy" if (svc and svc.is_loaded()) else "not_loaded",
        "model_loaded": svc.is_loaded() if svc else False,
        "threshold": svc.threshold if svc else None,
        "score_mode": svc.score_mode if svc else None,
        "device": str(svc.device) if svc else None,
    }


@router.post("/infer")
async def infer(
    file: UploadFile = File(description="Hanwoo image to inspect."),
    preprocess: bool = True,
    heatmap: bool = True,
):
    image = await _read_image(file)

    t0 = time.perf_counter()
    if preprocess:
        from hanwoo.core.preprocessing import preprocess_for_matching as do_preprocess
        try:
            image = do_preprocess(image)
        except Exception:
            pass
    t_preprocess = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    result = get_dinomaly_service().predict(image, return_heatmap=heatmap)
    t_infer = (time.perf_counter() - t1) * 1000

    return {
        "filename": file.filename,
        "preprocess": preprocess,
        "heatmap": heatmap,
        "preprocess_ms": round(t_preprocess, 1),
        "infer_ms": round(t_infer, 1),
        "total_ms": round(t_preprocess + t_infer, 1),
        **result,
    }


class ThresholdRequest(BaseModel):
    threshold: float


@router.get("/threshold")
def get_threshold():
    svc = get_dinomaly_service()
    return {"threshold": svc.threshold}


@router.put("/threshold")
def set_threshold(body: ThresholdRequest):
    if body.threshold <= 0:
        raise HTTPException(status_code=422, detail="threshold must be > 0")
    get_dinomaly_service().set_threshold(body.threshold)
    return {"threshold": body.threshold, "updated": True}


class ScoreModeRequest(BaseModel):
    score_mode: Literal["full", "roi_max", "roi_topk"]


@router.get("/score-mode")
def get_score_mode():
    return {"score_mode": get_dinomaly_service().score_mode}


@router.put("/score-mode")
def set_score_mode(body: ScoreModeRequest):
    get_dinomaly_service().set_score_mode(body.score_mode)
    return {"score_mode": body.score_mode, "updated": True}
