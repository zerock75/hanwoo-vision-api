from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

from hanwoo.core.preprocessing import preprocess_for_matching
from hanwoo.core.zip_dataset import extracted_zip, find_named_dir, images_in
from hanwoo.services.dinomaly.pipeline import DinomalyService

import httpx
from hanwoo.core.config import HANWOO_API_KEY
import logging
import base64

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
        try:
            image = preprocess_for_matching(image)
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

class InferSaveRequest(BaseModel):
	cattle_no: str
	prod_date: str
	c_code: str
	anomaly_YN: str

@router.post("/infer/save")
async def infer_save(body: InferSaveRequest):

    # 전처리 된 이미지를 불러옴
    img_path 	= Path(f"/app/storage/rmb2/save/{body.prod_date}/{body.cattle_no}/{body.c_code}_before.png")

    if not img_path.exists():
        raise HTTPException(status_code=404, detail=f"이미지를 찾을 수 없습니다: {img_path}")
    
    t0 = time.perf_counter()
    try:
        image 	= Image.open(img_path).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"허용되지 않는 이미지 포맷입니다. {e}") from e

    if body.anomaly_YN == "Y":
        t_preprocess = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        result = get_dinomaly_service().predict(image, return_heatmap=True)
        t_infer = (time.perf_counter() - t1) * 1000

        heatmap_b64 = result["heatmap_b64"]
        heatmap_path = Path(f"/app/storage/rmb2/save/{body.prod_date}/{body.cattle_no}/{body.c_code}_htmap.png")
        with open(heatmap_path, "wb") as f:
            f.write(base64.b64decode(heatmap_b64))

            # heatmap_img = Image.open(io.BytesIO(base64.b64decode(heatmap_b64)))
            # heatmap_img = heatmap_img.resize((int(origin_w), int(origin_h)), Image.LANCZOS)
            # heatmap_img.save(heatmap_path)

        # 이물질 검사 시 에러 출력
        if result.get("is_anomaly") == True:
            return {
                "errno": 1,
                "message": "이물질 탐지",
                "anomaly": {			
                    "filename": img_path.name,
                    "infer_ms": round(t_infer, 1),
                    **result,
                },
            }
    else:
        result = {
            "no_check": True,
            "is_anomaly": False,
        }
        t_infer 	= (time.perf_counter() - t0) * 1000

    async with httpx.AsyncClient() as client:
        response 	= await client.post(
            "http://matching:8000/gallery/save",
            data={
                "cattle_no": body.cattle_no,
                "prod_date": body.prod_date,
                "c_code": body.c_code
            },
            headers={"X-API-Key": HANWOO_API_KEY}
        )

    return {
        "errno": 0,
        "message": "이물질 검사 성공",
        "anomaly": {
            
            "filename": img_path.name,
            "infer_ms": round(t_infer, 1),
            **result,
        },
        "matching":  response.json()		
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


@router.get("/system")
def system():
    svc = dinomaly_service
    return _dinomaly_system_info(svc)


def _dinomaly_system_info(svc: DinomalyService | None) -> dict:
    from hanwoo.core.config import U2NET_HOME
    from hanwoo.core.sysinfo import system_info

    if svc is None:
        raise HTTPException(status_code=503, detail="Dinomaly service is not initialized.")
    return system_info(
        service="dinomaly",
        device=svc.device,
        weights={
            "dinomaly": svc.model_path,
            "u2net": U2NET_HOME / "u2net.onnx",
        },
        model_loaded=svc.is_loaded(),
        config={
            "encoder_name": svc.encoder_name,
            "threshold": svc.threshold,
            "score_mode": svc.score_mode,
            "score_topk_ratio": svc.score_topk_ratio,
            "image_size": svc.image_size,
            "crop_size": svc.crop_size,
        },
    )


@router.post("/validate")
async def validate(
    file: UploadFile = File(description="ZIP holding test/abnormal/*.jpg and test/good/*.jpg."),
    preprocess: bool = True,
    heatmap: bool = False,
    warmup: int = 3,
    threshold: float | None = None,
    score_mode: Literal["full", "roi_max", "roi_topk"] | None = None,
):
    """Score a whole benchmark ZIP in-process: abnormal/ is anomaly, good/ is normal.

    threshold and score_mode override the service settings for this run only.
    """
    svc = get_dinomaly_service()

    with extracted_zip(file) as root:
        abnormal = find_named_dir(root, "abnormal")
        good = find_named_dir(root, "good")
        if abnormal and good and abnormal.parent != good.parent and (abnormal.parent / "good").is_dir():
            good = abnormal.parent / "good"
        if abnormal is None or good is None:
            raise HTTPException(
                status_code=422,
                detail="ZIP must contain an abnormal/ and a good/ folder with images.",
            )

        samples = sorted(
            [(p, True) for p in images_in(abnormal)] + [(p, False) for p in images_in(good)],
            key=lambda item: item[0].name,
        )

        def load(path: Path) -> tuple[Image.Image, float, bool]:
            image = Image.open(path).convert("RGB")
            t0 = time.perf_counter()
            failed = False
            if preprocess:
                # /infer degrades to the raw image here; report it instead of hiding it.
                try:
                    image = preprocess_for_matching(image)
                except Exception:
                    failed = True
            return image, (time.perf_counter() - t0) * 1000, failed

        previous = (svc.threshold, svc.score_mode)
        if threshold is not None:
            svc.set_threshold(threshold)
        if score_mode is not None:
            svc.set_score_mode(score_mode)
        try:
            # First inferences pay for cudnn autotune and lazy CUDA init; keep them out of the averages.
            for path, _ in samples[:max(0, warmup)]:
                svc.predict(load(path)[0], return_heatmap=heatmap)

            rows: list[dict] = []
            counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
            sum_preprocess = sum_infer = 0.0
            preprocess_failures = 0
            start = time.perf_counter()

            for path, expected_abnormal in samples:
                image, preprocess_ms, preprocess_failed = load(path)
                preprocess_failures += int(preprocess_failed)
                t0 = time.perf_counter()
                result = svc.predict(image, return_heatmap=heatmap)
                infer_ms = (time.perf_counter() - t0) * 1000
                sum_preprocess += preprocess_ms
                sum_infer += infer_ms

                is_anomaly = bool(result["is_anomaly"])
                key = ("tp" if is_anomaly else "fn") if expected_abnormal else ("fp" if is_anomaly else "tn")
                counts[key] += 1
                rows.append({
                    "image": path.name,
                    "score": result["anomaly_score"],
                    "score_details": result["score_details"],
                    "predicted": "anomaly" if is_anomaly else "normal",
                    "expected": "anomaly" if expected_abnormal else "normal",
                    "correct": is_anomaly == expected_abnormal,
                    "preprocess_failed": preprocess_failed,
                    "preprocess_ms": round(preprocess_ms, 1),
                    "infer_ms": round(infer_ms, 1),
                    "total_ms": round(preprocess_ms + infer_ms, 1),
                })

            elapsed_s = time.perf_counter() - start
            metrics = _classification_metrics(counts, rows)
            system = _dinomaly_system_info(svc)
        finally:
            svc.set_threshold(previous[0])
            svc.set_score_mode(previous[1])

    total = len(rows)
    return {
        "filename": file.filename,
        "preprocess": preprocess,
        "heatmap": heatmap,
        "warmup": max(0, warmup),
        "threshold": system["config"]["threshold"],
        "score_mode": system["config"]["score_mode"],
        "metrics": {
            **metrics,
            "total": total,
            "n_anomaly": counts["tp"] + counts["fn"],
            "n_normal": counts["tn"] + counts["fp"],
            "n_preprocess_failed": preprocess_failures,
            "avg_preprocess_ms": round(sum_preprocess / total, 1) if total else 0.0,
            "avg_infer_ms": round(sum_infer / total, 1) if total else 0.0,
            "avg_total_ms": round((sum_preprocess + sum_infer) / total, 1) if total else 0.0,
            "elapsed_s": round(elapsed_s, 1),
        },
        "results": rows,
        "system": system,
    }


def _classification_metrics(counts: dict[str, int], rows: list[dict]) -> dict:
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    metrics = {
        "accuracy": (tp + tn) / total if total else 0.0,
        "correct": tp + tn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "confusion": counts,
    }
    if tp + fn and tn + fp:
        from sklearn.metrics import roc_auc_score

        metrics["auroc"] = float(
            roc_auc_score(
                [row["expected"] == "anomaly" for row in rows],
                [row["score"] for row in rows],
            )
        )
    return metrics
