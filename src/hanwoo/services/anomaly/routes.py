from __future__ import annotations

import io
import time
# import cv2
import numpy as np
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

from hanwoo.services.anomaly.pipeline import AnomalyService

import httpx
from hanwoo.core.config import HANWOO_API_KEY

import asyncio
from PIL import Image 

router = APIRouter()
anomaly_service: AnomalyService | None = None


def set_anomaly_service(service: AnomalyService) -> None:
    global anomaly_service
    anomaly_service = service


def get_anomaly_service() -> AnomalyService:
    if anomaly_service is None or not anomaly_service.is_loaded():
        raise HTTPException(status_code=503, detail="Anomaly service is not ready.")
    return anomaly_service


async def _read_image(file: UploadFile) -> Image.Image:
    content = await file.read()
    try:
        return Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc


# ── Health ────────────────────────────────────────────────────────────────────

@router.get("/health")
def health():
    svc = anomaly_service
    return {
        "status": "healthy" if (svc and svc.is_loaded()) else "not_loaded",
        "bank_loaded": svc.is_loaded() if svc else False,
        "bank_size": svc.bank.size if (svc and svc.bank) else 0,
        "threshold": svc.bank.threshold if (svc and svc.bank) else None,
        "device": str(svc.device) if svc else None,
    }


# ── Inference ─────────────────────────────────────────────────────────────────

@router.post("/infer")
async def infer(
    file: Annotated[UploadFile, File(description="Hanwoo image to inspect.")],
    preprocess: bool = True,
):
    """이상탐지 단일 추론.

    preprocess=true (기본값): 배경제거 + 기울기보정 + 크롭 후 추론.
    preprocess=false: 리사이즈만 적용.
    """
    image = await _read_image(file)

    t0 = time.perf_counter()
    if preprocess:
        from hanwoo.core.preprocessing import preprocess as do_preprocess
        try:
            image = do_preprocess(image)
        except Exception:
            pass
    t_preprocess = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    result = get_anomaly_service().predict(image)
    t_infer = (time.perf_counter() - t1) * 1000

    return {
        "filename": file.filename,
        "preprocess": preprocess,
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

@router.post('/infer/save')
async def infer_save(body: InferSaveRequest):
	img_path 	= Path(f"/app/storage/rmb2/save/{body.prod_date}/{body.cattle_no}/{body.c_code}_before.png")

	print(f"img_path: {img_path}")

	if not img_path.exists():
		raise HTTPException(status_code=404, detail=f"이미지를 찾을 수 없습니다: {img_path}")

	try:
		img 	= Image.open(img_path).convert("RGB")
	except Exception as e:
		raise HTTPException(status_code=400, detail=f"허용되지 않는 이미지 포맷입니다. {e}") from e
	
	t0 		= time.perf_counter()
	

	# result_json 	= result.json()

	# result["is_anomaly"] = False
	
	if body.anomaly_YN == 'Y':
		result 	= get_anomaly_service().predict(img)
		t_infer 	= (time.perf_counter() - t0) * 1000
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
		result = {"no_check": True}
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


@router.get("/warmup")	
async def warmup():
	dummy 	= Image.new("RGB", (224, 224))
	get_anomaly_service().predict(dummy)
	
	async with httpx.AsyncClient(timeout=10.0) as client:
		response = await client.get(
			"http://matching:8000/warmup",
			headers={"X-API-Key": HANWOO_API_KEY}
		)

	return {
		"anomaly": {
			"errno": 0,
		},		
		"matching": response.json()
	}



# ── Threshold ─────────────────────────────────────────────────────────────────

class ThresholdRequest(BaseModel):
    threshold: float


@router.get("/threshold")
def get_threshold():
    svc = get_anomaly_service()
    return {"threshold": svc.bank.threshold}


@router.put("/threshold")
def set_threshold(body: ThresholdRequest):
    if body.threshold <= 0:
        raise HTTPException(status_code=422, detail="threshold must be > 0")
    get_anomaly_service().set_threshold(body.threshold)
    return {"threshold": body.threshold, "updated": True}


# ── Evaluate ──────────────────────────────────────────────────────────────────

class EvaluateRequest(BaseModel):
    test_base_dir: str = "/app/data/test"
    category_dirs: list[str] = ["비닐", "뼈", "실", "정맥혈응고체", "천"]
    images2_dir: str = "/app/data/test/images2"
    preprocess: bool = True


def _compute_metrics(scores: list[float], labels: list[int], threshold: float) -> dict:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    s = np.array(scores); l = np.array(labels)
    preds = (s >= threshold).astype(int)
    cm = confusion_matrix(l, preds, labels=[0,1]).tolist()
    return {
        "accuracy":  round(float(accuracy_score(l, preds)), 4),
        "precision": round(float(precision_score(l, preds, zero_division=0)), 4),
        "recall":    round(float(recall_score(l, preds, zero_division=0)), 4),
        "f1":        round(float(f1_score(l, preds, zero_division=0)), 4),
        "threshold": round(threshold, 4),
        "n_total":   len(l),
        "n_normal":  int((l==0).sum()),
        "n_anomaly": int((l==1).sum()),
        "confusion_matrix": cm,
    }


@router.post("/evaluate")
def evaluate(req: EvaluateRequest):
    """테스트 데이터셋 전체 성능평가 (260331_beef.py와 동일한 전처리)."""
    svc = get_anomaly_service()
    threshold = svc.bank.threshold
    if threshold is None:
        raise HTTPException(status_code=400, detail="임계값이 없습니다.")

    if req.preprocess:
        from hanwoo.core.preprocessing import preprocess as do_preprocess
    else:
        do_preprocess = None

    exts = {".jpg", ".jpeg", ".png"}
    all_scores, all_labels = [], []
    cat_data: dict[str, dict] = {}
    skipped = []

    for cat in req.category_dirs:
        image_dir = Path(req.test_base_dir) / cat / "images"
        label_dir = Path(req.test_base_dir) / cat / "labels"
        if not image_dir.exists():
            continue

        for img_file in sorted(image_dir.iterdir()):
            if img_file.suffix.lower() not in exts:
                continue

            lbl_file = None
            if label_dir.exists():
                for lf in label_dir.iterdir():
                    if lf.stem == img_file.stem + "_mask":
                        lbl_file = lf
                        break

            if lbl_file is None:
                skipped.append({"path": str(img_file), "error": "라벨 없음"})
                continue

            try:
                image = Image.open(img_file).convert("RGB")
                if do_preprocess:
                    try:
                        image = do_preprocess(image)
                    except Exception:
                        pass
                result = svc.predict(image)
                score = result["anomaly_score"]

                gt_np = np.array(Image.open(lbl_file).convert("L"))
                gt_binary = (gt_np > 127).astype(np.uint8) if gt_np.max() > 1 else gt_np.astype(np.uint8)
                gt_label = 1 if gt_binary.sum() > 0 else 0

                all_scores.append(score); all_labels.append(gt_label)
                if cat not in cat_data:
                    cat_data[cat] = {"scores": [], "labels": []}
                cat_data[cat]["scores"].append(score)
                cat_data[cat]["labels"].append(gt_label)

            except Exception as e:
                skipped.append({"path": str(img_file), "error": str(e)})

    images2_dir = Path(req.images2_dir)
    if images2_dir.exists():
        for img_file in sorted(images2_dir.iterdir()):
            if img_file.suffix.lower() not in exts:
                continue
            try:
                image = Image.open(img_file).convert("RGB")
                if do_preprocess:
                    try:
                        image = do_preprocess(image)
                    except Exception:
                        pass
                result = svc.predict(image)
                score = result["anomaly_score"]

                all_scores.append(score); all_labels.append(0)
                if "images2" not in cat_data:
                    cat_data["images2"] = {"scores": [], "labels": []}
                cat_data["images2"]["scores"].append(score)
                cat_data["images2"]["labels"].append(0)

            except Exception as e:
                skipped.append({"path": str(img_file), "error": str(e)})

    if not all_scores:
        raise HTTPException(status_code=400, detail=f"평가할 이미지가 없습니다. skipped: {skipped[:3]}")

    return {
        "total": _compute_metrics(all_scores, all_labels, threshold),
        "categories": {cat: _compute_metrics(d["scores"], d["labels"], threshold) for cat, d in cat_data.items()},
        "n_evaluated": len(all_scores),
        "n_skipped": len(skipped),
        "skipped": skipped[:10],
    }
