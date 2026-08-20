from __future__ import annotations

import io
from time import perf_counter
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from PIL import Image
from pydantic import BaseModel

from hanwoo.core.config import DEFAULT_TOP_K, MATCHING_MODEL_PATH, STORAGE_DIR
from hanwoo.core.image_payload import encode_image_payload
from hanwoo.core.preprocessing import preprocess_for_matching_with_rgba
from hanwoo.core.schemas import DirectoryImportRequest
from hanwoo.services.matching.pipeline import MatchingService




router = APIRouter()
matching_service: MatchingService | None = None


def set_matching_service(service: MatchingService) -> None:
    global matching_service
    matching_service = service


def get_matching_service() -> MatchingService:
    if matching_service is None:
        raise RuntimeError("Matching service is not initialized")
    return matching_service


def attach_match_image(match: dict) -> dict:
    image_path = Path(str(match["image_path"]))
    if not image_path.is_file():
        raise HTTPException(
            status_code=500,
            detail=f"Matched image file not found: {image_path}",
        )
    payload_path = image_path.parent / ".rgba" / image_path.name
    if not payload_path.is_file():
        payload_path = image_path
    try:
        return {**match, **encode_image_payload(payload_path)}
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Matched image file could not be read: {payload_path}",
        ) from exc


async def read_image(file: UploadFile) -> Image.Image:
    content = await file.read()
    try:
        return Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc


@router.get("/health")
def health():
    service = get_matching_service()
    return {
        "status": "healthy",
        "model_loaded": service.model is not None,
        "device": str(service.device),
        "storage_dir": str(STORAGE_DIR),
    }


@router.get("/metadata")
def metadata():
    service = get_matching_service()
    return {
        "checkpoint_path": str(MATCHING_MODEL_PATH),
        "architecture": "SiameseViT",
        **service.checkpoint_metadata,
    }


@router.get("/gallery/images")
def list_gallery(
    lot_id: Annotated[str | None, Query()] = None,
    capture_date: Annotated[str | None, Query()] = None,
):
    try:
        return get_matching_service().list_gallery(
            lot_id=lot_id,
            capture_date=capture_date,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

@router.post("/gallery/save")
async def add_gallery_save(
    cattle_no: Annotated[str, Form()],
    prod_date: Annotated[str, Form()],
    c_code: Annotated[str, Form()],
):
    image_path = Path(f"/app/storage/rmb2/save/{prod_date}/{cattle_no}/{c_code}_before.png")
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"이미지를 찾을 수 없습니다: {image_path}")

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"허용되지 않은 이미지 포맷입니다. {e}") from e

    try:
        result = get_matching_service().add_gallery_image(
            f"{c_code}_before.png",
            image,
            lot_id=cattle_no,
            capture_date=prod_date,
            preprocessed=False,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    return result



@router.get('/warmup')
async def warmup():
    dummy = Image.new("RGB", (224, 224), color=(128, 128, 128))
    get_matching_service().embed_image(dummy)
    return {"errno": 0}


@router.post("/gallery/images")
async def add_gallery_images(
    files: Annotated[list[UploadFile], File()],
    lot_id: Annotated[str, Form(description="Lot identifier used to scope matching.")],
    preprocess: Annotated[
        bool,
        Form(description="Apply background removal, tilt correction, and crop."),
    ] = True,
    capture_date: Annotated[
        str | None,
        Form(description="Capture date in YYYY-MM-DD. Defaults to server date."),
    ] = None,
):
    service = get_matching_service()
    added = []
    for file in files:
        image = await read_image(file)
        rgba_image = None
        if preprocess:
            image, rgba_image = preprocess_for_matching_with_rgba(image)
        try:
            added.append(
                service.add_gallery_image(
                    file.filename,
                    image,
                    lot_id=lot_id,
                    capture_date=capture_date,
                    preprocessed=preprocess,
                    rgba_image=rgba_image,
                )
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"added": added, "count": len(added)}


@router.post("/gallery/import-directory")
def import_gallery_directory(request: DirectoryImportRequest):
    service = get_matching_service()
    directory = Path(request.directory)
    if not directory.exists() or not directory.is_dir():
        raise HTTPException(status_code=400, detail=f"Directory not found: {directory}")

    added = []
    skipped = []
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        try:
            image = Image.open(path).convert("RGB")
            rgba_image = None
            if request.preprocess:
                image, rgba_image = preprocess_for_matching_with_rgba(image)
            added.append(
                service.add_gallery_image(
                    path.name,
                    image,
                    lot_id=request.lot_id,
                    capture_date=request.capture_date,
                    preprocessed=request.preprocess,
                    rgba_image=rgba_image,
                )
            )
        except Exception as exc:
            skipped.append({"path": str(path), "error": str(exc)})
    return {"added": added, "skipped": skipped}


@router.delete("/gallery/images/{name}")
def remove_gallery_image(
    name: str,
    lot_id: Annotated[str, Query(description="Lot identifier used to scope deletion.")],
    capture_date: Annotated[str | None, Query()] = None,
):
    try:
        removed = get_matching_service().remove_gallery_image(
            name,
            lot_id=lot_id,
            capture_date=capture_date,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not removed:
        raise HTTPException(status_code=404, detail=f"Gallery image not found: {name}")
    return {"removed": name, "lot_id": lot_id, "capture_date": capture_date}


@router.delete("/gallery/images")
def clear_gallery(
    lot_id: Annotated[str, Query(description="Lot identifier used to scope deletion.")],
    capture_date: Annotated[str | None, Query()] = None,
):
    try:
        removed_count = get_matching_service().clear_gallery(
            lot_id=lot_id,
            capture_date=capture_date,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"removed_count": removed_count, "lot_id": lot_id, "capture_date": capture_date}


@router.post("/match")
async def match_image(
    file: Annotated[UploadFile, File()],
    lot_id: Annotated[str, Query(description="Lot identifier used to scope matching.")],
    top_k: Annotated[int, Query(ge=1, le=50)] = DEFAULT_TOP_K,
    preprocess: Annotated[
        bool,
        Query(description="Apply background removal, tilt correction, and crop."),
    ] = True,
    capture_date: Annotated[str | None, Query()] = None,
):
    image = await read_image(file)
    preprocess_ms = 0.0
    if preprocess:
        preprocess_start = perf_counter()
        image, _ = preprocess_for_matching_with_rgba(image)
        preprocess_ms = (perf_counter() - preprocess_start) * 1000.0
    compute_start = perf_counter()
    try:
        matches = get_matching_service().find_matches(
            image,
            top_k=top_k,
            lot_id=lot_id,
            capture_date=capture_date,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    query_compute_ms = (perf_counter() - compute_start) * 1000.0
    if not matches:
        raise HTTPException(status_code=404, detail="Gallery scope is empty")
    matches = [attach_match_image(matches[0]), *matches[1:]]
    return {
        "query_file": file.filename,
        "lot_id": lot_id,
        "capture_date": capture_date,
        "top_k": min(top_k, len(matches)),
        "preprocess": preprocess,
        "preprocess_ms": round(preprocess_ms, 1),
        "query_compute_ms": query_compute_ms,
        "matches": matches,
    }

class matchImageSaveRequest(BaseModel):
	image_name: str
	cattle_no: str
	prod_date: str

@router.post("/match/save")
async def match_image_save(body: matchImageSaveRequest):

	image_path 	= Path(f"/app/storage/rmb2/save/{body.prod_date}/{body.cattle_no}/{body.image_name}")
    
	print(f"{image_path}")

	image = Image.open(image_path).convert("RGB")
	try:
		matches = get_matching_service().find_matches(
			image,
			top_k=1,
			lot_id=body.cattle_no,
			capture_date=body.prod_date
		)
	except ValueError as exc:
		raise HTTPException(status_code=422, detail=str(exc)) from exc

	if not matches:
		raise HTTPException(status_code=404, detail="갤러리에 맞는 이미지가 없음")
	return {
		"errno": 0,
		"message": "성공",
		"matches": matches
	}
	
