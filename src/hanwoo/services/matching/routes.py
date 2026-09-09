from __future__ import annotations

import io
from time import perf_counter
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from PIL import Image

from hanwoo.core.config import (
    DEFAULT_TOP_K,
    MATCHING_MODEL_PATH,
    STORAGE_DIR,
    U2NET_HOME,
)
from hanwoo.core.image_payload import encode_image_payload
from hanwoo.core.preprocessing import preprocess_for_matching_with_rgba
from hanwoo.core.schemas import DirectoryImportRequest
from hanwoo.core.sysinfo import system_info
from hanwoo.core.zip_dataset import extracted_zip, find_subtree, group_by_subdir
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


def _matching_system_info(service: MatchingService) -> dict:
    qdrant: dict = {
        "url": service.qdrant_url,
        "collection": service.qdrant_collection,
    }
    try:
        if service.store is None:
            raise RuntimeError("Qdrant gallery store is not initialized")
        info = service.store.client.get_collection(service.qdrant_collection)
        qdrant |= {"reachable": True, "points": info.points_count}
    except Exception as exc:
        qdrant |= {"reachable": False, "error": str(exc)}
    return system_info(
        service="matching",
        device=service.device,
        weights={
            "matching": service.model_path,
            "u2net": U2NET_HOME / "u2net.onnx",
        },
        model_loaded=service.model is not None,
        config={
            "architecture": "SiameseViT",
            "gallery_dir": str(service.gallery_dir),
            "storage_dir": str(STORAGE_DIR),
            **service.checkpoint_metadata,
        },
        qdrant=qdrant,
    )


@router.get("/system")
def system():
    return _matching_system_info(get_matching_service())


@router.post("/validate")
async def validate(
    file: Annotated[UploadFile, File(description="ZIP holding <gallery_folder>/<date>/*.jpg and <query_folder>/<date>/*.jpg.")],
    lot_id: Annotated[str, Query(description="Scratch lot used for the run. Its gallery is wiped before and after.")] = "validator-test",
    top_k: Annotated[int, Query(ge=1, le=50)] = DEFAULT_TOP_K,
    preprocess: Annotated[bool, Query()] = True,
    gallery_folder: Annotated[str, Query()] = "test/after",
    query_folder: Annotated[str, Query()] = "test/before",
):
    """Run a whole benchmark ZIP in-process: build the gallery per date dir, then match every query.

    A query is correct when the top-1 gallery match carries the same filename stem.
    Each date dir is matched independently, so `lot_id`'s gallery is cleared between
    dates and emptied when the run ends. Do not point it at a production lot.
    """
    service = get_matching_service()

    with extracted_zip(file) as root:
        gallery_root = find_subtree(root, gallery_folder)
        query_root = find_subtree(root, query_folder)
        if gallery_root is None or query_root is None:
            raise HTTPException(
                status_code=422,
                detail=f"ZIP must contain {gallery_folder}/ and {query_folder}/ folders.",
            )
        gallery_by_date = group_by_subdir(gallery_root)
        query_by_date = group_by_subdir(query_root)
        dates = sorted(set(gallery_by_date) & set(query_by_date))
        if not dates:
            raise HTTPException(
                status_code=422,
                detail=f"No shared date folders under {gallery_folder}/ and {query_folder}/.",
            )

        rows: list[dict] = []
        per_date: list[dict] = []
        gallery_total = 0
        start = perf_counter()
        try:
            for capture_date in dates:
                gallery = gallery_by_date[capture_date]
                stems = {MatchingService.safe_stem(path.name) for path in gallery}
                queries = [
                    path for path in query_by_date[capture_date]
                    if MatchingService.safe_stem(path.name) in stems
                ]
                if not queries:
                    continue

                service.clear_gallery(lot_id)
                for path in gallery:
                    image = Image.open(path).convert("RGB")
                    rgba_image = None
                    if preprocess:
                        image, rgba_image = preprocess_for_matching_with_rgba(image)
                    service.add_gallery_image(
                        path.name,
                        image,
                        lot_id=lot_id,
                        preprocessed=preprocess,
                        rgba_image=rgba_image,
                    )
                gallery_total += len(gallery)

                date_start = perf_counter()
                correct = 0
                for path in queries:
                    expected = MatchingService.safe_stem(path.name)
                    image = Image.open(path).convert("RGB")
                    t0 = perf_counter()
                    if preprocess:
                        image, _ = preprocess_for_matching_with_rgba(image)
                    preprocess_ms = (perf_counter() - t0) * 1000.0

                    t0 = perf_counter()
                    matches = service.find_matches(image, top_k=top_k, lot_id=lot_id)
                    match_ms = (perf_counter() - t0) * 1000.0

                    top1 = matches[0] if matches else {}
                    is_correct = top1.get("name") == expected
                    correct += int(is_correct)
                    rows.append({
                        "date": capture_date,
                        "query_image": path.name,
                        "expected": expected,
                        "top1": top1.get("name"),
                        "correct": is_correct,
                        "matches": [
                            {
                                "rank": match["rank"],
                                "name": match["name"],
                                "similarity": match["similarity"],
                                "distance": match["distance"],
                                "matched_variant": match["matched_variant"],
                            }
                            for match in matches
                        ],
                        "preprocess_ms": round(preprocess_ms, 1),
                        "match_ms": round(match_ms, 1),
                        "total_ms": round(preprocess_ms + match_ms, 1),
                    })
                per_date.append({
                    "date": capture_date,
                    "gallery": len(gallery),
                    "queries": len(queries),
                    "correct": correct,
                    "accuracy": correct / len(queries) if queries else 0.0,
                    "elapsed_s": round(perf_counter() - date_start, 1),
                })
        finally:
            service.clear_gallery(lot_id)

    total = len(rows)
    correct = sum(row["correct"] for row in rows)
    return {
        "filename": file.filename,
        "lot_id": lot_id,
        "top_k": top_k,
        "preprocess": preprocess,
        "gallery_folder": gallery_folder,
        "query_folder": query_folder,
        "metrics": {
            "accuracy": correct / total if total else 0.0,
            "correct": correct,
            "total": total,
            "gallery_total": gallery_total,
            "dates": len(per_date),
            "avg_preprocess_ms": round(sum(row["preprocess_ms"] for row in rows) / total, 1) if total else 0.0,
            "avg_match_ms": round(sum(row["match_ms"] for row in rows) / total, 1) if total else 0.0,
            "avg_total_ms": round(sum(row["total_ms"] for row in rows) / total, 1) if total else 0.0,
            "elapsed_s": round(perf_counter() - start, 1),
        },
        "per_date": per_date,
        "results": rows,
        "system": _matching_system_info(service),
    }
