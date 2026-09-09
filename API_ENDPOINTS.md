# Hanwoo Vision API Endpoints

Matching base URL:

```text
http://localhost:8888
```

Dinomaly base URL:

```text
http://localhost:8890
```

Current implemented services:

- Matching gallery enrollment/query on host port `8888`.
- Dinomaly anomaly inference on host port `8890`, also served on `8889`.

## Authentication

All endpoints except `/health`, `/docs`, `/redoc`, and `/openapi.json` require
an API key:

```bash
export HANWOO_API_KEY="change-me"
curl -H "X-API-Key: $HANWOO_API_KEY" "http://localhost:8888/metadata"
```

Gallery data is scoped by `lot_id`. Images are stored under
`storage/matching/gallery_images/{lot_id}/{capture_date}`. Embeddings and
metadata are stored in Qdrant collection `hanwoo_matching_gallery`.

## Common Params

| Name | Location | Required | Type | Description |
| --- | --- | --- | --- | --- |
| `lot_id` | form/query/body | Yes for enroll, match, delete, clear | string | Lot identifier used to separate gallery pools. Same filename can exist in different lots. |
| `capture_date` | form/query/body | No | string, `YYYY-MM-DD` | Narrows gallery scope by date. During upload, defaults to server date when omitted. |
| `preprocess` | form/query/body | No | boolean | Applies background removal, tilt correction, and crop. |
| `top_k` | query | No | integer, 1-50 | Number of nearest matches to return. Only the top match includes base64 image data. Defaults to server config. |

## GET `/health`

Checks service status and model/device state.

### Params

None.

### Example

```bash
curl "http://localhost:8888/health"
```

### Response Example

```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "storage_dir": "/app/storage/matching"
}
```

## GET `/metadata`

Returns matching checkpoint and model metadata.

### Params

None.

### Example

```bash
curl -H "X-API-Key: $HANWOO_API_KEY" "http://localhost:8888/metadata"
```

### Response Example

```json
{
  "checkpoint_path": "/app/models/matching/encoder.pt",
  "architecture": "SiameseViT",
  "backbone": "swin",
  "embedding_dim": 256,
  "image_size": 224,
  "epoch": 10,
  "metrics": {}
}
```

## GET `/gallery/images`

Lists enrolled gallery images. Can list all images or filter by lot/date.

### Params

| Name | Location | Required | Type | Description |
| --- | --- | --- | --- | --- |
| `lot_id` | query | No | string | Return images only from this lot. |
| `capture_date` | query | No | string, `YYYY-MM-DD` | Return images only from this date. |

### Examples

List all gallery images:

```bash
curl -H "X-API-Key: $HANWOO_API_KEY" "http://localhost:8888/gallery/images"
```

List one lot:

```bash
curl -H "X-API-Key: $HANWOO_API_KEY" "http://localhost:8888/gallery/images?lot_id=LOT-001"
```

List one lot and date:

```bash
curl -H "X-API-Key: $HANWOO_API_KEY" "http://localhost:8888/gallery/images?lot_id=LOT-001&capture_date=2026-06-22"
```

### Response Example

```json
{
  "count": 1,
  "filenames": ["before_packaging"],
  "images": [
    {
      "name": "before_packaging",
      "lot_id": "LOT-001",
      "capture_date": "2026-06-22",
      "image_path": "/app/storage/matching/gallery_images/LOT-001/2026-06-22/before_packaging.png",
      "original_filename": "before_packaging.jpg",
      "preprocessed": false,
      "created_at": "2026-06-22T01:23:45+00:00"
    }
  ]
}
```

## POST `/gallery/images`

Uploads one or more images into a lot-scoped gallery. Each image creates two
Qdrant vectors: original orientation and rotated orientation.

### Params

Multipart form-data.

| Name | Location | Required | Type | Description |
| --- | --- | --- | --- | --- |
| `files` | form file | Yes | one or more image files | Images to enroll. |
| `lot_id` | form | Yes | string | Lot identifier used to scope matching. |
| `capture_date` | form | No | string, `YYYY-MM-DD` | Date folder and metadata. Defaults to server date. |
| `preprocess` | form | No | boolean | Defaults to `true`. |

### Example

```bash
curl -X POST "http://localhost:8888/gallery/images" \
  -H "X-API-Key: $HANWOO_API_KEY" \
  -F "lot_id=LOT-001" \
  -F "capture_date=2026-06-22" \
  -F "preprocess=false" \
  -F "files=@before_packaging.jpg"
```

Multiple files:

```bash
curl -X POST "http://localhost:8888/gallery/images" \
  -H "X-API-Key: $HANWOO_API_KEY" \
  -F "lot_id=LOT-001" \
  -F "capture_date=2026-06-22" \
  -F "preprocess=true" \
  -F "files=@image_1.jpg" \
  -F "files=@image_2.jpg"
```

### Response Example

```json
{
  "added": [
    {
      "name": "before_packaging",
      "lot_id": "LOT-001",
      "capture_date": "2026-06-22",
      "path": "/app/storage/matching/gallery_images/LOT-001/2026-06-22/before_packaging.png"
    }
  ],
  "count": 1
}
```

## POST `/gallery/import-directory`

Imports every file in a server-side directory into a lot-scoped gallery.

Important: `directory` must exist inside the API container or be mounted into it.

### Params

JSON body.

| Name | Location | Required | Type | Description |
| --- | --- | --- | --- | --- |
| `directory` | body | Yes | string | Directory path visible to API container. |
| `lot_id` | body | Yes | string | Lot identifier used to scope matching. |
| `capture_date` | body | No | string, `YYYY-MM-DD` | Date folder and metadata. Defaults to server date. |
| `preprocess` | body | No | boolean | Defaults to `true`. |

### Example

```bash
curl -X POST "http://localhost:8888/gallery/import-directory" \
  -H "X-API-Key: $HANWOO_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "directory": "/app/data/lot_001",
    "lot_id": "LOT-001",
    "capture_date": "2026-06-22",
    "preprocess": true
  }'
```

### Response Example

```json
{
  "added": [
    {
      "name": "image_1",
      "lot_id": "LOT-001",
      "capture_date": "2026-06-22",
      "path": "/app/storage/matching/gallery_images/LOT-001/2026-06-22/image_1.png"
    }
  ],
  "skipped": []
}
```

## DELETE `/gallery/images/{name}`

Deletes one enrolled image from one lot. If `capture_date` is omitted, deletes
matching `name` entries across all dates in that lot.

### Params

| Name | Location | Required | Type | Description |
| --- | --- | --- | --- | --- |
| `name` | path | Yes | string | Sanitized image name returned by enroll/list. No extension. |
| `lot_id` | query | Yes | string | Lot identifier used to scope deletion. |
| `capture_date` | query | No | string, `YYYY-MM-DD` | Delete only this date. |

### Examples

Delete from a lot:

```bash
curl -X DELETE "http://localhost:8888/gallery/images/before_packaging?lot_id=LOT-001" \
  -H "X-API-Key: $HANWOO_API_KEY"
```

Delete from a lot/date:

```bash
curl -X DELETE "http://localhost:8888/gallery/images/before_packaging?lot_id=LOT-001&capture_date=2026-06-22" \
  -H "X-API-Key: $HANWOO_API_KEY"
```

### Response Example

```json
{
  "removed": "before_packaging",
  "lot_id": "LOT-001",
  "capture_date": "2026-06-22"
}
```

## DELETE `/gallery/images`

Clears a scoped gallery. `lot_id` is required. If `capture_date` is supplied,
only that date is cleared; otherwise the whole lot is cleared.

### Params

| Name | Location | Required | Type | Description |
| --- | --- | --- | --- | --- |
| `lot_id` | query | Yes | string | Lot identifier used to scope deletion. |
| `capture_date` | query | No | string, `YYYY-MM-DD` | Clear only this date. |

### Examples

Clear one lot:

```bash
curl -X DELETE "http://localhost:8888/gallery/images?lot_id=LOT-001" \
  -H "X-API-Key: $HANWOO_API_KEY"
```

Clear one date in one lot:

```bash
curl -X DELETE "http://localhost:8888/gallery/images?lot_id=LOT-001&capture_date=2026-06-22" \
  -H "X-API-Key: $HANWOO_API_KEY"
```

### Response Example

```json
{
  "removed_count": 12,
  "lot_id": "LOT-001",
  "capture_date": null
}
```

## POST `/match`

Matches a query image against the gallery for the requested lot. If
`capture_date` is supplied, matching is restricted to that lot/date. The
response returns up to `top_k` matches with image paths; only the first match
includes the full image as transparent RGBA PNG base64.
The transparent image is generated during gallery preprocessing and does not
change the RGB image used for embeddings.

### Params

Multipart form-data plus query params.

| Name | Location | Required | Type | Description |
| --- | --- | --- | --- | --- |
| `file` | form file | Yes | image file | Query image. |
| `lot_id` | query | Yes | string | Lot identifier used to scope matching. |
| `capture_date` | query | No | string, `YYYY-MM-DD` | Search only this date. |
| `top_k` | query | No | integer, 1-50 | Number of matches to return. Only rank 1 includes transparent RGBA PNG bytes. |
| `preprocess` | query | No | boolean | Defaults to `true`. Applies background removal, tilt correction, and crop. |

### Examples

Match against one lot:

```bash
curl -X POST "http://localhost:8888/match?lot_id=LOT-001&top_k=5" \
  -H "X-API-Key: $HANWOO_API_KEY" \
  -F "file=@after_packaging.jpg"
```

Match against one lot/date:

```bash
curl -X POST "http://localhost:8888/match?lot_id=LOT-001&capture_date=2026-06-22&top_k=5" \
  -H "X-API-Key: $HANWOO_API_KEY" \
  -F "file=@after_packaging.jpg"
```

### Response Example

```json
{
  "query_file": "after_packaging.jpg",
  "lot_id": "LOT-001",
  "capture_date": "2026-06-22",
  "top_k": 1,
  "preprocess": true,
  "matches": [
    {
      "rank": 1,
      "name": "before_packaging",
      "lot_id": "LOT-001",
      "capture_date": "2026-06-22",
      "distance": 0.0,
      "similarity": 100.0,
      "image_path": "/app/storage/matching/gallery_images/LOT-001/2026-06-22/before_packaging.png",
      "image_mime_type": "image/png",
      "image_size_bytes": 123456,
      "image_base64": "iVBORw0KGgo...",
      "matched_variant": "original"
    }
  ]
}
```

## POST `/validate`

Runs a whole benchmark ZIP server-side: builds the gallery from
`<gallery_folder>/<date>/*.jpg`, then matches every image in
`<query_folder>/<date>/*.jpg`. A query is correct when the top-1 match shares
its filename stem. Each date folder is scored independently, so the gallery for
`lot_id` is cleared between dates and emptied when the run ends — point it at a
scratch lot, never a production one.

The ZIP may wrap the folders in any number of parent directories. Images sitting
directly in the gallery/query folder (no date subfolder) are treated as one group.

### Params

Multipart form-data plus query params.

| Name | Location | Required | Type | Description |
| --- | --- | --- | --- | --- |
| `file` | form file | Yes | zip file | Benchmark archive. |
| `lot_id` | query | No | string | Scratch lot for the run. Defaults to `validator-test`. Its gallery is wiped before and after. |
| `top_k` | query | No | integer, 1-50 | Matches kept per query. Defaults to `DEFAULT_TOP_K`. |
| `preprocess` | query | No | boolean | Defaults to `true`. |
| `gallery_folder` | query | No | string | Defaults to `test/after`. |
| `query_folder` | query | No | string | Defaults to `test/before`. |

### Example

```bash
curl -X POST "http://localhost:8888/validate?lot_id=validator-test&top_k=5" \
  -H "X-API-Key: $HANWOO_API_KEY" \
  -F "file=@benchmark.zip"
```

### Response Example

```json
{
  "filename": "benchmark.zip",
  "lot_id": "validator-test",
  "top_k": 5,
  "preprocess": true,
  "gallery_folder": "test/after",
  "query_folder": "test/before",
  "metrics": {
    "accuracy": 1.0,
    "correct": 2,
    "total": 2,
    "gallery_total": 2,
    "dates": 1,
    "avg_preprocess_ms": 22.8,
    "avg_match_ms": 66.0,
    "avg_total_ms": 88.7,
    "elapsed_s": 0.6
  },
  "per_date": [
    {"date": "2026-06-29", "gallery": 2, "queries": 2, "correct": 2, "accuracy": 1.0, "elapsed_s": 0.2}
  ],
  "results": [
    {
      "date": "2026-06-29",
      "query_image": "001709.png",
      "expected": "001709",
      "top1": "001709",
      "correct": true,
      "matches": [
        {"rank": 1, "name": "001709", "similarity": 100.0, "distance": 0.0, "matched_variant": "original"}
      ],
      "preprocess_ms": 22.9,
      "match_ms": 67.5,
      "total_ms": 90.4
    }
  ],
  "system": { "...": "see GET /system" }
}
```

## GET `/system`

Reports what the process actually loaded: weight checksums, device, GPU, driver,
and library versions. Use it to confirm two hosts are running the same thing.
`/validate` embeds the same block under `system`.

### Example

```bash
curl "http://localhost:8888/system" -H "X-API-Key: $HANWOO_API_KEY"
```

### Response Example

```json
{
  "service": "matching",
  "host": "d4ad16cb1b82",
  "pid": 1,
  "python": "3.12.3",
  "platform": "Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39",
  "hardware": {
    "device": "cuda",
    "device_type": "cuda",
    "cuda_available": true,
    "torch_version": "2.13.0+cu130",
    "torch_cuda_version": "13.0",
    "cudnn_version": 92000,
    "gpu_name": "NVIDIA GeForce RTX 5070",
    "gpu_index": 0,
    "gpu_count": 1,
    "gpu_capability": "12.0",
    "gpu_total_memory_mb": 12227,
    "gpu_memory_allocated_mb": 597,
    "gpu_memory_reserved_mb": 1876,
    "driver_version": "591.86",
    "smi_gpu_name": "NVIDIA GeForce RTX 5070",
    "smi_gpu_memory": "12227 MiB"
  },
  "weights": {
    "matching": {
      "path": "/app/models/matching/best_model.pth",
      "exists": true,
      "size_bytes": 1049832199,
      "modified": "2026-08-14T04:52:54.816793+00:00",
      "sha256": "7a78775826e8..."
    },
    "u2net": {"path": "/app/models/u2net/u2net.onnx", "exists": true, "size_bytes": 175997641, "modified": "2026-06-18T06:06:13.821399+00:00", "sha256": "8d10d2f3bb75..."}
  },
  "packages": {"torch": "2.13.0", "torchvision": "0.28.0", "timm": "1.0.28", "numpy": "2.5.2", "opencv-python-headless": "5.0.0.93", "pillow": "12.3.0", "onnxruntime-gpu": "1.28.0", "qdrant-client": "1.19.0", "fastapi": "0.141.1"},
  "model_loaded": true,
  "config": {
    "architecture": "SiameseViT",
    "gallery_dir": "/app/storage/matching/gallery_images",
    "storage_dir": "/app/storage/matching",
    "backbone": "swin",
    "embedding_dim": 256,
    "image_size": 224,
    "epoch": 13,
    "metrics": {"top1_acc": 1.0, "top5_acc": 1.0}
  },
  "qdrant": {"url": "http://qdrant:6333", "collection": "hanwoo_matching_gallery", "reachable": true, "points": 412}
}
```

GPU fields and `driver_version` appear only when CUDA is available. A checkpoint
that is missing reports `{"path": "...", "exists": false}`, and an unreachable
Qdrant reports `"reachable": false` with the error.

## Error Responses

Missing or invalid params usually return `422`.

```json
{
  "detail": "lot_id is required"
}
```

Invalid image uploads return `400`.

```json
{
  "detail": "Invalid image: cannot identify image file"
}
```

Empty matching scope returns `404`.

```json
{
  "detail": "Gallery scope is empty"
}
```

# Dinomaly Endpoints

Dinomaly runs separately on host port `8890`, and also answers on `8889`, the
retired anomaly service's port.

```text
http://localhost:8890
```

It requires `models/dinomaly/best_model.pth` and downloads its DINOv2 backbone
on first use. If the checkpoint is missing or invalid, the container stays up
and `/health` returns `not_loaded`.

## GET `/health`

Checks model, threshold, score mode, and runtime device.

### Example

```bash
curl "http://localhost:8890/health"
```

### Response Example

```json
{
  "status": "healthy",
  "model_loaded": true,
  "threshold": 0.192822,
  "score_mode": "roi_topk",
  "device": "cuda"
}
```

Not loaded example:

```json
{
  "status": "not_loaded",
  "model_loaded": false,
  "threshold": null,
  "score_mode": null,
  "device": null
}
```

## POST `/infer`

Runs anomaly detection on one uploaded image. An image is anomalous when its
score is greater than or equal to the active threshold.

### Params

Multipart form-data plus query params.

| Name | Location | Required | Type | Description |
| --- | --- | --- | --- | --- |
| `file` | form file | Yes | image file | Hanwoo image to inspect. |
| `preprocess` | query | No | boolean | Defaults to `true`. Applies background removal, tilt correction, and crop. Send `false` only for images whose background is already removed. |
| `heatmap` | query | No | boolean | Defaults to `true`. Returns a base64 PNG overlay. |

### Example

```bash
curl -X POST "http://localhost:8890/infer?preprocess=true&heatmap=false" \
  -H "X-API-Key: $HANWOO_API_KEY" \
  -F "file=@sample.jpg"
```

### Response Example

```json
{
  "filename": "sample.jpg",
  "preprocess": true,
  "heatmap": false,
  "preprocess_ms": 516.0,
  "infer_ms": 110.6,
  "total_ms": 626.6,
  "anomaly_score": 0.1809,
  "is_anomaly": false,
  "threshold": 0.1928,
  "score_mode": "roi_topk",
  "score_details": {
    "full_image_score": 0.1943,
    "roi_score": 0.1892,
    "roi_topk_score": 0.1809
  },
  "infer_timings_ms": {
    "mask": 73.8,
    "transform": 9.5,
    "forward": 12.2,
    "cosine_loop": 11.2,
    "gaussian": 1.9,
    "score": 0.2,
    "_compute_total": 108.9,
    "predict_total": 108.9
  }
}
```

With `heatmap=true` the response also carries `heatmap_b64`, a base64 PNG of
the anomaly overlay.

`score_details` always carries all three scores; `score_mode` selects which one
becomes `anomaly_score`.

## GET `/threshold`

Returns the active threshold.

### Example

```bash
curl -H "X-API-Key: $HANWOO_API_KEY" "http://localhost:8890/threshold"
```

### Response Example

```json
{
  "threshold": 0.192822
}
```

## PUT `/threshold`

Updates the active threshold. The change lasts until the service restarts,
which restores `DINOMALY_THRESHOLD`.

### Params

JSON body.

| Name | Location | Required | Type | Description |
| --- | --- | --- | --- | --- |
| `threshold` | body | Yes | float | Must be greater than `0`. |

### Example

```bash
curl -X PUT "http://localhost:8890/threshold" \
  -H "X-API-Key: $HANWOO_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"threshold": 0.192822}'
```

### Response Example

```json
{
  "threshold": 0.192822,
  "updated": true
}
```

## GET `/score-mode`

Returns the active score mode.

### Example

```bash
curl -H "X-API-Key: $HANWOO_API_KEY" "http://localhost:8890/score-mode"
```

### Response Example

```json
{
  "score_mode": "roi_topk"
}
```

## PUT `/score-mode`

Sets which score drives the verdict. The change lasts until the service
restarts, which restores `DINOMALY_SCORE_MODE`.

The three modes sit on different scales, so a threshold calibrated for one is
meaningless for another. `0.192822` is calibrated for `roi_topk`.

### Params

JSON body.

| Name | Location | Required | Type | Description |
| --- | --- | --- | --- | --- |
| `score_mode` | body | Yes | string | One of `roi_topk`, `roi_max`, `full`. Anything else returns `422`. |

### Example

```bash
curl -X PUT "http://localhost:8890/score-mode" \
  -H "X-API-Key: $HANWOO_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"score_mode": "roi_topk"}'
```

### Response Example

```json
{
  "score_mode": "roi_topk",
  "updated": true
}
```

## POST `/validate`

Scores a whole benchmark ZIP server-side, in-process: every image under
`abnormal/` is expected to be an anomaly, every image under `good/` normal. The
folders may sit at any depth in the archive. Heatmaps are never returned here —
`heatmap=true` only makes the run pay the same cost per image.

`threshold` and `score_mode` override the service settings for this run only;
the previous values are restored when it finishes. Use `PUT /threshold` and
`PUT /score-mode` to change them for good.

### Params

Multipart form-data plus query params.

| Name | Location | Required | Type | Description |
| --- | --- | --- | --- | --- |
| `file` | form file | Yes | zip file | Archive with `abnormal/` and `good/` folders. |
| `preprocess` | query | No | boolean | Defaults to `true`. |
| `heatmap` | query | No | boolean | Defaults to `false`. Computes heatmaps for timing only. |
| `warmup` | query | No | integer | Defaults to `3`. Untimed inferences run first so CUDA warm-up stays out of the averages. |
| `threshold` | query | No | float | Overrides the threshold for this run. |
| `score_mode` | query | No | string | One of `roi_topk`, `roi_max`, `full`. Overrides the mode for this run. |

### Example

```bash
curl -X POST "http://localhost:8890/validate?preprocess=true&score_mode=roi_topk" \
  -H "X-API-Key: $HANWOO_API_KEY" \
  -F "file=@benchmark.zip"
```

### Response Example

```json
{
  "filename": "benchmark.zip",
  "preprocess": true,
  "heatmap": false,
  "warmup": 3,
  "threshold": 0.192822,
  "score_mode": "roi_topk",
  "metrics": {
    "accuracy": 0.955,
    "correct": 191,
    "precision": 0.96,
    "recall": 0.95,
    "f1": 0.955,
    "specificity": 0.96,
    "confusion": {"tp": 95, "fp": 4, "fn": 5, "tn": 96},
    "auroc": 0.9846,
    "total": 200,
    "n_anomaly": 100,
    "n_normal": 100,
    "n_preprocess_failed": 0,
    "avg_preprocess_ms": 23.1,
    "avg_infer_ms": 51.4,
    "avg_total_ms": 74.5,
    "elapsed_s": 15.1
  },
  "results": [
    {
      "image": "001709.png",
      "score": 0.4582,
      "score_details": {"full_image_score": 0.7403, "roi_score": 0.6246, "roi_topk_score": 0.4582},
      "predicted": "anomaly",
      "expected": "anomaly",
      "correct": true,
      "preprocess_failed": false,
      "preprocess_ms": 23.2,
      "infer_ms": 51.7,
      "total_ms": 74.9
    }
  ],
  "system": { "...": "see GET /system" }
}
```

`auroc` is present only when both classes are in the ZIP. Images whose
preprocessing fails are still scored on the raw image and flagged with
`preprocess_failed`.

## GET `/system`

Same block as the matching service's `/system`, with `weights.dinomaly`,
`weights.u2net`, and a Dinomaly `config`:

```json
{
  "service": "dinomaly",
  "model_loaded": true,
  "config": {
    "encoder_name": "dinov2reg_vit_base_14",
    "threshold": 0.192822,
    "score_mode": "roi_topk",
    "score_topk_ratio": 0.01,
    "image_size": 448,
    "crop_size": 392
  }
}
```

```bash
curl "http://localhost:8890/system" -H "X-API-Key: $HANWOO_API_KEY"
```

Dinomaly has no `/evaluate`. Batch evaluation runs either server-side through
`/validate`, or client-side: the validator's Dinomaly tab scores a zip of
`abnormal/` and `good/` folders one image at a time through `/infer`.

