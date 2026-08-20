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

Dinomaly has no `/evaluate`. Batch evaluation runs client-side: the validator's
Dinomaly tab scores a zip of `abnormal/` and `good/` folders one image at a
time through `/infer`.
