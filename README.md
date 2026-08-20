# Hanwoo Vision API

FastAPI services for Hanwoo vision workflows. The matching service manages a
lot-scoped gallery, stores gallery embeddings in Qdrant, and matches a query
image against the requested lot. The dinomaly service runs ViTill
reconstruction over DINOv2 features and returns anomaly score, threshold
result, and heatmap overlay.

The earlier PatchCore-style anomaly service is retired. It stays in the repo
behind the `legacy` Compose profile and is not started by default; dinomaly
now serves its port.

Full endpoint reference with parameter tables and curl examples:
[API_ENDPOINTS.md](API_ENDPOINTS.md)

## Current Status

- Matching service is exposed on host port `8888`.
- Dinomaly service is exposed on host ports `8890` and `8889` (the retired
  anomaly service's port).
- The legacy anomaly service is not started by default. Bring it up with
  `docker compose --profile legacy up -d anomaly`, stopping dinomaly first so
  they do not both claim `8889`.
- Qdrant is used as the embedding database.
- Gallery images are stored on disk under a lot/date folder structure.
- GPU runtime is supported through Docker Compose GPU override.
- `HANWOO_DEVICE=auto` uses CUDA when available and falls back to CPU when CUDA
  is not detected.
- API endpoints require `X-API-Key` authentication except health/docs routes.

## Architecture

```text
Client
  |
  v
FastAPI services
  |
  +-- matching: optional preprocessing -> Swin encoder -> Qdrant search
  |
  +-- dinomaly: optional preprocessing -> DINOv2 encoder -> ViTill decoder
  |
  +-- Qdrant: matching vectors and metadata
  |
  +-- disk storage: gallery images and inference artifacts
```

Gallery separation is handled with Qdrant payload filters:

- `lot_id`: required for enroll, match, delete, and clear.
- `capture_date`: optional `YYYY-MM-DD`; defaults to server date on enroll.
- `name`: sanitized image stem.
- `variant`: `original` or rotated variant.

Images are stored at:

```text
storage/matching/gallery_images/{lot_id}/{capture_date}/{name}.png
```

Embeddings and metadata are stored in Qdrant collection:

```text
hanwoo_matching_gallery
```

## Repository Layout

```text
hanwoo-vision-api/
├── docker-compose.yml
├── docker-compose.gpu.yml
├── .env.example
├── pyproject.toml
├── Dockerfile
├── README.md
├── API_ENDPOINTS.md
├── src/hanwoo/
│   ├── core/
│   │   ├── config.py
│   │   ├── preprocessing.py
│   │   ├── encoders/
│   │   ├── vectorstore/
│   │   ├── models.py
│   │   ├── schemas.py
│   │   ├── telemetry.py
│   │   └── gpu.py
│   └── services/
│       ├── matching/
│       ├── dinomaly/
│       └── validator/
├── models/
├── gateway/
├── scripts/
└── tests/
```

## Requirements

Base runtime:

- Docker
- Docker Compose
- Matching checkpoint at `models/matching/encoder.pt`
- U2NET model files under `models/u2net`

GPU runtime:

- NVIDIA GPU
- NVIDIA driver
- NVIDIA Container Toolkit
- Docker Compose with GPU support

## Configuration

Copy the example file when running outside Docker Compose defaults:

```bash
cp .env.example .env
```

Environment variables:

| Name | Default | Description |
| --- | --- | --- |
| `HANWOO_DEVICE` | `auto` | `auto`, `cuda`, or `cpu`. `auto` prefers CUDA and falls back to CPU. |
| `MATCHING_MODEL_PATH` | `/app/models/matching/encoder.pt` | Matching model checkpoint path. |
| `HANWOO_STORAGE_DIR` | `/app/storage/matching` | Runtime storage directory. |
| `U2NET_HOME` | `/app/models/u2net` | Background removal model directory. |
| `DEFAULT_TOP_K` | `5` | Default match result count. |
| `HANWOO_API_KEY` | required | Shared API key required in the `X-API-Key` header. |
| `QDRANT_URL` | `http://qdrant:6333` | Qdrant service URL. |
| `QDRANT_COLLECTION` | `hanwoo_matching_gallery` | Qdrant collection name. |
| `DINOMALY_MODEL_PATH` | `/app/models/dinomaly/best_model.pth` | Dinomaly checkpoint path. |
| `DINOMALY_THRESHOLD` | `0.192822` | Anomaly if score >= threshold. Calibrated on the 200-image held-out set. |
| `DINOMALY_SCORE_MODE` | `roi_topk` | `roi_topk`, `roi_max`, or `full`. Changeable at runtime via `PUT /score-mode`. |
| `DINOMALY_TOP_K_RATIO` | `0.01` | Fraction of ROI pixels averaged for `roi_topk`. |
| `MATCHING_DOWNSCALE` | `1` | Divisor applied before preprocessing. `1` matches how the training set was built; `4` is ~2x faster and costs ~2.5 accuracy points. |

## Model Files

Model weights are not committed to Git. Put them here before running:

```text
models/
├── matching/
│   └── encoder.pt
├── u2net/
│   └── u2net.onnx
└── dinomaly/
    └── best_model.pth
```

The matching service needs `models/matching/encoder.pt`. Background removal
needs U2NET files under `models/u2net`. The dinomaly service needs
`models/dinomaly/best_model.pth`, and downloads its DINOv2 backbone on first
use. Without a checkpoint the container stays up but `/health` reports
`not_loaded`.

The retired anomaly service still ships under the `legacy` Compose profile and
keeps its own `ANOMALY_*` settings and `models/anomaly/` weights; see
`docker-compose.yml` if you need it.

## Matching Benchmark

Held-out set: `hanwoo_matching_benchmark_v2` test split, **145 gallery images
(`test/after`) and 145 queries (`test/before`) across 24 capture dates**, paired
1:1 by filename stem. Measured on an RTX 5070 through the same flow the
validator's Matching tab uses: per date the lot gallery is cleared and
re-uploaded, then each query goes through `POST /match` with `top_k=5` and
`preprocess=true`. A query counts as correct when the rank-1 match name equals
the query filename stem.

| Metric | Value |
| --- | --- |
| **Top-1 accuracy** | **0.9586** (139 / 145) |
| **Top-5 recall** | **1.0000** (145 / 145) |

Per-query latency, mean (median / p95), milliseconds:

| Stage | Mean | Median | p95 |
| --- | --- | --- | --- |
| `preprocess_ms` | 87.8 | 85.8 | 108.0 |
| `query_compute_ms` | 45.8 | 46.2 | 78.7 |
| Round trip incl. upload | 191.1 | 196.2 | 232.3 |

145 queries plus all 24 gallery rebuilds finished in 71.0 s.

Accuracy tracks gallery size. The 23 dates with 1-7 gallery images scored
**74/74 (100%)**; the single date with a 71-image gallery scored **65/71
(91.5%)** and contributed every miss. All six missed queries still had the
correct image inside the top 5, so ranking degrades gracefully rather than
retrieving something unrelated.

The matching service uses `MATCHING_DOWNSCALE=4`, unlike dinomaly. Its gallery
embeddings were built that way, so changing it invalidates every stored vector.

## Dinomaly Benchmark

Held-out set: `hanwoo_anomaly_v3` test split, **200 images (100 abnormal / 100
good)**, 3552x2664 JPEG with background. Defect classes are 천 (cloth), 비닐
(vinyl), 실 (thread), and 뼈 (bone). Measured on an RTX 5070, one image per
request through `POST /infer`, `heatmap=false`, warm.

Settings: `DINOMALY_SCORE_MODE=roi_topk`, `DINOMALY_THRESHOLD=0.192822`,
`preprocess=true`.

| `MATCHING_DOWNSCALE` | Accuracy | Precision | Recall | F1 | TP / FP / FN / TN |
| --- | --- | --- | --- | --- | --- |
| `1` (default) | **0.9550** | 0.9596 | 0.9500 | 0.9548 | 95 / 4 / 5 / 96 |
| `4` | 0.9300 | 0.9388 | 0.9200 | 0.9293 | 92 / 6 / 8 / 94 |

Per-image latency, mean (median / p95), milliseconds:

| `MATCHING_DOWNSCALE` | Preprocess | Infer | Server total | Throughput |
| --- | --- | --- | --- | --- |
| `1` (default) | 520.6 (516 / 560) | 109.9 (104 / 145) | 630.5 (625 / 688) | 1.46 img/s |
| `4` | 90.7 (86 / 121) | 60.0 (52 / 94) | 150.7 (140 / 191) | 4.93 img/s |

`MATCHING_DOWNSCALE=4` is roughly 4x faster and costs about 2.5 accuracy
points, because the v3 training set was preprocessed without that downscale.
Matching the training pipeline matters more than any other tuning here.

Every misclassification in both runs falls within 0.008 of the threshold, in a
band from 0.1817 to 0.1990. The score distribution for the whole set spans only
about 0.02, so treat any change that touches preprocessing as accuracy
affecting until this benchmark says otherwise.

### Preprocessing must be applied exactly once

The same images already background-removed, with `preprocess=false`, reproduce
the numbers above (0.9550 / 0.9596 / 0.9500). Getting the flag wrong in either
direction is catastrophic and does not look like a failure:

| Input | `preprocess` | Accuracy | Specificity |
| --- | --- | --- | --- |
| With background | `true` | 0.9550 | 0.96 |
| Already removed | `false` | 0.9550 | 0.97 |
| Already removed | `true` (double) | 0.2800 | 0.00 |
| With background | `false` (none) | 0.2800 | 0.00 |

In the two broken cases the mean score for abnormal images is 0.3419 and for
good images 0.3420: the classes become indistinguishable and everything is
flagged anomalous, so recall reads 1.00 while specificity is 0.00.

## Run With Docker

CPU or automatic fallback mode:

```bash
docker compose up -d --build
```

GPU mode:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

Start an already-built GPU runtime:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

Check containers:

```bash
docker compose ps
```

Check logs:

```bash
docker compose logs -f matching
```

Stop:

```bash
docker compose down
```

## Verify Runtime

Health check:

```bash
curl "http://localhost:8888/health"
curl "http://localhost:8889/health"
```

Expected GPU response includes:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

Current verified GPU health responses:

```json
{
  "matching": {
    "status": "healthy",
    "model_loaded": true,
    "device": "cuda",
    "storage_dir": "/app/storage/matching"
  },
  "dinomaly": {
    "status": "healthy",
    "model_loaded": true,
    "threshold": 0.192822,
    "score_mode": "roi_topk",
    "device": "cuda"
  }
}
```

Protected endpoints require the configured API key:

```bash
export HANWOO_API_KEY="change-me"
curl -H "X-API-Key: $HANWOO_API_KEY" "http://localhost:8888/metadata"
```

If no GPU is available and `HANWOO_DEVICE=auto`, the service runs on CPU:

```json
{
  "device": "cpu"
}
```

If `HANWOO_DEVICE=cuda` is set and CUDA is unavailable, startup fails instead
of silently using CPU.

## API Summary

Base URL:

```text
matching: http://localhost:8888
dinomaly: http://localhost:8890  (also on 8889)
```

Validator UI:

```text
http://localhost:8501/validator/
```

Use a folder with `before/` query images and `after/` gallery images for
matching accuracy. The page also has a Dinomaly tab that scores a zip of
`abnormal/` and `good/` folders and reports accuracy, precision, and recall.

Matching endpoints:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Check service, model, and device status. |
| `GET` | `/metadata` | Return matching checkpoint/model metadata. |
| `GET` | `/gallery/images` | List gallery images, optionally by lot/date. |
| `POST` | `/gallery/images` | Upload images into a lot-scoped gallery. |
| `POST` | `/gallery/import-directory` | Import server-side directory into a lot-scoped gallery. |
| `DELETE` | `/gallery/images/{name}` | Delete one image by name inside a lot/date scope. |
| `DELETE` | `/gallery/images` | Clear a lot or lot/date gallery scope. |
| `POST` | `/match` | Match one query image against a lot/date gallery scope. |

Dinomaly endpoints:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Check model, threshold, score mode, and device status. |
| `POST` | `/infer` | Run anomaly detection on one uploaded image. |
| `GET` | `/threshold` | Read active threshold. |
| `PUT` | `/threshold` | Update active threshold. |
| `GET` | `/score-mode` | Read active score mode. |
| `PUT` | `/score-mode` | Set score mode to `roi_topk`, `roi_max`, or `full`. |

Legacy anomaly endpoints (`legacy` profile only) match the first four, plus
`POST /evaluate` for server-side test folders. Dinomaly has no `/evaluate`; the
validator's Dinomaly tab evaluates a zip client-side instead.

Common parameters:

| Name | Required | Used By | Description |
| --- | --- | --- | --- |
| `lot_id` | Yes for upload, import, match, delete, clear | Gallery and matching endpoints | Separates gallery pools by production lot. |
| `capture_date` | No | Gallery and matching endpoints | Narrows a lot to one date. Format: `YYYY-MM-DD`. |
| `preprocess` | No | Upload, import, match | Enables background removal and normalization. |
| `top_k` | No | Match | Number of nearest matches to return. Only rank 1 includes transparent RGBA PNG base64 image data. |

## API Examples

Upload one image:

```bash
curl -X POST "http://localhost:8888/gallery/images" \
  -H "X-API-Key: $HANWOO_API_KEY" \
  -F "lot_id=LOT-001" \
  -F "capture_date=2026-06-22" \
  -F "preprocess=true" \
  -F "files=@before_packaging.jpg"
```

Upload multiple images:

```bash
curl -X POST "http://localhost:8888/gallery/images" \
  -H "X-API-Key: $HANWOO_API_KEY" \
  -F "lot_id=LOT-001" \
  -F "capture_date=2026-06-22" \
  -F "files=@image_1.jpg" \
  -F "files=@image_2.jpg"
```

List one lot:

```bash
curl -H "X-API-Key: $HANWOO_API_KEY" "http://localhost:8888/gallery/images?lot_id=LOT-001"
```

List one lot/date:

```bash
curl -H "X-API-Key: $HANWOO_API_KEY" "http://localhost:8888/gallery/images?lot_id=LOT-001&capture_date=2026-06-22"
```

Match a query image against one lot:

```bash
curl -X POST "http://localhost:8888/match?lot_id=LOT-001&top_k=5" \
  -H "X-API-Key: $HANWOO_API_KEY" \
  -F "file=@after_packaging.jpg"
```

Match a query image against one lot/date:

```bash
curl -X POST "http://localhost:8888/match?lot_id=LOT-001&capture_date=2026-06-22&top_k=5" \
  -H "X-API-Key: $HANWOO_API_KEY" \
  -F "file=@after_packaging.jpg"
```

Delete one image from one lot:

```bash
curl -X DELETE "http://localhost:8888/gallery/images/before_packaging?lot_id=LOT-001" \
  -H "X-API-Key: $HANWOO_API_KEY"
```

Clear one lot/date:

```bash
curl -X DELETE "http://localhost:8888/gallery/images?lot_id=LOT-001&capture_date=2026-06-22" \
  -H "X-API-Key: $HANWOO_API_KEY"
```

More examples and response bodies are in [API_ENDPOINTS.md](API_ENDPOINTS.md).

## Data Organization Strategy

Use `lot_id` as the primary gallery partition. Use `capture_date` as a secondary
filter when a lot spans multiple production days.

Recommended naming:

```text
lot_id = LOT-20260622-A
capture_date = 2026-06-22
```

This lets the same image name exist in different lots without collision:

```text
LOT-001/2026-06-22/tray_001.png
LOT-002/2026-06-22/tray_001.png
```

Matching always searches only the requested `lot_id`. Add `capture_date` when
the query must be restricted to one production day.

## Development

Install package in editable mode:

```bash
pip install -e .
```

Run service locally:

```bash
uvicorn hanwoo.services.matching.main:app --host 0.0.0.0 --port 8000
```

Compile check:

```bash
python3 -m compileall src/hanwoo
```

## Troubleshooting

Qdrant not reachable:

```bash
docker compose ps qdrant
docker compose logs qdrant
```

Model missing:

```text
FileNotFoundError: /app/models/matching/encoder.pt
```

Fix by placing the checkpoint at `models/matching/encoder.pt`.

CUDA not used:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
curl "http://localhost:8888/health"
```

If health still reports `cpu`, verify NVIDIA runtime on the host:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## 한국어 안내

Hanwoo Vision API는 한우 이미지 매칭과 이물질 이상탐지를 위한 FastAPI
서비스입니다. 매칭 서비스는 로트별 갤러리 등록, Qdrant 기반 임베딩 검색, 쿼리
이미지 매칭을 담당합니다. dinomaly 서비스는 DINOv2 feature에 ViTill 복원을 적용해 anomaly score,
threshold 결과, heatmap을 반환합니다. 기존 PatchCore 방식 이상탐지 서비스는
사용을 중단했고, `legacy` Compose profile로만 남아 있습니다.

전체 엔드포인트 파라미터와 curl 예시는 [API_ENDPOINTS.md](API_ENDPOINTS.md)에
정리되어 있습니다.

## 현재 구현 상태

- 매칭 서비스는 host port `8888`에서 접근합니다.
- dinomaly 서비스는 host port `8890`과 `8889`(기존 이상탐지 포트)에서
  접근합니다.
- 기존 anomaly 서비스는 기본 실행 대상이 아닙니다. 필요하면 dinomaly를 먼저
  중지한 뒤 `docker compose --profile legacy up -d anomaly`로 실행합니다.
- 임베딩 DB로 Qdrant를 사용합니다.
- 갤러리 원본 이미지는 로컬 디스크에 `lot_id/capture_date` 구조로 저장합니다.
- Docker Compose GPU override로 GPU 실행을 지원합니다.
- `HANWOO_DEVICE=auto`이면 CUDA 사용 가능 시 GPU를 사용하고, 없으면 CPU로
  fallback합니다.

## 데이터 저장 구조

이미지 파일:

```text
storage/matching/gallery_images/{lot_id}/{capture_date}/{name}.png
```

Qdrant 컬렉션:

```text
hanwoo_matching_gallery
```

Qdrant payload 주요 필드:

| 필드 | 설명 |
| --- | --- |
| `lot_id` | 생산 로트 ID. 갤러리를 분리하는 핵심 값입니다. |
| `capture_date` | 촬영일 또는 생산일. `YYYY-MM-DD` 형식입니다. |
| `name` | 이미지 파일명에서 확장자를 제거한 값입니다. |
| `variant` | 원본 방향 또는 회전 방향 임베딩 구분입니다. |

## 실행 방법

CPU 또는 자동 fallback 모드:

```bash
docker compose up -d --build
```

GPU 모드:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

이미 빌드된 GPU 런타임 실행:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

상태 확인:

```bash
curl "http://localhost:8888/health"
curl "http://localhost:8889/health"
```

GPU 사용 중이면 응답에 `"device": "cuda"`가 포함됩니다.

현재 검증된 GPU 응답:

```json
{
  "matching": {
    "status": "healthy",
    "model_loaded": true,
    "device": "cuda",
    "storage_dir": "/app/storage/matching"
  },
  "dinomaly": {
    "status": "healthy",
    "model_loaded": true,
    "threshold": 0.192822,
    "score_mode": "roi_topk",
    "device": "cuda"
  }
}
```

## 설정값

| 이름 | 기본값 | 설명 |
| --- | --- | --- |
| `HANWOO_DEVICE` | `auto` | `auto`, `cuda`, `cpu`. `auto`는 GPU 우선, 없으면 CPU입니다. |
| `MATCHING_MODEL_PATH` | `/app/models/matching/encoder.pt` | 매칭 모델 checkpoint 경로입니다. |
| `HANWOO_STORAGE_DIR` | `/app/storage/matching` | 런타임 저장소 경로입니다. |
| `U2NET_HOME` | `/app/models/u2net` | 배경 제거 모델 경로입니다. |
| `DEFAULT_TOP_K` | `5` | 기본 매칭 결과 개수입니다. |
| `HANWOO_API_KEY` | 필수 | `X-API-Key` 헤더로 전달할 공유 API 키입니다. |
| `QDRANT_URL` | `http://qdrant:6333` | Qdrant 접속 URL입니다. |
| `QDRANT_COLLECTION` | `hanwoo_matching_gallery` | Qdrant 컬렉션 이름입니다. |

## 모델 파일

모델 파일은 Git에 포함하지 않습니다. 실행 전 아래 위치에 배치해야 합니다.

```text
models/
├── matching/
│   └── encoder.pt
├── u2net/
│   └── u2net.onnx
└── dinomaly/
    └── best_model.pth
```

매칭 서비스는 `models/matching/encoder.pt`가 필요합니다. 전처리의 배경 제거를
사용하려면 `models/u2net` 아래에 U2NET 모델이 필요합니다.

## 매칭 성능 평가

평가 데이터: `hanwoo_matching_benchmark_v2` test split, **갤러리 145장
(`test/after`), 쿼리 145장(`test/before`), 촬영일 24개**이며 파일명 stem으로
1:1 대응합니다. validator의 매칭 탭과 동일한 흐름으로 RTX 5070에서
측정했습니다. 날짜별로 lot 갤러리를 비우고 다시 업로드한 뒤, 각 쿼리를
`top_k=5`, `preprocess=true`로 `POST /match`에 보냅니다. rank-1 매칭 이름이 쿼리
파일명 stem과 같으면 정답으로 계산합니다.

| 지표 | 값 |
| --- | --- |
| **Top-1 정확도** | **0.9586** (139 / 145) |
| **Top-5 재현율** | **1.0000** (145 / 145) |

쿼리 1건당 처리 시간, 평균(중앙값 / p95), 단위 ms:

| 단계 | 평균 | 중앙값 | p95 |
| --- | --- | --- | --- |
| `preprocess_ms` | 87.8 | 85.8 | 108.0 |
| `query_compute_ms` | 45.8 | 46.2 | 78.7 |
| 업로드 포함 왕복 | 191.1 | 196.2 | 232.3 |

쿼리 145건과 갤러리 24회 재구축을 합쳐 71.0초가 걸렸습니다.

정확도는 갤러리 크기에 따라 달라집니다. 갤러리가 1~7장인 23개 날짜는
**74/74(100%)**였고, 갤러리가 71장인 날짜 하나가 **65/71(91.5%)**로 모든 오답을
만들었습니다. 오답 6건도 모두 정답 이미지가 top 5 안에 있었습니다.

매칭 서비스는 dinomaly와 달리 `MATCHING_DOWNSCALE=4`를 사용합니다. 갤러리
임베딩이 이 설정으로 생성되었으므로, 값을 바꾸면 저장된 벡터가 모두 무효가
됩니다.

## Dinomaly 성능 평가

평가 데이터: `hanwoo_anomaly_v3` test split, **200장(이상 100 / 정상 100)**,
배경이 있는 3552x2664 JPEG입니다. 결함 종류는 천, 비닐, 실, 뼈입니다. RTX
5070에서 `POST /infer`로 1장씩(`heatmap=false`, warm) 측정했습니다.

설정: `DINOMALY_SCORE_MODE=roi_topk`, `DINOMALY_THRESHOLD=0.192822`,
`preprocess=true`.

| `MATCHING_DOWNSCALE` | 정확도 | 정밀도 | 재현율 | F1 | TP / FP / FN / TN |
| --- | --- | --- | --- | --- | --- |
| `1` (기본값) | **0.9550** | 0.9596 | 0.9500 | 0.9548 | 95 / 4 / 5 / 96 |
| `4` | 0.9300 | 0.9388 | 0.9200 | 0.9293 | 92 / 6 / 8 / 94 |

이미지 1장당 처리 시간, 평균(중앙값 / p95), 단위 ms:

| `MATCHING_DOWNSCALE` | 전처리 | 추론 | 서버 합계 | 처리량 |
| --- | --- | --- | --- | --- |
| `1` (기본값) | 520.6 (516 / 560) | 109.9 (104 / 145) | 630.5 (625 / 688) | 1.46 img/s |
| `4` | 90.7 (86 / 121) | 60.0 (52 / 94) | 150.7 (140 / 191) | 4.93 img/s |

`MATCHING_DOWNSCALE=4`는 약 4배 빠르지만 정확도가 약 2.5%p 낮습니다. v3 학습
데이터가 이 축소 없이 전처리되었기 때문이며, 학습 시 전처리와 동일하게 맞추는
것이 다른 어떤 튜닝보다 중요합니다.

두 실행 모두 오분류 사례가 threshold로부터 0.008 이내(0.1817~0.1990)에
모여 있습니다. 전체 점수 분포 폭이 약 0.02에 불과하므로, 전처리를 건드리는
변경은 이 평가로 확인하기 전까지 정확도에 영향이 있다고 보아야 합니다.

### 전처리는 정확히 한 번만 적용해야 합니다

이미 배경이 제거된 동일 이미지를 `preprocess=false`로 평가하면 위와 같은 결과
(0.9550 / 0.9596 / 0.9500)가 나옵니다. 이 값을 반대로 주면 실패처럼 보이지
않으면서 결과가 완전히 무너집니다:

| 입력 | `preprocess` | 정확도 | 특이도 |
| --- | --- | --- | --- |
| 배경 있음 | `true` | 0.9550 | 0.96 |
| 배경 제거됨 | `false` | 0.9550 | 0.97 |
| 배경 제거됨 | `true` (이중 전처리) | 0.2800 | 0.00 |
| 배경 있음 | `false` (전처리 없음) | 0.2800 | 0.00 |

잘못된 두 경우에는 이상 이미지 평균 점수가 0.3419, 정상 이미지가 0.3420으로
두 분포가 구분되지 않아 전부 이상으로 판정됩니다. 재현율은 1.00으로 보이지만
특이도는 0.00입니다.

## API 요약

기본 URL:

```text
matching: http://localhost:8888
dinomaly: http://localhost:8890  (8889에서도 접근 가능)
```

매칭 엔드포인트:

| Method | Path | 설명 |
| --- | --- | --- |
| `GET` | `/health` | 서비스, 모델, device 상태를 확인합니다. |
| `GET` | `/metadata` | 매칭 모델 checkpoint와 metadata를 반환합니다. |
| `GET` | `/gallery/images` | 갤러리 이미지를 조회합니다. lot/date 필터 가능. |
| `POST` | `/gallery/images` | 이미지를 특정 lot 갤러리에 업로드합니다. |
| `POST` | `/gallery/import-directory` | 서버 내부 디렉터리 이미지를 lot 갤러리에 등록합니다. |
| `DELETE` | `/gallery/images/{name}` | 특정 lot/date 안의 이미지 1개를 삭제합니다. |
| `DELETE` | `/gallery/images` | 특정 lot 또는 lot/date 갤러리를 비웁니다. |
| `POST` | `/match` | 쿼리 이미지를 특정 lot/date 갤러리와 매칭합니다. |

dinomaly 엔드포인트:

| Method | Path | 설명 |
| --- | --- | --- |
| `GET` | `/health` | 모델, threshold, score mode, device 상태를 확인합니다. |
| `POST` | `/infer` | 이미지 1장에 대해 이상탐지를 수행합니다. |
| `GET` | `/threshold` | 현재 threshold를 조회합니다. |
| `PUT` | `/threshold` | threshold를 변경합니다. |
| `GET` | `/score-mode` | 현재 score mode를 조회합니다. |
| `PUT` | `/score-mode` | score mode를 `roi_topk`, `roi_max`, `full` 중 하나로 설정합니다. |

기존 anomaly 엔드포인트(`legacy` profile)는 위 4개와 `POST /evaluate`를
제공합니다. dinomaly에는 `/evaluate`가 없고, validator의 Dinomaly 탭이 zip을
클라이언트 측에서 평가합니다.

주요 파라미터:

| 이름 | 필수 여부 | 설명 |
| --- | --- | --- |
| `lot_id` | 등록, 매칭, 삭제에서 필수 | 갤러리를 로트별로 분리합니다. |
| `capture_date` | 선택 | 날짜 단위로 추가 필터링합니다. 형식은 `YYYY-MM-DD`. |
| `preprocess` | 선택 | 배경 제거, tilt 보정, crop 전처리를 수행합니다. |
| `top_k` | 선택 | 반환할 매칭 결과 개수입니다. 1위 결과만 투명 배경 RGBA PNG base64 이미지 데이터를 포함합니다. |

## API 예시

이미지 1개 등록:

```bash
curl -X POST "http://localhost:8888/gallery/images" \
  -F "lot_id=LOT-001" \
  -F "capture_date=2026-06-22" \
  -F "preprocess=true" \
  -F "files=@before_packaging.jpg"
```

로트별 이미지 목록 조회:

```bash
curl "http://localhost:8888/gallery/images?lot_id=LOT-001"
```

로트와 날짜 기준 조회:

```bash
curl "http://localhost:8888/gallery/images?lot_id=LOT-001&capture_date=2026-06-22"
```

이미지 매칭:

```bash
curl -X POST "http://localhost:8888/match?lot_id=LOT-001&capture_date=2026-06-22&top_k=5" \
  -F "file=@after_packaging.jpg"
```

이미지 1개 삭제:

```bash
curl -X DELETE "http://localhost:8888/gallery/images/before_packaging?lot_id=LOT-001"
```

특정 로트/날짜 전체 삭제:

```bash
curl -X DELETE "http://localhost:8888/gallery/images?lot_id=LOT-001&capture_date=2026-06-22"
```

## 로트 관리 권장 방식

`lot_id`를 갤러리 분리의 기본 단위로 사용하세요. 같은 날짜에 여러 생산 로트가
있으면 로트별로 별도 `lot_id`를 부여합니다.

권장 예시:

```text
lot_id = LOT-20260622-A
capture_date = 2026-06-22
```

같은 파일명이라도 서로 다른 lot에 안전하게 저장됩니다.

```text
LOT-001/2026-06-22/tray_001.png
LOT-002/2026-06-22/tray_001.png
```

매칭은 항상 요청한 `lot_id` 안에서만 수행됩니다. 날짜까지 제한해야 하면
`capture_date`를 함께 전달하세요.

## 문제 해결

Qdrant 상태 확인:

```bash
docker compose ps qdrant
docker compose logs qdrant
```

매칭 모델 파일 누락 시:

```text
FileNotFoundError: /app/models/matching/encoder.pt
```

`models/matching/encoder.pt` 위치에 checkpoint를 넣으세요.

CUDA 확인:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```
