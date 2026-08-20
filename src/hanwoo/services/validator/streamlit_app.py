from __future__ import annotations

import io
import html
import os
import shutil
import time
from uuid import uuid4
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import requests
import streamlit as st


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EVALUATE_CATEGORIES = ["비닐", "뼈", "실", "정맥혈응고체", "천"]
UPLOAD_ROOT = Path("/app/storage/validator_uploads")
TEXT = {
    "English": {
        "title": "Hanwoo Validator",
        "caption": "Upload a benchmark ZIP, build a gallery, and validate matching accuracy.",
        "language": "Language",
        "settings": "Matching API and folder settings",
        "matching_api": "Matching API",
        "lot_id": "Lot ID",
        "top_k": "Top-K",
        "preprocess": "Preprocess images",
        "gallery_folder": "Gallery folder",
        "query_folder": "Query folder",
        "api_key": "API key",
        "test_folder": "Test Folder",
        "benchmark_hint": "Upload a ZIP with <code>test/after/&lt;date&gt;/*.jpg</code> and <code>test/before/&lt;date&gt;/*.jpg</code> structure. Each date dir is matched independently (different embedding per date).",
        "no_date_dirs": "No date subdirectories found in {folder}/{gallery_folder} and {folder}/{query_folder}.",
        "per_date_results": "Per-date results",
        "date": "Date",
        "zip_hint": "Upload a ZIP containing the benchmark folders, for example <code>test/after/*.jpg</code> and <code>test/before/*.jpg</code>.",
        "choose_zip": "Choose test folder ZIP",
        "run": "Run matching validation",
        "upload_first": "Upload a ZIP file first.",
        "missing_settings": "Matching API and Lot ID are required.",
        "no_pairs": "No matching gallery/query image pairs found in the ZIP.",
        "api_error": "API error",
        "bad_zip": "Uploaded file is not a valid ZIP.",
        "accuracy": "Accuracy",
        "accuracy_hint": "Accuracy is Top-1 correct divided by total query images. Round-trip includes browser upload and response download. Compute is reported by the API and excludes upload/response encoding.",
        "correct": "Correct",
        "avg_total": "Avg total ms",
        "avg_round_trip": "Avg round-trip ms",
        "avg_compute": "Avg compute ms",
        "elapsed": "Elapsed",
        "results": "Results",
        "results_hint": "<b>Top-K pairs</b> format: <code>Rank N: image_name. Similarity similarity%. Distance embedding_distance.</code> Higher similarity is better; lower distance is better. <b>Top 1 image</b> is the image returned by the API for the best match. <b>Round-trip ms</b> is browser request time. <b>Compute ms</b> is server query compute.",
        "top1_image": "Top 1 image",
        "query_image": "Query image",
        "expected": "Expected",
        "top1": "Top 1",
        "topk_pairs": "Top-K pairs",
        "round_trip_ms": "Round-trip ms",
        "compute_ms": "Compute ms",
        "yes": "yes",
        "no": "no",
        "uploading_gallery": "Uploading gallery",
        "uploaded_gallery": "Uploaded {done}/{total} gallery images",
        "running_queries": "Running query images",
        "queried": "Queried {done}/{total}",
        "running_accuracy": "Running accuracy: {correct}/{total}",
        "rank_line": "Rank {rank}: {name}. Similarity {similarity}. Distance {distance}.",
        "matching_tab": "Matching",
        "anomaly_tab": "PatchCore Anomaly Detection",
        "dinomaly_tab": "Dinomaly Anomaly Detection",
        "anomaly_settings": "PatchCore /evaluate settings",
        "anomaly_api": "PatchCore API",
        "dinomaly_settings": "Dinomaly /infer settings",
        "dinomaly_api": "Dinomaly API",
        "heatmap": "Heatmap",
        "avg_preprocess_ms": "Avg preprocess ms",
        "avg_infer_ms": "Avg infer ms",
        "run_dinomaly": "Run dinomaly evaluation",
        "missing_dinomaly_settings": "Dinomaly API is required.",
        "no_dinomaly_folders": "Could not find test/abnormal and/or test/good folders in the ZIP. Expected structure: test/abnormal/*.jpg and test/good/*.jpg.",
        "dinomaly_zip_hint": "Upload a ZIP containing test/abnormal/*.jpg and test/good/*.jpg folders.",
        "dinomaly_accuracy_hint": "Uses /infer endpoint directly per image. test/abnormal expected as anomaly, test/good expected as normal.",
        "category_dirs": "Category folders",
        "category_dirs_help": "Comma-separated category folder names under the test folder. Each category needs images/ and labels/ subfolders.",
        "anomaly_zip_hint": "Upload a ZIP containing the /evaluate dataset, for example benchmark_v3/test/{category}/images, benchmark_v3/test/{category}/labels, and benchmark_v3/test/images2.",
        "run_anomaly": "Run anomaly evaluation",
        "missing_anomaly_settings": "Anomaly API is required.",
        "no_evaluate_folder": "Could not find a valid /evaluate test folder in the ZIP.",
        "anomaly_accuracy_hint": "/evaluate uses mask labels under each category's labels/ folder. images2 is counted as normal.",
        "average_results": "Average results",
        "per_category_results": "Per-category results",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "threshold": "Threshold",
        "n_evaluated": "Evaluated",
        "n_skipped": "Skipped",
        "category": "Category",
        "n_total": "Total",
        "n_normal": "Normal",
        "n_anomaly": "Anomaly",
    },
    "한국어": {
        "title": "한우 검증 도구",
        "caption": "벤치마크 ZIP을 업로드하고 갤러리를 만든 뒤 매칭 정확도를 검증합니다.",
        "language": "언어",
        "settings": "매칭 API 및 폴더 설정",
        "matching_api": "매칭 API",
        "lot_id": "Lot ID",
        "top_k": "Top-K",
        "preprocess": "이미지 전처리",
        "gallery_folder": "갤러리 폴더",
        "query_folder": "쿼리 폴더",
        "api_key": "API 키",
        "test_folder": "테스트 폴더",
        "benchmark_hint": "ZIP 업로드. 구조: <code>test/after/&lt;날짜&gt;/*.jpg</code>, <code>test/before/&lt;날짜&gt;/*.jpg</code>. 날짜별 독립 매칭 (날짜별 다른 임베딩).",
        "no_date_dirs": "{folder}/{gallery_folder} 및 {folder}/{query_folder}에서 날짜 하위 디렉토리를 찾지 못했습니다.",
        "per_date_results": "날짜별 결과",
        "date": "날짜",
        "zip_hint": "벤치마크 폴더가 들어있는 ZIP을 업로드하세요. 예: <code>test/after/*.jpg</code>, <code>test/before/*.jpg</code>",
        "choose_zip": "테스트 폴더 ZIP 선택",
        "run": "매칭 검증 실행",
        "upload_first": "먼저 ZIP 파일을 업로드하세요.",
        "missing_settings": "매칭 API와 Lot ID가 필요합니다.",
        "no_pairs": "ZIP에서 매칭 가능한 갤러리/쿼리 이미지 쌍을 찾지 못했습니다.",
        "api_error": "API 오류",
        "bad_zip": "업로드한 파일이 올바른 ZIP이 아닙니다.",
        "accuracy": "정확도",
        "accuracy_hint": "정확도는 Top-1 정답 수를 전체 쿼리 이미지 수로 나눈 값입니다. Round-trip은 브라우저 업로드와 응답 다운로드를 포함합니다. Compute는 API가 반환한 서버 쿼리 계산 시간이며 업로드/응답 인코딩은 제외합니다.",
        "correct": "정답",
        "avg_total": "평균 전체 ms",
        "avg_round_trip": "평균 왕복 ms",
        "avg_compute": "평균 계산 ms",
        "elapsed": "소요 시간",
        "results": "결과",
        "results_hint": "<b>Top-K 쌍</b> 형식: <code>순위 N: 이미지명. 유사도 similarity%. 거리 embedding_distance.</code> 유사도는 높을수록 좋고, 거리는 낮을수록 좋습니다. <b>Top 1 이미지</b>는 API가 반환한 1순위 매칭 이미지입니다. <b>왕복 ms</b>는 브라우저 요청 시간입니다. <b>계산 ms</b>는 서버 쿼리 계산 시간입니다.",
        "top1_image": "Top 1 이미지",
        "query_image": "쿼리 이미지",
        "expected": "기대값",
        "top1": "Top 1",
        "topk_pairs": "Top-K 쌍",
        "round_trip_ms": "왕복 ms",
        "compute_ms": "계산 ms",
        "yes": "예",
        "no": "아니오",
        "uploading_gallery": "갤러리 업로드 중",
        "uploaded_gallery": "갤러리 이미지 {done}/{total} 업로드",
        "running_queries": "쿼리 이미지 실행 중",
        "queried": "쿼리 {done}/{total} 완료",
        "running_accuracy": "진행 중 정확도: {correct}/{total}",
        "rank_line": "순위 {rank}: {name}. 유사도 {similarity}. 거리 {distance}.",
        "matching_tab": "매칭",
        "anomaly_tab": "PatchCore 이상탐지",
        "dinomaly_tab": "Dinomaly 이상탐지",
        "anomaly_settings": "PatchCore /evaluate 설정",
        "anomaly_api": "PatchCore API",
        "dinomaly_settings": "Dinomaly /infer 설정",
        "dinomaly_api": "Dinomaly API",
        "heatmap": "히트맵",
        "avg_preprocess_ms": "평균 전처리 ms",
        "avg_infer_ms": "평균 추론 ms",
        "run_dinomaly": "Dinomaly 평가 실행",
        "missing_dinomaly_settings": "Dinomaly API가 필요합니다.",
        "no_dinomaly_folders": "ZIP에서 test/abnormal 또는 test/good 폴더를 찾지 못했습니다. 구조: test/abnormal/*.jpg, test/good/*.jpg",
        "dinomaly_zip_hint": "test/abnormal/*.jpg와 test/good/*.jpg 폴더가 있는 ZIP을 업로드하세요.",
        "dinomaly_accuracy_hint": "/infer 엔드포인트를 이미지별로 직접 호출합니다. test/abnormal은 이상으로, test/good은 정상으로 기대합니다.",
        "category_dirs": "카테고리 폴더",
        "category_dirs_help": "테스트 폴더 아래 카테고리 폴더명을 쉼표로 입력합니다. 각 카테고리에는 images/와 labels/가 필요합니다.",
        "anomaly_zip_hint": "/evaluate 데이터셋 ZIP을 업로드하세요. 예: benchmark_v3/test/{category}/images, benchmark_v3/test/{category}/labels, benchmark_v3/test/images2",
        "run_anomaly": "이상탐지 평가 실행",
        "missing_anomaly_settings": "이상탐지 API가 필요합니다.",
        "no_evaluate_folder": "ZIP에서 유효한 /evaluate 테스트 폴더를 찾지 못했습니다.",
        "anomaly_accuracy_hint": "/evaluate는 각 카테고리 labels/ 폴더의 마스크 라벨을 사용합니다. images2는 정상으로 계산합니다.",
        "average_results": "평균 결과",
        "per_category_results": "카테고리별 결과",
        "precision": "정밀도",
        "recall": "재현율",
        "f1": "F1",
        "threshold": "임계값",
        "n_evaluated": "평가 수",
        "n_skipped": "건너뜀",
        "category": "카테고리",
        "n_total": "전체",
        "n_normal": "정상",
        "n_anomaly": "이상",
    },
}


@dataclass(frozen=True)
class TestImage:
    name: str
    data: bytes
    mime: str


def image_mime(name: str) -> str:
    suffix = PurePosixPath(name).suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    if suffix == ".bmp":
        return "image/bmp"
    return "image/png"


def images_from_zip(blob: bytes, folder: str) -> list[TestImage]:
    folder = folder.strip("/")
    images: list[TestImage] = []
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            if info.is_dir() or path.suffix.lower() not in IMAGE_EXTS:
                continue
            if folder not in path.parts:
                continue
            images.append(
                TestImage(path.name, archive.read(info), image_mime(path.name))
            )
    return sorted(images, key=lambda item: item.name)


def images_from_dir(root: Path, folder: str) -> dict[str, list[TestImage]]:
    root = root.resolve()
    folder_path = root / folder
    if not folder_path.is_dir():
        return {}
    date_dirs: dict[str, list[TestImage]] = {}
    for entry in sorted(folder_path.iterdir()):
        if not entry.is_dir():
            continue
        images: list[TestImage] = []
        for f in sorted(entry.iterdir()):
            if f.suffix.lower() not in IMAGE_EXTS:
                continue
            images.append(TestImage(f.name, f.read_bytes(), image_mime(f.name)))
        if images:
            date_dirs[entry.name] = images
    return date_dirs


def extract_zip(blob) -> Path:
    """Extract an uploaded zip, streaming rather than copying it in memory.

    Streamlit already holds the upload in RAM; calling .getvalue() on top of that
    doubled it, and reading each member whole tripled the peak. A 2 GB zip used to
    OOM the container.
    """
    if hasattr(blob, "seek"):
        blob.seek(0)
    source = blob if hasattr(blob, "read") else io.BytesIO(blob)
    target = UPLOAD_ROOT / uuid4().hex
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as archive:
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            if info.is_dir():
                continue
            if path.is_absolute() or ".." in path.parts:
                continue
            output = target.joinpath(*path.parts)
            output.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as src, open(output, "wb") as dst:
                shutil.copyfileobj(src, dst)
    return target


def find_evaluate_test_dir(root: Path, categories: list[str]) -> Path | None:
    candidates = [root, *[path for path in root.rglob("*") if path.is_dir()]]
    best: tuple[int, Path] | None = None
    for path in candidates:
        score = int((path / "images2").is_dir())
        score += sum(int((path / category / "images").is_dir()) for category in categories)
        score += sum(int((path / category / "labels").is_dir()) for category in categories)
        if score and (best is None or score > best[0]):
            best = (score, path)
    return best[1] if best else None


def evaluate_categories(test_dir: Path, categories: list[str]) -> list[str]:
    existing = [
        category
        for category in categories
        if (test_dir / category / "images").is_dir()
    ]
    if existing:
        return existing
    return sorted(
        path.name
        for path in test_dir.iterdir()
        if path.is_dir()
        and path.name != "images2"
        and (path / "images").is_dir()
        and (path / "labels").is_dir()
    )


def api_headers(api_key: str) -> dict[str, str]:
    return {"X-API-Key": api_key} if api_key else {}


def api_request(method: str, base_url: str, path: str, api_key: str, **kwargs):
    response = requests.request(
        method,
        f"{base_url.rstrip('/')}{path}",
        headers={**api_headers(api_key), **kwargs.pop("headers", {})},
        timeout=kwargs.pop("timeout", 900),
        **kwargs,
    )
    response.raise_for_status()
    return response.json()


SCORE_MODES = ("roi_topk", "roi_max", "full")


@st.cache_data(ttl=60, show_spinner=False)
def fetch_dinomaly_settings(base_url: str, api_key: str) -> tuple[str, str]:
    try:
        health = api_request("GET", base_url, "/health", api_key, timeout=5)
        return str(health["threshold"]), str(health["score_mode"])
    except Exception:
        return "0.192822", SCORE_MODES[0]


def upload_gallery(
    base_url: str,
    api_key: str,
    lot_id: str,
    gallery: list[TestImage],
    preprocess: bool,
    text: dict[str, str],
    batch_size: int = 25,
) -> None:
    progress = st.progress(0, text=text["uploading_gallery"])
    for offset in range(0, len(gallery), batch_size):
        batch = gallery[offset : offset + batch_size]
        files = [
            ("files", (image.name, io.BytesIO(image.data), image.mime))
            for image in batch
        ]
        api_request(
            "POST",
            base_url,
            "/gallery/images",
            api_key,
            data={"lot_id": lot_id, "preprocess": str(preprocess).lower()},
            files=files,
        )
        progress.progress(
            min(1.0, (offset + len(batch)) / len(gallery)),
            text=text["uploaded_gallery"].format(
                done=offset + len(batch),
                total=len(gallery),
            ),
        )


def format_pair(match: dict, text: dict[str, str]) -> str:
    rank = match.get("rank", "-")
    name = match.get("name", "-")
    similarity = match.get("similarity")
    distance = match.get("distance")
    sim_text = f"{similarity:.2f}%" if isinstance(similarity, int | float) else "-"
    dist_text = f"{distance:.4f}" if isinstance(distance, int | float) else "-"
    return text["rank_line"].format(
        rank=rank,
        name=name,
        similarity=sim_text,
        distance=dist_text,
    )


def render_results_table(rows: list[dict], text: dict[str, str]) -> None:
    rendered_rows = []
    for row in rows:
        top_image = row.get("Top 1 image") or ""
        image_html = (
            f'<img class="result-thumb" src="{html.escape(top_image, quote=True)}" alt="">'
            if top_image
            else "-"
        )
        pair_lines = "".join(
            f'<div class="pair-line">{html.escape(line)}</div>'
            for line in str(row["Top-K pairs"]).splitlines()
        )
        rendered_rows.append(
            "<tr>"
            f"<td>{image_html}</td>"
            f"<td>{html.escape(str(row['query_image']))}</td>"
            f"<td>{html.escape(str(row['expected']))}</td>"
            f"<td>{html.escape(str(row['top1']))}</td>"
            f"<td class=\"{ 'ok' if row['Correct'] == 'yes' else 'bad' }\">"
            f"{html.escape(text['yes'] if row['Correct'] == 'yes' else text['no'])}</td>"
            f"<td class=\"pairs-cell\">{pair_lines}</td>"
            f"<td>{html.escape(str(row['round_trip_ms']))}</td>"
            f"<td>{html.escape(str(row['compute_ms']))}</td>"
            "</tr>"
        )
    st.markdown(
        f"""
        <div class="results-scroll">
          <table class="results-table">
            <thead>
              <tr>
                <th>{html.escape(text["top1_image"])}</th>
                <th>{html.escape(text["query_image"])}</th>
                <th>{html.escape(text["expected"])}</th>
                <th>{html.escape(text["top1"])}</th>
                <th>{html.escape(text["correct"])}</th>
                <th>{html.escape(text["topk_pairs"])}</th>
                <th>{html.escape(text["round_trip_ms"])}</th>
                <th>{html.escape(text["compute_ms"])}</th>
              </tr>
            </thead>
            <tbody>
        """
        + "\n".join(rendered_rows)
        + """
            </tbody>
          </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_evaluate_table(result: dict, text: dict[str, str]) -> None:
    rendered_rows = []
    for name, row in result.get("categories", {}).items():
        rendered_rows.append(
            "<tr>"
            f"<td>{html.escape(str(name))}</td>"
            f"<td>{html.escape(str(row.get('accuracy', '-')))}</td>"
            f"<td>{html.escape(str(row.get('precision', '-')))}</td>"
            f"<td>{html.escape(str(row.get('recall', '-')))}</td>"
            f"<td>{html.escape(str(row.get('f1', '-')))}</td>"
            f"<td>{html.escape(str(row.get('n_total', '-')))}</td>"
            f"<td>{html.escape(str(row.get('n_normal', '-')))}</td>"
            f"<td>{html.escape(str(row.get('n_anomaly', '-')))}</td>"
            "</tr>"
        )
    st.markdown(
        f"""
        <div class="results-scroll">
          <table class="results-table">
            <thead>
              <tr>
                <th>{html.escape(text["category"])}</th>
                <th>{html.escape(text["accuracy"])}</th>
                <th>{html.escape(text["precision"])}</th>
                <th>{html.escape(text["recall"])}</th>
                <th>{html.escape(text["f1"])}</th>
                <th>{html.escape(text["n_total"])}</th>
                <th>{html.escape(text["n_normal"])}</th>
                <th>{html.escape(text["n_anomaly"])}</th>
              </tr>
            </thead>
            <tbody>
        """
        + "\n".join(rendered_rows)
        + """
            </tbody>
          </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def run_matching(
    base_url: str,
    api_key: str,
    lot_id: str,
    top_k: int,
    preprocess: bool,
    gallery: list[TestImage],
    query: list[TestImage],
    text: dict[str, str],
) -> tuple[list[dict], dict]:
    api_request("DELETE", base_url, f"/gallery/images?lot_id={lot_id}", api_key)
    upload_gallery(base_url, api_key, lot_id, gallery, preprocess, text)

    correct = 0
    round_trip_ms = 0
    compute_ms = 0.0
    preprocess_ms = 0.0
    rows: list[dict] = []
    status = st.empty()
    progress = st.progress(0, text=text["running_queries"])
    start = time.perf_counter()

    for index, image in enumerate(query, start=1):
        t0 = time.perf_counter()
        data = api_request(
            "POST",
            base_url,
            f"/match?lot_id={lot_id}&top_k={top_k}&preprocess={str(preprocess).lower()}",
            api_key,
            files={"file": (image.name, io.BytesIO(image.data), image.mime)},
        )
        rt_ms = round((time.perf_counter() - t0) * 1000)
        server_ms = float(data.get("query_compute_ms") or 0.0)
        server_preprocess_ms = float(data.get("preprocess_ms") or 0.0)
        round_trip_ms += rt_ms
        compute_ms += server_ms
        preprocess_ms += server_preprocess_ms

        matches = data.get("matches", [])
        top_match = matches[0] if matches else {}
        expected = PurePosixPath(image.name).stem
        predicted = top_match.get("name", "")
        is_correct = predicted == expected
        correct += int(is_correct)
        top_image = ""
        if top_match.get("image_base64"):
            top_image = (
                f"data:{top_match.get('image_mime_type', 'image/png')};"
                f"base64,{top_match['image_base64']}"
            )
        rows.append(
            {
                "Top 1 image": top_image,
                "query_image": image.name,
                "expected": expected,
                "top1": predicted or "-",
                "Correct": "yes" if is_correct else "no",
                "Top-K pairs": "\n".join(format_pair(match, text) for match in matches),
                "round_trip_ms": rt_ms,
                "compute_ms": round(server_ms),
                "preprocess_ms": round(server_preprocess_ms, 1),
            }
        )
        progress.progress(
            index / len(query),
            text=text["queried"].format(done=index, total=len(query)),
        )
        status.caption(text["running_accuracy"].format(correct=correct, total=index))

    total = len(query)
    metrics = {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "avg_round_trip_ms": round(round_trip_ms / total) if total else 0,
        "avg_compute_ms": round(compute_ms / total) if total else 0,
        "avg_preprocess_ms": round(preprocess_ms / total, 1) if total else 0,
        "elapsed_s": round(time.perf_counter() - start, 1),
    }
    return rows, metrics


def run_anomaly(
    base_url: str,
    api_key: str,
    preprocess: bool,
    zip_blob,
    categories: list[str],
    text: dict[str, str],
) -> tuple[dict, dict]:
    extracted_root = extract_zip(zip_blob)
    test_dir = find_evaluate_test_dir(extracted_root, categories)
    if test_dir is None:
        shutil.rmtree(extracted_root, ignore_errors=True)
        raise ValueError(text["no_evaluate_folder"])
    categories = evaluate_categories(test_dir, categories)

    start = time.perf_counter()
    try:
        result = api_request(
            "POST",
            base_url,
            "/evaluate",
            api_key,
            json={
                "test_base_dir": str(test_dir),
                "category_dirs": categories,
                "images2_dir": str(test_dir / "images2"),
                "preprocess": preprocess,
            },
            timeout=3600,
        )
        elapsed_s = time.perf_counter() - start
    finally:
        shutil.rmtree(extracted_root, ignore_errors=True)
    total = result.get("total", {})
    n_evaluated = int(result.get("n_evaluated") or 0)
    return result, {
        "accuracy": float(total.get("accuracy") or 0.0),
        "precision": float(total.get("precision") or 0.0),
        "recall": float(total.get("recall") or 0.0),
        "f1": float(total.get("f1") or 0.0),
        "threshold": total.get("threshold", "-"),
        "n_evaluated": n_evaluated,
        "n_skipped": result.get("n_skipped", 0),
        "avg_infer_ms": result.get("avg_infer_ms", 0),
        "avg_round_trip_ms": round((elapsed_s * 1000) / n_evaluated, 1) if n_evaluated else 0,
    }


def _find_dinomaly_dirs(root: Path) -> tuple[Path | None, Path | None]:
    abnormal = None
    good = None
    for d in root.rglob("*"):
        if not d.is_dir():
            continue
        if d.name == "abnormal" and any(
            f.is_file() and f.suffix.lower() in IMAGE_EXTS for f in d.iterdir()
        ):
            abnormal = d
        if d.name == "good" and any(
            f.is_file() and f.suffix.lower() in IMAGE_EXTS for f in d.iterdir()
        ):
            good = d
    if abnormal and good:
        for f in abnormal.iterdir():
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                return abnormal, good
        for f in good.iterdir():
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                return abnormal, good
    candidates = sorted(
        p for p in root.rglob("abnormal") if p.is_dir()
    )
    for ab in candidates:
        for g in sorted(root.rglob("good")):
            if g.is_dir() and ab.parent == g.parent:
                return ab, g
    return None, None


def run_dinomaly(
    base_url: str,
    api_key: str,
    preprocess: bool,
    heatmap: bool,
    zip_blob,
    text: dict[str, str],
    threshold: float | None = None,
    score_mode: str | None = None,
) -> tuple[list[dict], dict]:
    extracted_root = extract_zip(zip_blob)
    test_abnormal, test_good = _find_dinomaly_dirs(extracted_root)
    if test_abnormal is None or test_good is None:
        shutil.rmtree(extracted_root, ignore_errors=True)
        raise ValueError(text["no_dinomaly_folders"])

    if threshold is not None:
        api_request(
            "PUT",
            base_url,
            "/threshold",
            api_key,
            json={"threshold": float(threshold)},
        )

    if score_mode is not None:
        api_request(
            "PUT",
            base_url,
            "/score-mode",
            api_key,
            json={"score_mode": score_mode},
        )

    all_images: list[tuple[Path, bool]] = []
    for p in test_abnormal.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            all_images.append((p, True))
    for p in test_good.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            all_images.append((p, False))
    all_images.sort(key=lambda x: x[0].name)

    status = st.empty()
    warmup_progress = st.progress(0, text="Warming up...")
    import random
    warmup_set = random.sample(all_images, min(10, len(all_images)))
    for i, (img_path, _) in enumerate(warmup_set, start=1):
        with open(img_path, "rb") as f:
            api_request(
                "POST",
                base_url,
                f"/infer?preprocess={str(preprocess).lower()}&heatmap={str(heatmap).lower()}",
                api_key,
                files={"file": ("img.jpg", f, "image/jpeg")},
            )
        warmup_progress.progress(i / len(warmup_set))
    warmup_progress.empty()

    correct = 0
    tp = fp = fn = tn = 0
    sum_preprocess = 0.0
    sum_infer = 0.0
    rows: list[dict] = []
    progress = st.progress(0)
    start = time.perf_counter()

    for index, (img_path, expected_abnormal) in enumerate(all_images, start=1):
        with open(img_path, "rb") as f:
            t0 = time.perf_counter()
            data = api_request(
                "POST",
                base_url,
                f"/infer?preprocess={str(preprocess).lower()}&heatmap={str(heatmap).lower()}",
                api_key,
                files={"file": (img_path.name, f, "image/jpeg")},
            )
            rt_ms = (time.perf_counter() - t0) * 1000
        is_anomaly = bool(data.get("is_anomaly", False))
        score = float(data.get("anomaly_score", 0.0))
        pre_ms = float(data.get("preprocess_ms", 0))
        inf_ms = float(data.get("infer_ms", 0))
        sum_preprocess += pre_ms
        sum_infer += inf_ms

        pred_correct = is_anomaly == expected_abnormal
        correct += int(pred_correct)

        if expected_abnormal and is_anomaly:
            tp += 1
        elif expected_abnormal and not is_anomaly:
            fn += 1
        elif not expected_abnormal and is_anomaly:
            fp += 1
        else:
            tn += 1

        rows.append({
            "image": img_path.name,
            "score": round(score, 4),
            "predicted": "anomaly" if is_anomaly else "normal",
            "expected": "anomaly" if expected_abnormal else "normal",
            "correct": "yes" if pred_correct else "no",
            "preprocess_ms": round(pre_ms, 1),
            "infer_ms": round(inf_ms, 1),
            "total_ms": round(rt_ms, 1),
        })
        progress.progress(index / len(all_images))
        status.caption(f"Evaluated {index}/{len(all_images)} — running acc: {correct}/{index}")

    elapsed_s = time.perf_counter() - start
    total = len(all_images)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    metrics = {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_preprocess_ms": round(sum_preprocess / total, 1) if total else 0,
        "avg_infer_ms": round(sum_infer / total, 1) if total else 0,
        "avg_total_ms": round((sum_preprocess + sum_infer) / total, 1) if total else 0,
        "elapsed_s": round(elapsed_s, 1),
    }
    shutil.rmtree(extracted_root, ignore_errors=True)
    return rows, metrics


st.set_page_config(page_title="Hanwoo Validator", layout="wide")
st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; max-width: 1280px; }
      div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 12px;
        color: #111827;
      }
      div[data-testid="stMetric"] * {
        color: #111827 !important;
      }
      .hint {
        color: #4b5563;
        font-size: 0.92rem;
        line-height: 1.45;
      }
      @media (prefers-color-scheme: dark) {
        .hint { color: #cbd5e1; }
      }
      .results-scroll {
        border: 1px solid #d1d5db;
        border-radius: 8px;
        overflow-x: auto;
      }
      .results-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.9rem;
      }
      .results-table th,
      .results-table td {
        border-bottom: 1px solid #e5e7eb;
        padding: 10px;
        text-align: left;
        vertical-align: top;
      }
      .results-table th {
        background: #f8fafc;
        color: #111827;
        font-weight: 700;
        white-space: nowrap;
      }
      .result-thumb {
        width: 88px;
        height: 66px;
        object-fit: contain;
        border: 1px solid #d1d5db;
        border-radius: 6px;
        background: #f8fafc;
      }
      .pairs-cell {
        min-width: 360px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        line-height: 1.45;
      }
      .pair-line {
        display: block;
        margin-bottom: 4px;
        white-space: nowrap;
      }
      .ok { color: #15803d; font-weight: 700; }
      .bad { color: #b91c1c; font-weight: 700; }
      @media (prefers-color-scheme: dark) {
        .results-scroll { border-color: #334155; }
        .results-table th { background: #1e293b; color: #f8fafc; }
        .results-table td { border-bottom-color: #334155; }
        .result-thumb { background: #0f172a; border-color: #334155; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

language = st.radio(
    "Language / 언어",
    ["English", "한국어"],
    horizontal=True,
    label_visibility="collapsed",
)
text = TEXT[language]

st.title(text["title"])
st.caption(text["caption"])

matching_tab, anomaly_tab, dinomaly_tab = st.tabs(
    [text["matching_tab"], text["anomaly_tab"], text["dinomaly_tab"]]
)

with matching_tab:
    with st.expander(text["settings"], expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            base_url = st.text_input(
                text["matching_api"],
                value=os.getenv("MATCHING_API_BASE_URL", "http://matching:8000"),
                key="matching_api",
            )
            lot_id = st.text_input(text["lot_id"], value="validator-test")
        with col2:
            top_k = st.number_input(text["top_k"], min_value=1, max_value=50, value=5, step=1)
            preprocess = st.checkbox(text["preprocess"], value=True, key="matching_preprocess")
        with col3:
            gallery_folder = st.text_input(text["gallery_folder"], value="test/after")
            query_folder = st.text_input(text["query_folder"], value="test/before")
        api_key = st.text_input(
            text["api_key"],
            value=os.getenv("HANWOO_API_KEY", ""),
            type="password",
            key="matching_api_key",
        )

    st.subheader(text["test_folder"])
    st.markdown(
        f'<div class="hint">{text["benchmark_hint"]}</div>',
        unsafe_allow_html=True,
    )
    uploaded_zip = st.file_uploader(text["choose_zip"], type=["zip"], key="matching_zip")
    run = st.button(text["run"], type="primary", use_container_width=True)

    if run:
        if not uploaded_zip:
            st.error(text["upload_first"])
            st.stop()
        if not base_url or not lot_id:
            st.error(text["missing_settings"])
            st.stop()

        try:
            root = extract_zip(uploaded_zip)

            gallery_by_date = images_from_dir(root, gallery_folder)
            query_by_date = images_from_dir(root, query_folder)
            common_dates = sorted(set(gallery_by_date) & set(query_by_date))
            if not common_dates:
                st.error(
                    text["no_date_dirs"].format(
                        folder=root, gallery_folder=gallery_folder, query_folder=query_folder
                    )
                )
                st.stop()

            all_rows: list[dict] = []
            total_correct = 0
            total_queries = 0
            total_round_trip_ms = 0
            total_compute_ms = 0.0
            per_date_metrics: list[dict] = []
            overall_start = time.perf_counter()

            for date in common_dates:
                gallery_images = gallery_by_date[date]
                query_images = query_by_date[date]
                gallery_stems = {PurePosixPath(item.name).stem for item in gallery_images}
                query_images = [
                    image for image in query_images
                    if PurePosixPath(image.name).stem in gallery_stems
                ]
                if not gallery_images or not query_images:
                    continue

                api_request("DELETE", base_url, f"/gallery/images?lot_id={lot_id}", api_key)
                rows, metrics = run_matching(
                    base_url, api_key, lot_id, int(top_k), preprocess,
                    gallery_images, query_images, text,
                )
                for row in rows:
                    row["date"] = date
                all_rows.extend(rows)
                total_correct += metrics["correct"]
                total_queries += metrics["total"]
                total_round_trip_ms += metrics["avg_round_trip_ms"] * metrics["total"]
                total_compute_ms += metrics["avg_compute_ms"] * metrics["total"]
                per_date_metrics.append({**metrics, "date": date})

            overall_accuracy = total_correct / total_queries if total_queries else 0.0
            overall_avg_rt = round(total_round_trip_ms / total_queries) if total_queries else 0
            overall_avg_comp = round(total_compute_ms / total_queries) if total_queries else 0
            metrics = {
                "accuracy": overall_accuracy,
                "correct": total_correct,
                "total": total_queries,
                "avg_round_trip_ms": overall_avg_rt,
                "avg_compute_ms": overall_avg_comp,
                "elapsed_s": round(time.perf_counter() - overall_start, 1),
            }
        except requests.HTTPError as exc:
            detail = exc.response.text if exc.response is not None else str(exc)
            st.error(f"{text['api_error']}: {detail}")
            st.stop()

        st.subheader(text["average_results"])
        st.markdown(
            f'<div class="hint">{text["accuracy_hint"]}</div>',
            unsafe_allow_html=True,
        )
        metric_cols = st.columns(5)
        metric_cols[0].metric(text["accuracy"], f"{metrics['accuracy'] * 100:.2f}%")
        metric_cols[1].metric(text["correct"], f"{metrics['correct']}/{metrics['total']}")
        metric_cols[2].metric(text["avg_round_trip"], metrics["avg_round_trip_ms"])
        metric_cols[3].metric(text["avg_compute"], metrics["avg_compute_ms"])
        metric_cols[4].metric(text["elapsed"], f"{metrics['elapsed_s']}s")

        st.subheader(text["per_date_results"])
        per_date_rows = [
            {
                text["date"]: m["date"],
                text["accuracy"]: f"{m['accuracy'] * 100:.2f}%",
                text["correct"]: f"{m['correct']}/{m['total']}",
                text["avg_round_trip"]: m["avg_round_trip_ms"],
                text["avg_compute"]: m["avg_compute_ms"],
                text["elapsed"]: m["elapsed_s"],
            }
            for m in per_date_metrics
        ]
        st.table(per_date_rows)

        st.subheader(text["results"])
        st.markdown(
            f'<div class="hint">{text["results_hint"]}</div>',
            unsafe_allow_html=True,
        )
        render_results_table(all_rows, text)

with anomaly_tab:
    with st.expander(text["anomaly_settings"], expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            anomaly_base_url = st.text_input(
                text["anomaly_api"],
                value=os.getenv("ANOMALY_API_BASE_URL", "http://anomaly:8001"),
                key="anomaly_api",
            )
            anomaly_api_key = st.text_input(
                text["api_key"],
                value=os.getenv("HANWOO_API_KEY", ""),
                type="password",
                key="anomaly_api_key",
            )
        with col2:
            anomaly_preprocess = st.checkbox(text["preprocess"], value=True, key="anomaly_preprocess")
            category_dirs = st.text_input(
                text["category_dirs"],
                value=",".join(EVALUATE_CATEGORIES),
                help=text["category_dirs_help"],
            )

    st.subheader(text["test_folder"])
    st.markdown(
        f'<div class="hint">{text["anomaly_zip_hint"]}</div>',
        unsafe_allow_html=True,
    )
    anomaly_zip = st.file_uploader(text["choose_zip"], type=["zip"], key="anomaly_zip")
    run_anomaly_button = st.button(text["run_anomaly"], type="primary", use_container_width=True)

    if run_anomaly_button:
        if not anomaly_zip:
            st.error(text["upload_first"])
            st.stop()
        if not anomaly_base_url:
            st.error(text["missing_anomaly_settings"])
            st.stop()

        try:
            categories = [item.strip() for item in category_dirs.split(",") if item.strip()]
            result, metrics = run_anomaly(
                anomaly_base_url,
                anomaly_api_key,
                anomaly_preprocess,
                anomaly_zip,
                categories,
                text,
            )
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
        except requests.HTTPError as exc:
            detail = exc.response.text if exc.response is not None else str(exc)
            st.error(f"{text['api_error']}: {detail}")
            st.stop()
        except zipfile.BadZipFile:
            st.error(text["bad_zip"])
            st.stop()

        st.subheader(text["accuracy"])
        st.markdown(
            f'<div class="hint">{text["anomaly_accuracy_hint"]}</div>',
            unsafe_allow_html=True,
        )
        metric_cols = st.columns(4)
        metric_cols[0].metric(text["accuracy"], f"{metrics['accuracy'] * 100:.2f}%")
        metric_cols[1].metric(text["precision"], f"{metrics['precision'] * 100:.2f}%")
        metric_cols[2].metric(text["recall"], f"{metrics['recall'] * 100:.2f}%")
        metric_cols[3].metric(text["f1"], f"{metrics['f1'] * 100:.2f}%")
        metric_cols = st.columns(5)
        metric_cols[0].metric(text["n_evaluated"], metrics["n_evaluated"])
        metric_cols[1].metric(text["n_skipped"], metrics["n_skipped"])
        metric_cols[2].metric(text["threshold"], metrics["threshold"])
        metric_cols[3].metric(text["avg_infer_ms"], metrics["avg_infer_ms"])
        metric_cols[4].metric(text["avg_round_trip"], metrics["avg_round_trip_ms"])

        st.subheader(text["per_category_results"])
        render_evaluate_table(result, text)

with dinomaly_tab:
    with st.expander(text["dinomaly_settings"], expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            dinomaly_base_url = st.text_input(
                text["dinomaly_api"],
                value=os.getenv("DINOMALY_API_BASE_URL", "http://dinomaly:8002"),
                key="dinomaly_api",
            )
            dinomaly_api_key = st.text_input(
                text["api_key"],
                value=os.getenv("HANWOO_API_KEY", ""),
                type="password",
                key="dinomaly_api_key",
            )
        with col2:
            server_threshold, server_score_mode = fetch_dinomaly_settings(
                dinomaly_base_url, dinomaly_api_key
            )
            dinomaly_preprocess = st.checkbox(text["preprocess"], value=True, key="dinomaly_preprocess")
            dinomaly_heatmap = st.checkbox(text["heatmap"], value=False, key="dinomaly_heatmap")
            dinomaly_threshold = st.text_input(
                "Threshold",
                value=server_threshold,
                key="dinomaly_threshold",
                help="Set DINOMALY server threshold via PUT /threshold. Anomaly if score >= threshold.",
            )
            dinomaly_score_mode = st.selectbox(
                "Score mode",
                SCORE_MODES,
                index=SCORE_MODES.index(server_score_mode) if server_score_mode in SCORE_MODES else 0,
                key="dinomaly_score_mode",
                help="Set DINOMALY server score mode via PUT /score-mode.",
            )

    st.subheader(text["test_folder"])
    st.markdown(
        f'<div class="hint">{text["dinomaly_zip_hint"]}</div>',
        unsafe_allow_html=True,
    )
    dinomaly_zip = st.file_uploader(text["choose_zip"], type=["zip"], key="dinomaly_zip")
    run_dinomaly_button = st.button(text["run_dinomaly"], type="primary", use_container_width=True)

    if run_dinomaly_button:
        if not dinomaly_zip:
            st.error(text["upload_first"])
            st.stop()
        if not dinomaly_base_url:
            st.error(text["missing_dinomaly_settings"])
            st.stop()

        try:
            dinomaly_threshold = float(dinomaly_threshold)
        except ValueError:
            st.error("Threshold must be a valid number.")
            st.stop()

        try:
            rows, metrics = run_dinomaly(
                dinomaly_base_url,
                dinomaly_api_key,
                dinomaly_preprocess,
                dinomaly_heatmap,
                dinomaly_zip,
                text,
                threshold=dinomaly_threshold,
                score_mode=dinomaly_score_mode,
            )
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
        except requests.HTTPError as exc:
            detail = exc.response.text if exc.response is not None else str(exc)
            st.error(f"{text['api_error']}: {detail}")
            st.stop()
        except zipfile.BadZipFile:
            st.error(text["bad_zip"])
            st.stop()

        st.subheader(text["accuracy"])
        st.markdown(
            f'<div class="hint">{text["dinomaly_accuracy_hint"]}</div>',
            unsafe_allow_html=True,
        )
        metric_cols = st.columns(6)
        metric_cols[0].metric(text["accuracy"], f"{metrics['accuracy'] * 100:.2f}%")
        metric_cols[1].metric(text["precision"], f"{metrics['precision'] * 100:.2f}%")
        metric_cols[2].metric(text["recall"], f"{metrics['recall'] * 100:.2f}%")
        metric_cols[3].metric(text["correct"], f"{metrics['correct']}/{metrics['total']}")
        metric_cols[4].metric(text["avg_preprocess_ms"], metrics["avg_preprocess_ms"])
        metric_cols[5].metric(text["elapsed"], f"{metrics['elapsed_s']}s")

        st.subheader(text["results"])
        rendered_rows = []
        for row in rows:
            rendered_rows.append(
                "<tr>"
                f"<td>{html.escape(str(row['image']))}</td>"
                f"<td>{html.escape(str(row['score']))}</td>"
                f"<td>{html.escape(str(row['predicted']))}</td>"
                f"<td>{html.escape(str(row['expected']))}</td>"
                f"<td class=\"{ 'ok' if row['correct'] == 'yes' else 'bad' }\">"
                f"{html.escape(text['yes'] if row['correct'] == 'yes' else text['no'])}</td>"
                f"<td>{html.escape(str(row['preprocess_ms']))}</td>"
                f"<td>{html.escape(str(row['infer_ms']))}</td>"
                f"<td>{html.escape(str(row['total_ms']))}</td>"
                "</tr>"
            )
        st.markdown(
            f"""
            <div class="results-scroll">
              <table class="results-table">
                <thead>
                  <tr>
                    <th>{html.escape('Image')}</th>
                    <th>{html.escape('Score')}</th>
                    <th>{html.escape('Predicted')}</th>
                    <th>{html.escape('Expected')}</th>
                    <th>{html.escape(text['correct'])}</th>
                    <th>{html.escape('Preproc ms')}</th>
                    <th>{html.escape('Infer ms')}</th>
                    <th>{html.escape('Total ms')}</th>
                  </tr>
                </thead>
                <tbody>
            """
            + "\n".join(rendered_rows)
            + """
                </tbody>
              </table>
            </div>
            """,
            unsafe_allow_html=True,
        )
