import json
import os
import sys
from pathlib import Path

import requests

URL = "http://localhost:8889/infer"
OUTPUT_FILE = "benchmark_results.json"
THRESHOLD = 31.5798974609375


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}

def main():
    api_key = os.getenv("HANWOO_API_KEY")
    if not api_key:
        raise SystemExit("HANWOO_API_KEY is required")

    image_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/mnt/c/Users/Deku/Downloads/benchmark_v3/benchmark_v3/test/images2")
    preprocess = _parse_bool(sys.argv[2]) if len(sys.argv) > 2 else False

    images = sorted([f for f in image_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not images:
        print("No images found.")
        return

    results = []
    normal_predictions = 0
    print(f"Evaluating {len(images)} images from {image_dir} with preprocess={preprocess}...")

    for img_path in images:
        with open(img_path, "rb") as f:
            files = {"file": (img_path.name, f, "image/jpeg")}
            params = {"preprocess": str(preprocess).lower()}
            headers = {"X-API-Key": api_key}
            
            try:
                response = requests.post(URL, files=files, params=params, headers=headers)
                response.raise_for_status()
                data = response.json()
                score = data.get("anomaly_score")
                if score is not None and score < THRESHOLD:
                    normal_predictions += 1
                results.append(data)
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")
                results.append({"filename": img_path.name, "error": str(e)})

    valid_results = [item for item in results if "error" not in item]
    total = len(valid_results)
    false_positive = total - normal_predictions
    summary = {
        "type": "summary",
        "image_dir": str(image_dir),
        "preprocess": preprocess,
        "threshold": THRESHOLD,
        "total_images": total,
        "accuracy": round(normal_predictions / total, 4) if total else 0.0,
        "confusion_matrix": {
            "display_positive_class": "normal",
            "labels": ["normal", "anomaly"],
            "matrix": [[normal_predictions, false_positive], [0, 0]],
            "mapping": {
                "normal_pred_normal": "TP",
                "normal_pred_anomaly": "FN",
                "anomaly_pred_normal": "FP",
                "anomaly_pred_anomaly": "TN",
            },
        },
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump([summary, *results], f, indent=2, ensure_ascii=False)

    print(f"Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
