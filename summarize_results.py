import json

THRESHOLD = 31.5798974609375
INPUT_FILE = "benchmark_results.json"

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)

    summary = results[0] if results and results[0].get("type") == "summary" else None
    rows = results[1:] if summary else results

    y_true = []
    y_pred = []

    for res in rows:
        if "error" in res:
            continue
        
        # All images in images2 are normal (0)
        y_true.append(0)
        
        score = res.get("anomaly_score", 0)
        y_pred.append(1 if score >= THRESHOLD else 0)

    if not y_true:
        print("No valid results to summarize.")
        return

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    acc = correct / len(y_true)
    
    # Reverse positive-class naming for display only: normal=positive, anomaly=negative.
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    
    target_dir = summary["image_dir"] if summary else "images2"
    print(f"Summary for {target_dir} (Ground Truth: All Normal)")
    print(f"---------------------------------------------")
    print(f"Total Images: {len(y_true)}")
    print(f"Accuracy:     {acc:.4f}")
    if summary:
        print(f"Preprocess:   {summary['preprocess']}")
        print(f"Threshold:    {summary['threshold']}")
    print(f"\nConfusion Matrix (normal treated as positive class for display):")
    print(f"Actual \\ Pred | Normal (0) | Anomaly (1)")
    print(f"-------------|------------|------------")
    print(f"Normal (0)   | {tp:^10} | {fn:^11}")
    print(f"Anomaly (1)  | {fp:^10} | {tn:^11}")
    print(f"\nDisplay mapping: normal->normal = TP, normal->anomaly = FN, anomaly->normal = FP, anomaly->anomaly = TN")

if __name__ == "__main__":
    main()
