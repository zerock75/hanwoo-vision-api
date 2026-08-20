# Validator metrics snapshot

Where the validator computes its numbers. Source: `src/hanwoo/services/validator/streamlit_app.py`.
Snapshot taken 2026-08-10 on branch `dev_validate` (base commit `e0727d7`, file had uncommitted changes).

## Matching — Top-1 accuracy

`run_matching()`, lines 452-531. Ground truth is the query filename stem; prediction is the
rank-1 gallery match name; comparison is exact string equality.

```python
# per query image (lines 490-495)
matches = data.get("matches", [])
top_match = matches[0] if matches else {}
expected = PurePosixPath(image.name).stem
predicted = top_match.get("name", "")
is_correct = predicted == expected
correct += int(is_correct)

# final (lines 521-530)
total = len(query)
metrics = {
    "accuracy": correct / total if total else 0.0,   # Top-1 accuracy
    "correct": correct,
    "total": total,
    "avg_round_trip_ms": round(round_trip_ms / total) if total else 0,
    "avg_compute_ms": round(compute_ms / total) if total else 0,
    "avg_preprocess_ms": round(preprocess_ms / total, 1) if total else 0,
    "elapsed_s": round(time.perf_counter() - start, 1),
}
```

`top_k` only affects the Top-K pairs column; accuracy always uses `matches[0]`.

## Dinomaly — recall

`run_dinomaly()`, lines 613-731. Calls `/infer` per image. Labels come from the folder layout:

- `test/abnormal/**` → expected anomaly
- `test/good/**` → expected normal

Positive class is anomaly.

```python
# per image (lines 681-698)
is_anomaly = bool(data.get("is_anomaly", False))   # server applies the /threshold value
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

# final (lines 713-729)
total = len(all_images)
precision = tp / (tp + fp) if tp + fp else 0.0
recall = tp / (tp + fn) if tp + fn else 0.0
f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
```

Two things that move these numbers:

- `is_anomaly` is decided server-side from the threshold PUT at line 630, so recall is not
  threshold-free. Compare runs only at the same threshold.
- 10 randomly sampled warmup `/infer` calls run first (line 649) and are excluded from timing,
  but those images are re-run in the scored loop like any other.

## Anomaly (non-Dinomaly) — for contrast

`run_anomaly()`, lines 569-579. No local metric computation — `/evaluate` returns
`total.{accuracy, precision, recall, f1}`, derived from mask labels under each category's
`labels/` folder, with `images2` counted as normal.
