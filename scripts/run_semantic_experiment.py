from __future__ import annotations

import json

from app.config import Settings
from app.storage.json_repository import load_jsonl
from app.evaluation.metrics import (
    accuracy_score,
    precision_recall_f1_binary,
    classification_report_3class,
    confusion_matrix_3class,
)

LABELED_SAMPLE_PATH = Settings.EXPERIMENTS_DIR / "labeled_sample.jsonl"
VERIFICATION_RESULTS_PATH = Settings.PROCESSED_DIR / "semantic_verification_results.jsonl"
OUTPUT_PATH = Settings.EXPERIMENTS_DIR / "semantic_experiment_results.json"


def main() -> None:
    labeled = load_jsonl(LABELED_SAMPLE_PATH)
    predicted = load_jsonl(VERIFICATION_RESULTS_PATH)

    pred_index = {
        (item["telegram_channel"], item["telegram_message_id"]): item
        for item in predicted
    }

    y_true: list[str] = []
    y_pred: list[str] = []

    for item in labeled:
        key = (item["telegram_channel"], item["telegram_message_id"])
        pred = pred_index.get(key)
        if not pred:
            continue

        y_true.append(item["true_label"])
        y_pred.append(pred["status"])

    if not y_true:
        print("No overlapping labeled samples found.")
        return

    accuracy = accuracy_score(y_true, y_pred)

    y_true_binary = ["verified" if x == "verified" else "not_verified" for x in y_true]
    y_pred_binary = ["verified" if x == "verified" else "not_verified" for x in y_pred]

    binary_metrics = precision_recall_f1_binary(
        y_true_binary,
        y_pred_binary,
        positive_label="verified",
    )

    report_3class = classification_report_3class(y_true, y_pred)
    matrix_3class = confusion_matrix_3class(y_true, y_pred)

    print(f"Evaluated samples: {len(y_true)}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Binary metrics for 'verified' vs rest:")
    print(json.dumps(binary_metrics, ensure_ascii=False, indent=2))
    print("3-class report:")
    print(json.dumps(report_3class, ensure_ascii=False, indent=2))
    print("3-class confusion matrix:")
    print(json.dumps(matrix_3class, ensure_ascii=False, indent=2))

    results = {
        "evaluated_samples": len(y_true),
        "accuracy": round(accuracy, 10),
        "binary_verified_vs_rest": binary_metrics,
        "report_3class": report_3class,
        "confusion_matrix_3class": matrix_3class,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()