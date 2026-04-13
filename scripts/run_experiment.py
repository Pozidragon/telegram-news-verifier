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
VERIFICATION_RESULTS_PATH = Settings.PROCESSED_DIR / "verification_results.jsonl"


def main() -> None:
    labeled = load_jsonl(LABELED_SAMPLE_PATH)
    predicted = load_jsonl(VERIFICATION_RESULTS_PATH)

    pred_index = {
        (item["telegram_channel"], item["telegram_message_id"]): item
        for item in predicted
    }

    y_true: list[str] = []
    y_pred: list[str] = []

    merged_rows = []

    for item in labeled:
        key = (item["telegram_channel"], item["telegram_message_id"])
        pred = pred_index.get(key)

        if not pred:
            continue

        y_true.append(item["true_label"])
        y_pred.append(pred["status"])

        merged_rows.append({
            "telegram_channel": item["telegram_channel"],
            "telegram_message_id": item["telegram_message_id"],
            "true_label": item["true_label"],
            "predicted_label": pred["status"],
            "similarity_score": pred.get("similarity_score", 0.0),
            "keyword_overlap": pred.get("keyword_overlap", 0),
            "matched_news_title": pred.get("matched_news_title"),
        })

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

    out_path = Settings.EXPERIMENTS_DIR / "experiment_results.json"
    out_payload = {
        "evaluated_samples": len(y_true),
        "accuracy": accuracy,
        "binary_verified_vs_rest": binary_metrics,
        "report_3class": report_3class,
        "confusion_matrix_3class": matrix_3class,
        "merged_rows": merged_rows,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out_payload, f, ensure_ascii=False, indent=2)

    print(f"Saved experiment results to {out_path}")


if __name__ == "__main__":
    main()