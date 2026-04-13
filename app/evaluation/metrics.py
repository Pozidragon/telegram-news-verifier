from __future__ import annotations


def accuracy_score(y_true: list[str], y_pred: list[str]) -> float:
    if not y_true:
        return 0.0
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def precision_recall_f1_binary(
    y_true: list[str],
    y_pred: list[str],
    positive_label: str = "verified",
) -> dict[str, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == positive_label and p == positive_label)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != positive_label and p == positive_label)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == positive_label and p != positive_label)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def classification_report_3class(y_true: list[str], y_pred: list[str]) -> dict[str, dict[str, float]]:
    labels = sorted(set(y_true) | set(y_pred))
    report: dict[str, dict[str, float]] = {}

    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        report[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(1 for t in y_true if t == label),
        }

    return report


def confusion_matrix_3class(y_true: list[str], y_pred: list[str]) -> dict[str, dict[str, int]]:
    labels = sorted(set(y_true) | set(y_pred))
    matrix = {true_label: {pred_label: 0 for pred_label in labels} for true_label in labels}

    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1

    return matrix