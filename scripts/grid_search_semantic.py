"""Grid search over semantic verifier thresholds using saved similarity scores."""
from __future__ import annotations

import json

from app.config import Settings
from app.storage.json_repository import load_jsonl
from app.evaluation.metrics import accuracy_score, classification_report_3class

LABELED_SAMPLE_PATH = Settings.EXPERIMENTS_DIR / "labeled_sample.jsonl"
SEMANTIC_RESULTS_PATH = Settings.PROCESSED_DIR / "semantic_verification_results.jsonl"


def reclassify(score: float, t_verified: float, t_uncertain: float) -> str:
    if score >= t_verified:
        return "verified"
    elif score >= t_uncertain:
        return "uncertain"
    return "unverified"


def main() -> None:
    labeled = load_jsonl(LABELED_SAMPLE_PATH)
    predicted = load_jsonl(SEMANTIC_RESULTS_PATH)

    pred_index = {
        (r["telegram_channel"], r["telegram_message_id"]): r for r in predicted
    }

    pairs: list[tuple[str, float]] = []
    for item in labeled:
        key = (item["telegram_channel"], item["telegram_message_id"])
        pred = pred_index.get(key)
        if pred:
            pairs.append((item["true_label"], pred["similarity_score"]))

    print(f"Matched samples: {len(pairs)}")

    thresholds = [round(v * 0.05, 2) for v in range(5, 20)]  # 0.25 .. 0.95
    best = {"accuracy": -1.0}

    results = []
    for tv in thresholds:
        for tu in thresholds:
            if tu >= tv:
                continue
            y_true = [p[0] for p in pairs]
            y_pred = [reclassify(p[1], tv, tu) for p in pairs]
            acc = accuracy_score(y_true, y_pred)
            report = classification_report_3class(y_true, y_pred)
            ver_f1 = round(report.get("verified", {}).get("f1", 0.0), 4)
            unc_f1 = round(report.get("uncertain", {}).get("f1", 0.0), 4)
            unv_f1 = round(report.get("unverified", {}).get("f1", 0.0), 4)
            macro_f1 = round((ver_f1 + unc_f1 + unv_f1) / 3, 4)
            results.append({
                "threshold_verified": tv,
                "threshold_uncertain": tu,
                "accuracy": round(acc, 4),
                "macro_f1": macro_f1,
                "verified_f1": ver_f1,
                "uncertain_f1": unc_f1,
                "unverified_f1": unv_f1,
            })
            if macro_f1 > best.get("macro_f1", -1.0):
                best = results[-1]

    results.sort(key=lambda x: (-x["macro_f1"], -x["verified_f1"]))

    print(f"\nBest thresholds: verified={best['threshold_verified']}  uncertain={best['threshold_uncertain']}")
    print(f"  Accuracy:      {best['accuracy']:.4f}")
    print(f"  Macro F1:      {best['macro_f1']:.4f}")
    print(f"  Verified  F1:  {best['verified_f1']:.4f}")
    print(f"  Uncertain F1:  {best['uncertain_f1']:.4f}")
    print(f"  Unverified F1: {best['unverified_f1']:.4f}")

    print("\nTop 10 combinations by macro-F1:")
    header = f"{'tv':>6} {'tu':>6} {'acc':>7} {'mac_f1':>8} {'ver_f1':>8} {'unc_f1':>8} {'unv_f1':>8}"
    print(header)
    print("-" * len(header))
    for r in results[:10]:
        print(
            f"{r['threshold_verified']:>6.2f} {r['threshold_uncertain']:>6.2f} "
            f"{r['accuracy']:>7.4f} {r['macro_f1']:>8.4f} {r['verified_f1']:>8.4f} "
            f"{r['uncertain_f1']:>8.4f} {r['unverified_f1']:>8.4f}"
        )

    out_path = Settings.EXPERIMENTS_DIR / "semantic_grid_search.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"best": best, "all_results": results}, f, ensure_ascii=False, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
