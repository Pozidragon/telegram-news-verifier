"""Grid search over entity bonus and corroboration boost magnitudes plus thresholds.

Uses stored top5_base_scores / top5_entity_overlaps from verification results to
replay different bonus configurations without re-running the full verification pipeline.
"""
from __future__ import annotations

import json
from itertools import product

from app.config import Settings
from app.storage.json_repository import load_jsonl
from app.evaluation.metrics import accuracy_score, classification_report_3class

LABELED_SAMPLE_PATH = Settings.EXPERIMENTS_DIR / "labeled_sample.jsonl"
TFIDF_RESULTS_PATH = Settings.PROCESSED_DIR / "verification_results.jsonl"
SEMANTIC_RESULTS_PATH = Settings.PROCESSED_DIR / "semantic_verification_results.jsonl"


def apply_bonuses(
    top5_base: list[float],
    top5_ent: list[int],
    ent_bonus: float,
    corr_boost: float,
    t_uncertain: float,
) -> float:
    if not top5_base:
        return 0.0
    adjusted = [
        min(1.0, b * (1 + ent_bonus * e)) if e > 0 else b
        for b, e in zip(top5_base, top5_ent)
    ]
    best = adjusted[0]
    soft = t_uncertain * 0.7
    n_corr = sum(1 for s in adjusted[1:] if s >= soft)
    return min(1.0, best + corr_boost * n_corr)


def reclassify(score: float, tv: float, tu: float) -> str:
    if score >= tv:
        return "verified"
    if score >= tu:
        return "uncertain"
    return "unverified"


def run_search(
    label_key: str,
    results_path,
    name: str,
    ent_bonuses: list[float],
    corr_boosts: list[float],
    thresholds: list[float],
) -> dict:
    labeled = load_jsonl(LABELED_SAMPLE_PATH)
    predicted = {
        (r["telegram_channel"], r["telegram_message_id"]): r
        for r in load_jsonl(results_path)
    }

    samples = []
    for item in labeled:
        key = (item["telegram_channel"], item["telegram_message_id"])
        pred = predicted.get(key)
        if pred and pred.get("top5_base_scores"):
            samples.append((
                item["true_label"],
                pred["top5_base_scores"],
                pred.get("top5_entity_overlaps", [0] * len(pred["top5_base_scores"])),
            ))

    print(f"\n{name}: matched {len(samples)} samples with bonus data")

    best: dict = {"macro_f1": -1.0}
    results = []

    for ent_b, corr_b, tv, tu in product(ent_bonuses, corr_boosts, thresholds, thresholds):
        if tu >= tv:
            continue
        y_true = [s[0] for s in samples]
        y_pred = [
            reclassify(apply_bonuses(s[1], s[2], ent_b, corr_b, tu), tv, tu)
            for s in samples
        ]
        acc = accuracy_score(y_true, y_pred)
        report = classification_report_3class(y_true, y_pred)
        ver_f1 = round(report.get("verified", {}).get("f1", 0.0), 4)
        unc_f1 = round(report.get("uncertain", {}).get("f1", 0.0), 4)
        unv_f1 = round(report.get("unverified", {}).get("f1", 0.0), 4)
        macro_f1 = round((ver_f1 + unc_f1 + unv_f1) / 3, 4)
        entry = {
            "entity_bonus": ent_b,
            "corr_boost": corr_b,
            "threshold_verified": tv,
            "threshold_uncertain": tu,
            "accuracy": round(acc, 4),
            "macro_f1": macro_f1,
            "verified_f1": ver_f1,
            "uncertain_f1": unc_f1,
            "unverified_f1": unv_f1,
        }
        results.append(entry)
        if macro_f1 > best["macro_f1"]:
            best = entry

    results.sort(key=lambda x: (-x["macro_f1"], -x["verified_f1"]))

    print(f"Best: ent_bonus={best['entity_bonus']}  corr_boost={best['corr_boost']}")
    print(f"      tv={best['threshold_verified']}  tu={best['threshold_uncertain']}")
    print(f"  Accuracy:      {best['accuracy']:.4f}")
    print(f"  Macro F1:      {best['macro_f1']:.4f}")
    print(f"  Verified  F1:  {best['verified_f1']:.4f}")
    print(f"  Uncertain F1:  {best['uncertain_f1']:.4f}")
    print(f"  Unverified F1: {best['unverified_f1']:.4f}")

    header = f"{'ent_b':>6} {'c_b':>5} {'tv':>6} {'tu':>6} {'acc':>7} {'mac_f1':>8} {'ver_f1':>8} {'unc_f1':>8} {'unv_f1':>8}"
    print(f"\nTop 10 by macro-F1:")
    print(header)
    print("-" * len(header))
    for r in results[:10]:
        print(
            f"{r['entity_bonus']:>6.2f} {r['corr_boost']:>5.2f} "
            f"{r['threshold_verified']:>6.2f} {r['threshold_uncertain']:>6.2f} "
            f"{r['accuracy']:>7.4f} {r['macro_f1']:>8.4f} {r['verified_f1']:>8.4f} "
            f"{r['uncertain_f1']:>8.4f} {r['unverified_f1']:>8.4f}"
        )

    return {"best": best, "all_results": results[:200]}


def main() -> None:
    ent_bonuses = [round(v * 0.02, 2) for v in range(0, 9)]   # 0.00 .. 0.16
    corr_boosts = [round(v * 0.01, 2) for v in range(0, 9)]   # 0.00 .. 0.08
    thresholds  = [round(v * 0.05, 2) for v in range(4, 20)]  # 0.20 .. 0.95

    tfidf_out = run_search(
        "tfidf", TFIDF_RESULTS_PATH, "TF-IDF",
        ent_bonuses, corr_boosts, thresholds,
    )
    out = Settings.EXPERIMENTS_DIR / "bonus_grid_search_tfidf.json"
    out.write_text(json.dumps(tfidf_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved to {out}")

    sem_out = run_search(
        "semantic", SEMANTIC_RESULTS_PATH, "Semantic",
        ent_bonuses, corr_boosts, thresholds,
    )
    out = Settings.EXPERIMENTS_DIR / "bonus_grid_search_semantic.json"
    out.write_text(json.dumps(sem_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
