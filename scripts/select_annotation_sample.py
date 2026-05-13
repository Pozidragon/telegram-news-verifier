from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from app.config import Settings
from app.storage.json_repository import load_jsonl


RANDOM_SEED = 42

TARGET_COUNTS = {
    "uncertain": 50,   # take all available — only 15 labeled so far, need ~35 more
    "verified": 35,
    "unverified": 20,
}


def _already_labeled(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    return {(r["telegram_channel"], r["telegram_message_id"]) for r in load_jsonl(path)}


def main() -> None:
    random.seed(RANDOM_SEED)

    input_path = Settings.PROCESSED_DIR / "verification_results.jsonl"
    output_jsonl_path = Settings.EXPERIMENTS_DIR / "annotation_sample_3.jsonl"

    rows = load_jsonl(input_path)

    labeled = _already_labeled(Settings.EXPERIMENTS_DIR / "labeled_sample.jsonl")
    rows = [r for r in rows if (r.get("telegram_channel"), r.get("telegram_message_id")) not in labeled]
    print(f"Excluded {len(labeled)} already-labeled posts. {len(rows)} candidates remain.")

    by_status: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        status = row.get("status", "")
        by_status[status].append(row)

    selected_rows: list[dict] = []

    for status, target_count in TARGET_COUNTS.items():
        candidates = by_status.get(status, [])
        if not candidates:
            continue

        if len(candidates) <= target_count:
            sampled = candidates
        else:
            sampled = random.sample(candidates, target_count)

        for row in sampled:
            selected_rows.append({
                "telegram_channel": row.get("telegram_channel", ""),
                "telegram_message_id": row.get("telegram_message_id", ""),
                "telegram_text": row.get("telegram_text", ""),
                "matched_news_title": row.get("matched_news_title", ""),
                "matched_news_url": row.get("matched_news_url", ""),
                "predicted_status": row.get("status", ""),
                "similarity_score": row.get("similarity_score", 0.0),
                "keyword_overlap": row.get("keyword_overlap", 0),
                "true_label": "",
                "notes": "",
            })

    selected_rows.sort(
        key=lambda x: (
            x.get("predicted_status", ""),
            -float(x.get("similarity_score", 0.0)),
            x.get("telegram_message_id", 0),
        )
    )

    output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    with output_jsonl_path.open("w", encoding="utf-8") as f:
        for row in selected_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Selected {len(selected_rows)} samples")
    for status in ("verified", "uncertain", "unverified"):
        count = sum(1 for row in selected_rows if row.get("predicted_status") == status)
        print(f"  {status}: {count}")

    print(f"Saved annotation sample to {output_jsonl_path}")
    print("Next steps:")
    print("  1. Run merge_annotation_samples.py to add this batch to annotation_sample_merged.jsonl")
    print("  2. Fill in true_label for each row in annotation_sample_3.jsonl")
    print("  3. Run build_labeled_sample.py to rebuild labeled_sample.jsonl")


if __name__ == "__main__":
    main()