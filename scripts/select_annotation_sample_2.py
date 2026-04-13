from __future__ import annotations

import json
import random
from collections import defaultdict

from app.config import Settings
from app.storage.json_repository import load_jsonl


RANDOM_SEED = 99

TARGET_COUNTS = {
    "verified": 25,
    "uncertain": 35,
    "unverified": 40,
}


def main() -> None:
    random.seed(RANDOM_SEED)

    input_path = Settings.PROCESSED_DIR / "verification_results.jsonl"
    first_sample_path = Settings.EXPERIMENTS_DIR / "annotation_sample.jsonl"
    output_path = Settings.EXPERIMENTS_DIR / "annotation_sample_2.jsonl"

    all_rows = load_jsonl(input_path)
    first_sample = load_jsonl(first_sample_path)

    used_keys = {
        (row.get("telegram_channel"), row.get("telegram_message_id"))
        for row in first_sample
    }

    remaining_rows = [
        row for row in all_rows
        if (row.get("telegram_channel"), row.get("telegram_message_id")) not in used_keys
    ]

    by_status: dict[str, list[dict]] = defaultdict(list)
    for row in remaining_rows:
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in selected_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Selected {len(selected_rows)} new samples")
    for status in ("verified", "uncertain", "unverified"):
        count = sum(1 for row in selected_rows if row.get("predicted_status") == status)
        print(f"  {status}: {count}")

    print(f"Saved second annotation sample to {output_path}")


if __name__ == "__main__":
    main()