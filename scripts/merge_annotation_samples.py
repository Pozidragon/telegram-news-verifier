from __future__ import annotations

import json

from app.config import Settings
from app.storage.json_repository import load_jsonl


def main() -> None:
    sample1_path = Settings.EXPERIMENTS_DIR / "annotation_sample.jsonl"
    sample2_path = Settings.EXPERIMENTS_DIR / "annotation_sample_2.jsonl"
    merged_path = Settings.EXPERIMENTS_DIR / "annotation_sample_merged.jsonl"

    sample1 = load_jsonl(sample1_path)
    sample2 = load_jsonl(sample2_path)

    merged = sample1 + sample2

    seen = set()
    unique_rows = []
    for row in merged:
        key = (row.get("telegram_channel"), row.get("telegram_message_id"))
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(row)

    with merged_path.open("w", encoding="utf-8") as f:
        for row in unique_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Merged {len(unique_rows)} rows into {merged_path}")


if __name__ == "__main__":
    main()