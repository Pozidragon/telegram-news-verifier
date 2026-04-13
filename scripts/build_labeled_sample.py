from __future__ import annotations

import json

from app.config import Settings
from app.storage.json_repository import load_jsonl


def main() -> None:
    input_path = Settings.EXPERIMENTS_DIR / "annotation_sample_merged.jsonl"
    output_path = Settings.EXPERIMENTS_DIR / "labeled_sample.jsonl"

    rows = load_jsonl(input_path)

    filtered_rows = []
    for row in rows:
        true_label = row.get("true_label", "").strip()
        if true_label not in {"verified", "uncertain", "unverified"}:
            continue

        filtered_rows.append({
            "telegram_channel": row.get("telegram_channel", ""),
            "telegram_message_id": row.get("telegram_message_id", 0),
            "true_label": true_label,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in filtered_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved {len(filtered_rows)} labeled rows to {output_path}")


if __name__ == "__main__":
    main()