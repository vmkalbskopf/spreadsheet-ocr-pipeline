"""
Merges per-shard screenshot manifests (from screenshot_generation/) into
train/val splits, pairing each screenshot with the CSV text it should
reproduce. Splits by SOURCE CSV, not by screenshot, so that different
rendering variants of the same source table never leak across the
train/val boundary -- otherwise val accuracy would be inflated by the
model having memorized that specific table's content from a sibling variant.

Usage:
    python prepare_dataset.py \
        --shard-manifests-glob "data/screenshots/manifest_shard*.jsonl" \
        --out-dir data/manifests --val-frac 0.05
"""

from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path


def load_all_records(pattern: str) -> list[dict]:
    records = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            for line in f:
                records.append(json.loads(line))
    return records


def build_training_record(record: dict) -> dict:
    csv_path = Path(record["csv_path"])
    with open(csv_path) as f:
        csv_text = f.read()

    # Simple, explicit instruction. Keep it fixed across the whole dataset --
    # varying the prompt wording is a separate augmentation axis you can add
    # later once the core visual task is working reliably.
    prompt = (
        "This image shows a spreadsheet open in spreadsheet software. "
        "Transcribe the visible table exactly as a CSV, using commas as "
        "delimiters. Include the header row. Output only the CSV, no commentary."
    )

    return {
        "image_path": record["screenshot_path"],
        "prompt": prompt,
        "target": csv_text,
        "source_csv": str(csv_path),   # kept for the group-aware split below
        "render_config": record["config"],
    }


def split_by_source(records: list[dict], val_frac: float, seed: int = 0) -> tuple[list, list]:
    sources = sorted({r["source_csv"] for r in records})
    rng = random.Random(seed)
    rng.shuffle(sources)

    n_val_sources = max(1, int(len(sources) * val_frac))
    val_sources = set(sources[:n_val_sources])

    train = [r for r in records if r["source_csv"] not in val_sources]
    val = [r for r in records if r["source_csv"] in val_sources]
    return train, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-manifests-glob", type=str, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--val-frac", type=float, default=0.05)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_records = load_all_records(args.shard_manifests_glob)
    print(f"Loaded {len(raw_records)} screenshot records")

    training_records = [build_training_record(r) for r in raw_records]
    train, val = split_by_source(training_records, args.val_frac)
    print(f"Split: {len(train)} train / {len(val)} val (grouped by source CSV)")

    with open(args.out_dir / "train.jsonl", "w") as f:
        for r in train:
            f.write(json.dumps(r) + "\n")

    with open(args.out_dir / "val.jsonl", "w") as f:
        for r in val:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
