"""
Final stage of data collection: takes every normalized CSV, applies the
license gate, deduplicates (exact content hash + near-dup header/dtype
signature), bins by (row_count, col_count) per config/data_sources.yaml's
shape_bins, and caps each bin at target_per_bin so the final training set
has roughly even coverage across table sizes rather than being dominated
by whatever shape happens to be most abundant online.

Writes accepted files into data/raw_csv/ (flat, hash-prefixed filenames)
and the final manifest that screenshot_generation/generate_dataset.py
reads from.

Usage:
    python dedup_and_bin.py --config config/data_sources.yaml
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import shutil
from collections import defaultdict
from pathlib import Path

from common import (
    header_dtype_signature,
    jaccard_similarity,
    license_allowed,
    load_yaml_config,
    normalized_content_hash,
    read_manifest,
    write_manifest,
)


def compute_log_bin_edges(lo: float, hi: float, n_bins: int) -> list[float]:
    log_lo, log_hi = math.log(lo), math.log(hi)
    step = (log_hi - log_lo) / n_bins
    return [math.exp(log_lo + i * step) for i in range(n_bins + 1)]


def bin_index(value: float, edges: list[float]) -> int:
    for i in range(len(edges) - 1):
        if edges[i] <= value < edges[i + 1]:
            return i
    return len(edges) - 2 if value >= edges[-1] else 0


def load_table(path: Path) -> tuple[list[str], list[list[str]]]:
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return [], []
    return rows[0], rows[1:]


def _file_content_hash(path: Path) -> str:
    import hashlib

    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _write_truncated_csv(
    src: Path, dest: Path, max_rows: int | None, max_cols: int | None
) -> tuple[int, int, bool]:
    """Writes src to dest, truncated to at most max_rows lines (header
    included) and max_cols columns. Returns (n_rows_written, n_cols_written,
    was_truncated) so the caller can update manifest shape metadata and
    track how much of the dataset this actually affects.

    Truncating rows is straightforward (just stop reading after max_rows
    lines). Truncating columns means every row -- including the header --
    gets sliced to the first max_cols fields, which keeps header/data
    alignment intact."""
    with open(src, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = []
        was_truncated = False
        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                was_truncated = True
                break
            if max_cols is not None and len(row) > max_cols:
                row = row[:max_cols]
                was_truncated = True
            rows.append(row)

    with open(dest, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    n_cols_written = max((len(r) for r in rows), default=0)
    return len(rows), n_cols_written, was_truncated


def dedup(records: list[dict], near_dup_threshold: float) -> tuple[list[dict], dict]:
    """Two-stage dedup: exact content hash first (cheap, catches byte-identical
    re-uploads), then near-dup signature comparison within remaining records
    (catches the same table re-exported with different formatting/whitespace,
    e.g. a Kaggle dataset that's also posted as a raw GitHub CSV)."""
    seen_hashes: set[str] = set()
    exact_deduped: list[dict] = []
    stats = {"exact_dup_dropped": 0, "near_dup_dropped": 0, "failed_to_load": 0}

    signatures: list[tuple[dict, frozenset]] = []
    for r in records:
        path = Path(r["normalized_path"])
        try:
            header, data_rows = load_table(path)
        except (OSError, csv.Error):
            stats["failed_to_load"] += 1
            continue
        if not header:
            stats["failed_to_load"] += 1
            continue

        h = normalized_content_hash([header] + data_rows)
        if h in seen_hashes:
            stats["exact_dup_dropped"] += 1
            continue
        seen_hashes.add(h)
        exact_deduped.append(r)

        sample = data_rows[:20]  # enough to infer dtype without hashing huge files
        signatures.append((r, header_dtype_signature(header, sample)))

    # Near-dup pass: O(n^2) signature comparison. Fine at the scale this
    # pipeline targets (thousands, not millions, of source tables) --
    # revisit with LSH/minhash bucketing if the source pool grows much
    # larger than that.
    keep_mask = [True] * len(signatures)
    for i in range(len(signatures)):
        if not keep_mask[i]:
            continue
        for j in range(i + 1, len(signatures)):
            if not keep_mask[j]:
                continue
            sim = jaccard_similarity(signatures[i][1], signatures[j][1])
            if sim >= near_dup_threshold:
                keep_mask[j] = False
                stats["near_dup_dropped"] += 1

    near_deduped = [signatures[i][0] for i in range(len(signatures)) if keep_mask[i]]
    return near_deduped, stats


def bin_and_cap(
    records: list[dict], shape_cfg: dict, seed: int = 0
) -> tuple[list[dict], dict]:
    row_edges = compute_log_bin_edges(
        shape_cfg["rows"]["min"], shape_cfg["rows"]["max"], shape_cfg["rows"]["n_bins"]
    )
    col_edges = compute_log_bin_edges(
        shape_cfg["cols"]["min"], shape_cfg["cols"]["max"], shape_cfg["cols"]["n_bins"]
    )
    target_per_bin = shape_cfg["target_per_bin"]

    buckets: dict[tuple[int, int], list[dict]] = defaultdict(list)
    out_of_range = 0
    for r in records:
        n_rows, n_cols = r["n_rows"], r["n_cols"]
        if not (shape_cfg["rows"]["min"] <= n_rows) or not (shape_cfg["cols"]["min"] <= n_cols):
            out_of_range += 1
            continue
        ri = bin_index(n_rows, row_edges)
        ci = bin_index(n_cols, col_edges)
        buckets[(ri, ci)].append(r)

    rng = random.Random(seed)
    accepted = []
    bin_report = {}
    for key, items in buckets.items():
        rng.shuffle(items)
        capped = items[:target_per_bin]
        accepted.extend(capped)
        bin_report[key] = {"available": len(items), "kept": len(capped)}

    stats = {"out_of_range_dropped": out_of_range, "n_bins_populated": len(buckets), "bin_report": bin_report}
    return accepted, stats


def finalize(cfg: dict, accepted: list[dict]) -> None:
    raw_csv_dir = Path(cfg["output"]["raw_csv_dir"])
    raw_csv_dir.mkdir(parents=True, exist_ok=True)

    render_cap = cfg.get("render_cap", {})
    max_rows = render_cap.get("max_rows")
    max_cols = render_cap.get("max_cols")

    final_records = []
    n_truncated = 0
    for r in accepted:
        src = Path(r["normalized_path"])
        # Content-hash-prefixed filename: guarantees uniqueness across
        # sources without needing to track a global counter, and makes it
        # obvious if the same file somehow ends up copied twice. Hashed
        # over the actual file bytes (not shape/metadata) so two
        # different-content files of the same row/col shape don't collide.
        file_hash = _file_content_hash(src)[:12]
        dest_name = f"{file_hash}_{src.name}"
        dest = raw_csv_dir / dest_name

        n_rows, n_cols = r["n_rows"], r["n_cols"]
        if not dest.exists():
            if max_rows or max_cols:
                n_rows, n_cols, was_truncated = _write_truncated_csv(src, dest, max_rows, max_cols)
                n_truncated += was_truncated
            else:
                shutil.copy2(src, dest)

        final_records.append(
            {
                "path": str(dest.resolve()),
                "source": r["source"],
                "source_id": r["source_id"],
                "license": r.get("license"),
                "url": r.get("url"),
                # Reflects the TRUNCATED shape actually written to dest, not
                # the original source file's shape -- this is what
                # screenshot rendering and prepare_dataset.py's training
                # target both see, so it's what downstream code should
                # reason about (e.g. shape-bin analysis of the final set).
                "n_rows": n_rows,
                "n_cols": n_cols,
            }
        )

    write_manifest(cfg["output"]["final_manifest_path"], final_records)
    print(f"Wrote {len(final_records)} accepted CSVs to {raw_csv_dir}")
    if max_rows or max_cols:
        print(
            f"Applied render_cap (max_rows={max_rows}, max_cols={max_cols}): "
            f"{n_truncated}/{len(final_records)} files were larger than the cap and got truncated"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/data_sources.yaml")
    args = ap.parse_args()
    cfg = load_yaml_config(args.config)

    normalized_records = read_manifest(cfg["output"]["normalized_manifest_path"])
    print(f"Loaded {len(normalized_records)} normalized tables")

    license_policy = cfg["license_policy"]
    license_ok = [
        r for r in normalized_records
        if license_allowed(r.get("license"), r["source"], license_policy)
    ]
    print(f"After license gate: {len(license_ok)} ({len(normalized_records) - len(license_ok)} dropped)")

    deduped, dedup_stats = dedup(license_ok, cfg["dedup"]["near_dup_threshold"])
    print(f"After dedup: {len(deduped)} -- {dedup_stats}")

    accepted, bin_stats = bin_and_cap(deduped, cfg["shape_bins"])
    print(f"After shape binning/capping: {len(accepted)} -- {bin_stats}")

    finalize(cfg, accepted)


if __name__ == "__main__":
    main()
