"""
Converts every staged download (any of csv/tsv/xlsx/xls/ods/json) into one
or more canonical CSV files, and records row/col counts for each -- needed
by dedup_and_bin.py's shape-binning stage.

A single xlsx/ods file with multiple sheets produces one output CSV per
sheet, since each sheet is really a separate table for our purposes (a
screenshot of a spreadsheet shows one sheet, not a merged view of all of
them).

Usage:
    python normalize_to_csv.py --config config/data_sources.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from common import append_manifest, read_manifest, safe_filename

MIN_ROWS = 5   # below this, not worth a training example (mostly-empty file, header-only, etc.)
MIN_COLS = 2   # a single-column file isn't really a "spreadsheet screenshot" table


def normalize_all(cfg: dict) -> None:
    source_records = read_manifest(cfg["output"]["sources_manifest_path"])
    normalized_dir = Path(cfg["output"]["normalized_dir"])
    normalized_manifest_path = cfg["output"]["normalized_manifest_path"]

    # Idempotency: if this has been run before, don't reprocess files
    # already in the normalized manifest -- scraping runs incrementally
    # over time and re-normalizing everything on every run would waste
    # significant time once the staging directory is large.
    already_done = {r["source_local_path"] for r in read_manifest(normalized_manifest_path)}

    n_ok, n_skipped, n_failed = 0, 0, 0
    for record in source_records:
        local_path = record["local_path"]
        if local_path in already_done:
            n_skipped += 1
            continue

        try:
            outputs = _normalize_one(Path(local_path), normalized_dir)
        except Exception as e:  # noqa: BLE001 -- one malformed file shouldn't halt the run
            print(f"FAILED normalizing {local_path}: {e}")
            n_failed += 1
            continue

        for out_path, n_rows, n_cols in outputs:
            if n_rows < MIN_ROWS or n_cols < MIN_COLS:
                out_path.unlink(missing_ok=True)
                continue
            append_manifest(
                normalized_manifest_path,
                {
                    "source_local_path": local_path,
                    "source": record["source"],
                    "source_id": record["source_id"],
                    "license": record.get("license"),
                    "url": record.get("url"),
                    "normalized_path": str(out_path.resolve()),
                    "n_rows": n_rows,
                    "n_cols": n_cols,
                },
            )
            n_ok += 1

    print(f"Normalized: {n_ok} tables written, {n_skipped} already done, {n_failed} failed")


def _normalize_one(src_path: Path, out_dir: Path) -> list[tuple[Path, int, int]]:
    ext = src_path.suffix.lower()
    stem = safe_filename(src_path.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    if ext == ".csv":
        return [_copy_as_canonical_csv(src_path, out_dir / f"{stem}.csv", delimiter=",")]
    if ext == ".tsv":
        return [_copy_as_canonical_csv(src_path, out_dir / f"{stem}.csv", delimiter="\t")]
    if ext in (".xlsx", ".xls", ".ods"):
        return _excel_to_csvs(src_path, out_dir, stem)
    if ext == ".json":
        return _json_to_csv(src_path, out_dir / f"{stem}.csv")

    raise ValueError(f"Unsupported extension: {ext}")


def _copy_as_canonical_csv(src_path: Path, out_path: Path, delimiter: str) -> tuple[Path, int, int]:
    # Re-write through csv.reader/writer (rather than a raw byte copy) so
    # downstream code can always assume standard comma-delimited, properly
    # quoted CSV regardless of the source's original delimiter or quoting
    # style.
    encoding = _detect_encoding(src_path)
    with open(src_path, encoding=encoding, errors="replace", newline="") as f:
        rows = list(csv.reader(f, delimiter=delimiter))

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    n_cols = max((len(r) for r in rows), default=0)
    return out_path, len(rows), n_cols


def _detect_encoding(path: Path) -> str:
    # Government and international data sources especially (data.norge.no)
    # are inconsistent about UTF-8 vs. Latin-1 -- sniff rather than assume.
    try:
        import chardet

        with open(path, "rb") as f:
            raw = f.read(65536)
        detected = chardet.detect(raw)
        return detected["encoding"] or "utf-8"
    except ImportError:
        return "utf-8"


def _excel_to_csvs(src_path: Path, out_dir: Path, stem: str) -> list[tuple[Path, int, int]]:
    import pandas as pd

    engine = "odf" if src_path.suffix.lower() == ".ods" else None
    sheets = pd.read_excel(src_path, sheet_name=None, header=None, engine=engine)

    outputs = []
    for sheet_name, df in sheets.items():
        if df.empty:
            continue
        out_path = out_dir / f"{stem}__{safe_filename(str(sheet_name))}.csv"
        df.to_csv(out_path, index=False, header=False)
        outputs.append((out_path, df.shape[0], df.shape[1]))
    return outputs


def _json_to_csv(src_path: Path, out_path: Path) -> list[tuple[Path, int, int]]:
    with open(src_path, encoding="utf-8", errors="replace") as f:
        data = json.load(f)

    # Only handle the common "list of flat dicts" shape (a JSON array of
    # records) -- nested/irregular JSON isn't really "spreadsheet-shaped"
    # data and would need bespoke flattening logic per source, not
    # something to guess generically here.
    if isinstance(data, dict):
        # Some APIs wrap the record list in a top-level key
        for value in data.values():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                data = value
                break

    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        return []

    fieldnames: list[str] = []
    for record in data:
        for k in record.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in data:
            writer.writerow(record)

    return [(out_path, len(data) + 1, len(fieldnames))]  # +1 for header row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/data_sources.yaml")
    args = ap.parse_args()
    from common import load_yaml_config

    cfg = load_yaml_config(args.config)
    normalize_all(cfg)


if __name__ == "__main__":
    main()
