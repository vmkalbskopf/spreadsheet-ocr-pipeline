"""
Shared helpers used across scrape_kaggle.py, scrape_gov_data.py,
scrape_github.py, normalize_to_csv.py, and dedup_and_bin.py. Kept in one
place so manifest schema and hashing logic can't silently drift between
the scrapers.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


def load_yaml_config(path: str | Path) -> dict:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def safe_filename(text: str, max_len: int = 120) -> str:
    """Collapses an arbitrary dataset/repo/file identifier into a
    filesystem-safe stem. Not guaranteed unique on its own -- callers
    should prefix/suffix with a hash or counter when collisions matter."""
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip())
    return cleaned[:max_len].strip("_") or "unnamed"


def append_manifest(path: str | Path, record: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def read_manifest(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_manifest(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def content_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def normalized_content_hash(rows: list[list[str]]) -> str:
    """Hash over normalized table content: sorted column order is NOT
    applied here (column order is semantically meaningful for a table, and
    scrambling it would create false dedup matches between unrelated
    tables that happen to share the same header set) -- normalization is
    limited to whitespace stripping and consistent line joining, so two
    downloads of the literal same file with different trailing whitespace
    or line-ending conventions still hash identically."""
    normalized = "\n".join(",".join(cell.strip() for cell in row) for row in rows)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def header_dtype_signature(header: list[str], sample_rows: list[list[str]]) -> frozenset[str]:
    """A coarse signature used for near-dup detection: lowercased column
    names paired with a rough inferred dtype for that column, sampled from
    the first few rows. Two datasets with the same signature are probably
    the same table re-uploaded/re-exported (e.g. a Kaggle dataset that also
    shows up as a GitHub CSV), even if formatting differs enough that the
    exact content hash doesn't match."""

    def infer_dtype(values: list[str]) -> str:
        non_empty = [v for v in values if v.strip()]
        if not non_empty:
            return "empty"
        if all(_is_int(v) for v in non_empty):
            return "int"
        if all(_is_float(v) for v in non_empty):
            return "float"
        return "str"

    sig = set()
    for col_idx, name in enumerate(header):
        col_values = [row[col_idx] for row in sample_rows if col_idx < len(row)]
        sig.add(f"{name.strip().lower()}:{infer_dtype(col_values)}")
    return frozenset(sig)


def _is_int(v: str) -> bool:
    try:
        int(v)
        return True
    except ValueError:
        return False


def _is_float(v: str) -> bool:
    try:
        float(v)
        return True
    except ValueError:
        return False


def jaccard_similarity(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def license_allowed(
    license_id: str | None,
    source_name: str,
    license_policy: dict,
) -> bool:
    """Applies config/data_sources.yaml's license_policy. Government
    sources are trusted at the portal level even with missing per-dataset
    license metadata; other sources drop on missing/unrecognized license
    by default (configurable via unknown_license_action)."""
    allowed = set(license_policy.get("allowed_licenses", []))
    trusted_sources = set(license_policy.get("trust_portal_level_for_sources", []))
    unknown_action = license_policy.get("unknown_license_action", "drop")

    if source_name in trusted_sources:
        return True

    if not license_id:
        return unknown_action == "keep"

    return license_id in allowed
