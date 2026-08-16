"""
Searches Kaggle for datasets matching config query terms, downloads
license-eligible ones, and registers each downloaded file in the staging
manifest for normalize_to_csv.py to pick up.

Requires Kaggle API credentials: either ~/.kaggle/kaggle.json, or the
KAGGLE_USERNAME / KAGGLE_KEY environment variables. See
https://www.kaggle.com/docs/api for how to generate a token.

Usage:
    python scrape_kaggle.py --config config/data_sources.yaml
"""

from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path

from common import append_manifest, license_allowed, load_yaml_config, safe_filename


def _safe_extractall(zf: zipfile.ZipFile, dest_dir: Path) -> None:
    """extractall() trusts member paths verbatim -- a malicious entry like
    '../../etc/passwd' or an absolute path can write outside dest_dir
    ("Zip Slip"). Kaggle's own uploads are unlikely to be hostile, but an
    automated scraper pulling from public search shouldn't assume that.
    Resolve each member's destination and refuse anything that escapes
    dest_dir, rather than trusting the archive's internal paths."""
    dest_root = dest_dir.resolve()
    for member in zf.infolist():
        member_path = (dest_dir / member.filename).resolve()
        if os.path.commonpath([dest_root, member_path]) != str(dest_root):
            print(f"    skipping unsafe zip member (path traversal): {member.filename}")
            continue
        if member.is_dir():
            member_path.mkdir(parents=True, exist_ok=True)
            continue
        member_path.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(member) as src, open(member_path, "wb") as dst:
            dst.write(src.read())


def scrape_kaggle(cfg: dict) -> None:
    src_cfg = cfg["sources"]["kaggle"]
    if not src_cfg.get("enabled", True):
        print("kaggle source disabled in config, skipping")
        return

    # Imported lazily so the rest of the data_collection package doesn't
    # hard-require the kaggle package (and its credential check at import
    # time) for people only running the gov_data or github scrapers.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    staging_dir = Path(cfg["output"]["staging_dir"]) / "kaggle"
    staging_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cfg["output"]["sources_manifest_path"]

    max_datasets = src_cfg["max_datasets"]
    seen = 0

    for term in src_cfg["query_terms"]:
        if seen >= max_datasets:
            break
        print(f"Searching Kaggle for: '{term}'")
        # Kaggle's search API paginates; iterate pages until exhausted or
        # we hit max_datasets, rather than assuming one page is enough.
        page = 1
        while seen < max_datasets:
            datasets = api.dataset_list(search=term, page=page, file_type="csv")
            if not datasets:
                break

            for ds in datasets:
                if seen >= max_datasets:
                    break

                license_id = getattr(ds, "licenseName", None)
                if not license_allowed(license_id, "kaggle", cfg["license_policy"]):
                    continue

                ds_ref = ds.ref  # "owner/dataset-slug"
                dest_dir = staging_dir / safe_filename(ds_ref)
                dest_dir.mkdir(parents=True, exist_ok=True)

                try:
                    api.dataset_download_files(ds_ref, path=str(dest_dir), unzip=False, quiet=True)
                except Exception as e:  # noqa: BLE001 -- one bad dataset shouldn't halt the scrape
                    print(f"  FAILED download {ds_ref}: {e}")
                    continue

                _extract_and_register(dest_dir, ds_ref, license_id, manifest_path)
                seen += 1
                print(f"  [{seen}/{max_datasets}] {ds_ref} (license={license_id})")

            page += 1

    print(f"Kaggle: registered files from {seen} datasets")


def _extract_and_register(
    dest_dir: Path, ds_ref: str, license_id: str | None, manifest_path: str
) -> None:
    zip_files = list(dest_dir.glob("*.zip"))
    for zf_path in zip_files:
        try:
            with zipfile.ZipFile(zf_path) as zf:
                _safe_extractall(zf, dest_dir)
        except zipfile.BadZipFile:
            print(f"    bad zip, skipping: {zf_path}")
            continue
        zf_path.unlink()

    # Register every tabular-looking file extracted, not just top-level
    # CSVs -- Kaggle datasets frequently ship xlsx/ods/tsv alongside or
    # instead of csv, and normalize_to_csv.py handles the conversion.
    tabular_exts = {".csv", ".tsv", ".xlsx", ".xls", ".ods", ".json"}
    for f in dest_dir.rglob("*"):
        if f.is_file() and f.suffix.lower() in tabular_exts:
            append_manifest(
                manifest_path,
                {
                    "source": "kaggle",
                    "source_id": ds_ref,
                    "license": license_id,
                    "local_path": str(f.resolve()),
                    "url": f"https://www.kaggle.com/datasets/{ds_ref}",
                },
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/data_sources.yaml")
    args = ap.parse_args()
    cfg = load_yaml_config(args.config)
    scrape_kaggle(cfg)


if __name__ == "__main__":
    main()
