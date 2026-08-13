"""
Pulls tabular resources from government open-data portals: data.gov
(CKAN API), data.norge.no (DCAT API), and Eurostat (SDMX bulk download).
These three have genuinely different APIs -- no shared client library --
so each gets its own function, but they write to the same staging
manifest so downstream stages don't need to know which portal a file
came from.

No API key required for any of the three at time of writing.

Usage:
    python scrape_gov_data.py --config config/data_sources.yaml
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import requests

from common import append_manifest, load_yaml_config, safe_filename

REQUEST_TIMEOUT_S = 30
REQUEST_DELAY_S = 0.5  # be a polite scraper; none of these portals are rate-limit-documented


def scrape_gov_data(cfg: dict) -> None:
    src_cfg = cfg["sources"]["gov_data"]
    if not src_cfg.get("enabled", True):
        print("gov_data source disabled in config, skipping")
        return

    staging_root = Path(cfg["output"]["staging_dir"]) / "gov_data"
    manifest_path = cfg["output"]["sources_manifest_path"]

    if src_cfg.get("data_gov", {}).get("enabled"):
        _scrape_data_gov(src_cfg["data_gov"], staging_root / "data_gov", manifest_path)

    if src_cfg.get("data_norge", {}).get("enabled"):
        _scrape_data_norge(src_cfg["data_norge"], staging_root / "data_norge", manifest_path)

    if src_cfg.get("eurostat", {}).get("enabled"):
        _scrape_eurostat(src_cfg["eurostat"], staging_root / "eurostat", manifest_path)


def _download_resource(url: str, dest_path: Path) -> bool:
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT_S, stream=True)
        resp.raise_for_status()
    except requests.RequestException as e:  # noqa: BLE001
        print(f"    FAILED download {url}: {e}")
        return False

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            f.write(chunk)
    return True


# --- data.gov (CKAN) ---------------------------------------------------

def _scrape_data_gov(sub_cfg: dict, staging_dir: Path, manifest_path: str) -> None:
    endpoint = sub_cfg["endpoint"]
    max_datasets = sub_cfg["max_datasets"]
    seen = 0

    for term in sub_cfg["query_terms"]:
        if seen >= max_datasets:
            break
        print(f"Searching data.gov for: '{term}'")
        start = 0
        rows = 100
        while seen < max_datasets:
            try:
                resp = requests.get(
                    endpoint,
                    params={"q": term, "rows": rows, "start": start},
                    timeout=REQUEST_TIMEOUT_S,
                )
                resp.raise_for_status()
            except requests.RequestException as e:  # noqa: BLE001
                print(f"  search failed: {e}")
                break

            results = resp.json().get("result", {}).get("results", [])
            if not results:
                break

            for pkg in results:
                if seen >= max_datasets:
                    break
                license_id = pkg.get("license_id")
                pkg_name = pkg.get("name", "unknown")

                for resource in pkg.get("resources", []):
                    fmt = (resource.get("format") or "").lower()
                    if fmt not in ("csv", "tsv", "xlsx", "xls", "json"):
                        continue
                    url = resource.get("url")
                    if not url:
                        continue

                    fname = safe_filename(f"{pkg_name}_{resource.get('id', '')}") + f".{fmt}"
                    dest = staging_dir / safe_filename(pkg_name) / fname
                    time.sleep(REQUEST_DELAY_S)
                    if not _download_resource(url, dest):
                        continue

                    append_manifest(
                        manifest_path,
                        {
                            "source": "gov_data",
                            "source_id": f"data.gov/{pkg_name}",
                            "license": license_id,
                            "local_path": str(dest.resolve()),
                            "url": url,
                        },
                    )
                seen += 1
                print(f"  [{seen}/{max_datasets}] {pkg_name}")

            start += rows


# --- data.norge.no (DCAT) ----------------------------------------------

def _scrape_data_norge(sub_cfg: dict, staging_dir: Path, manifest_path: str) -> None:
    endpoint = sub_cfg["endpoint"]
    max_datasets = sub_cfg["max_datasets"]
    seen = 0
    page = 0

    print("Fetching data.norge.no dataset catalog")
    while seen < max_datasets:
        try:
            resp = requests.get(
                endpoint, params={"page": page, "size": 100}, timeout=REQUEST_TIMEOUT_S
            )
            resp.raise_for_status()
        except requests.RequestException as e:  # noqa: BLE001
            print(f"  fetch failed: {e}")
            break

        body = resp.json()
        # DCAT-AP-NO responses are typically JSON-LD with a "dataset" list --
        # exact key names have shifted across API versions historically, so
        # this defends against a couple of likely shapes rather than
        # assuming one fixed schema.
        datasets = body.get("dataset") or body.get("data") or body.get("results") or []
        if not datasets:
            break

        for ds in datasets:
            if seen >= max_datasets:
                break
            ds_title = _first_text(ds.get("title")) or ds.get("id", "unknown")
            license_info = ds.get("license") or ds.get("accessRights")

            distributions = ds.get("distribution", [])
            for dist in distributions:
                fmt = (_first_text(dist.get("format")) or "").lower()
                if not any(k in fmt for k in ("csv", "tsv", "xlsx", "json")):
                    continue
                url = dist.get("accessURL") or dist.get("downloadURL")
                if not url:
                    continue
                url = _first_text(url) if isinstance(url, (list, dict)) else url

                fname = safe_filename(ds_title) + (Path(url).suffix or ".csv")
                dest = staging_dir / safe_filename(ds_title) / fname
                time.sleep(REQUEST_DELAY_S)
                if not _download_resource(url, dest):
                    continue

                append_manifest(
                    manifest_path,
                    {
                        "source": "gov_data",
                        "source_id": f"data.norge.no/{ds_title}",
                        "license": str(license_info) if license_info else None,
                        "local_path": str(dest.resolve()),
                        "url": url,
                    },
                )
            seen += 1
            print(f"  [{seen}/{max_datasets}] {ds_title}")

        page += 1


def _first_text(value):
    """DCAT fields are sometimes plain strings, sometimes {lang: text}
    dicts (multi-language), sometimes lists of either. Best-effort
    extraction of a single display string."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return next(iter(value.values()), None)
    if isinstance(value, list) and value:
        return _first_text(value[0])
    return None


# --- Eurostat ------------------------------------------------------------

def _scrape_eurostat(sub_cfg: dict, staging_dir: Path, manifest_path: str) -> None:
    endpoint = sub_cfg["endpoint"]
    codes = sub_cfg.get("dataset_codes", [])
    if not codes:
        print("No eurostat dataset_codes configured, skipping")
        return

    for code in codes:
        print(f"Fetching Eurostat dataset: {code}")
        # SDMX 2.1 REST API: request CSV format directly rather than the
        # default SDMX-XML, to avoid needing an SDMX parser in this pipeline.
        url = f"{endpoint}/{code}?format=SDMX-CSV"
        dest = staging_dir / f"{safe_filename(code)}.csv"
        time.sleep(REQUEST_DELAY_S)
        if not _download_resource(url, dest):
            continue

        append_manifest(
            manifest_path,
            {
                "source": "gov_data",
                "source_id": f"eurostat/{code}",
                # Eurostat's standard reuse policy is CC-BY-4.0; verify
                # per-dataset if that matters for your institution.
                "license": "CC-BY-4.0",
                "local_path": str(dest.resolve()),
                "url": url,
            },
        )
        print(f"  saved {code}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/data_sources.yaml")
    args = ap.parse_args()
    cfg = load_yaml_config(args.config)
    scrape_gov_data(cfg)


if __name__ == "__main__":
    main()
