"""
Pulls tabular resources from government open-data portals: data.gov
(CKAN API), data.norge.no (DCAT API), Eurostat (SDMX bulk download), 
Socrata-powered portals (SODA API), and Our World in Data (GitHub).

Usage:
    python scrape_gov_data.py --config config/data_sources.yaml
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from urllib.parse import urlparse

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
        
    if src_cfg.get("socrata", {}).get("enabled"):
        _scrape_socrata(src_cfg["socrata"], staging_root / "socrata", manifest_path)

    if src_cfg.get("owid", {}).get("enabled"):
        _scrape_owid(src_cfg["owid"], staging_root / "owid", manifest_path)


def _download_resource(url: str, dest_path: Path) -> bool:
    # Skip download if the file already exists and is not empty
    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"    Skipping download, file already exists: {dest_path.name}")
        return True

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT_S, stream=True)
        resp.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                f.write(chunk)
        return True
    except (requests.RequestException, OSError) as e:  # noqa: BLE001
        print(f"    FAILED download {url}: {e}")
        if dest_path.exists():
            dest_path.unlink()  # Remove partial file on failure
        return False
    
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT_S, stream=True)
        resp.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                f.write(chunk)
        return True
    except (requests.RequestException, OSError) as e:  # noqa: BLE001
        print(f"    FAILED download {url}: {e}")
        if dest_path.exists():
            dest_path.unlink()  # Remove partial file on failure
        return False


# --- data.gov (CKAN) ---------------------------------------------------

def _scrape_data_gov(sub_cfg: dict, staging_dir: Path, manifest_path: str) -> None:
    endpoint = sub_cfg["endpoint"]
    max_datasets = sub_cfg["max_datasets"]
    seen = 0

    for term in sub_cfg.get("query_terms") or []:
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

            results = resp.json().get("result", {}).get("results") or []
            if not results:
                break

            for pkg in results:
                if seen >= max_datasets:
                    break
                license_id = pkg.get("license_id")
                pkg_name = pkg.get("name", "unknown")
                downloaded_any = False

                for resource in pkg.get("resources") or []:
                    fmt = (resource.get("format") or "").lower()
                    url = resource.get("url")
                    
                    if not url:
                        continue
                        
                    url_lower = url.lower()
                    
                    # Expanded check for format string and URL extension to catch unlabeled/misnamed CSVs
                    is_tabular = (
                        any(k in fmt for k in ("csv", "comma", "tsv", "xls", "json")) or
                        any(url_lower.endswith(ext) for ext in (".csv", ".tsv", ".xlsx", ".xls", ".json"))
                    )

                    if not is_tabular:
                        continue

                    # Standardize extension for the output file
                    ext = ".csv" if any(k in fmt for k in ("csv", "comma")) or url_lower.endswith(".csv") else \
                          ".json" if "json" in fmt or url_lower.endswith(".json") else \
                          ".xlsx" if "xlsx" in fmt or url_lower.endswith(".xlsx") else \
                          ".tsv" if "tsv" in fmt or url_lower.endswith(".tsv") else \
                          ".xls" if "xls" in fmt or url_lower.endswith(".xls") else ".csv"

                    fname = safe_filename(f"{pkg_name}_{resource.get('id', '')}") + ext
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
                    downloaded_any = True

                if downloaded_any:
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
        
        datasets = body.get("dataset") or body.get("data") or body.get("results") or []
        if not datasets:
            break

        for ds in datasets:
            if seen >= max_datasets:
                break
            ds_title = _first_text(ds.get("title")) or ds.get("id", "unknown")
            license_info = ds.get("license") or ds.get("accessRights")

            distributions = ds.get("distribution") or []
            if isinstance(distributions, dict):
                distributions = [distributions]

            downloaded_any = False
            for dist in distributions:
                if not isinstance(dist, dict):
                    continue

                fmt = (_first_text(dist.get("format")) or "").lower()
                media_type = (_first_text(dist.get("mediaType")) or "").lower()
                
                url = dist.get("accessURL") or dist.get("downloadURL")
                if not url:
                    continue
                    
                url = _first_text(url) if isinstance(url, (list, dict)) else url
                url_str = str(url)
                url_lower = url_str.lower()

                # DCAT media types are often full URIs. Broaden the check.
                is_tabular = (
                    any(k in fmt for k in ("csv", "comma", "tsv", "xls", "json")) or
                    any(k in media_type for k in ("csv", "comma", "tsv", "xls", "json")) or
                    any(url_lower.endswith(ext) for ext in (".csv", ".tsv", ".xlsx", ".xls", ".json"))
                )

                if not is_tabular:
                    continue

                # Cleanly extract extension from URL path without query params
                parsed_path = urlparse(url_str).path
                suffix = Path(parsed_path).suffix or ".csv"

                fname = safe_filename(ds_title) + suffix
                dest = staging_dir / safe_filename(ds_title) / fname
                time.sleep(REQUEST_DELAY_S)
                
                if not _download_resource(url_str, dest):
                    continue

                append_manifest(
                    manifest_path,
                    {
                        "source": "gov_data",
                        "source_id": f"data.norge.no/{ds_title}",
                        "license": str(license_info) if license_info else None,
                        "local_path": str(dest.resolve()),
                        "url": url_str,
                    },
                )
                downloaded_any = True

            if downloaded_any:
                seen += 1
                print(f"  [{seen}/{max_datasets}] {ds_title}")

        page += 1


def _first_text(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        first_val = next(iter(value.values()), None)
        return _first_text(first_val)
    if isinstance(value, list) and value:
        return _first_text(value[0])
    return None


# --- Eurostat ------------------------------------------------------------

def _scrape_eurostat(sub_cfg: dict, staging_dir: Path, manifest_path: str) -> None:
    endpoint = sub_cfg["endpoint"]
    codes = sub_cfg.get("dataset_codes") or []
    if not codes:
        print("No eurostat dataset_codes configured, skipping")
        return

    for code in codes:
        print(f"Fetching Eurostat dataset: {code}")
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
                "license": "CC-BY-4.0",
                "local_path": str(dest.resolve()),
                "url": url,
            },
        )
        print(f"  saved {code}")
        
        
# --- Socrata (SODA API) ------------------------------------------------

def _scrape_socrata(sub_cfg: dict, staging_dir: Path, manifest_path: str) -> None:
    endpoint = sub_cfg.get("endpoint", "https://api.us.socrata.com/api/catalog/v1")
    max_datasets = sub_cfg.get("max_datasets", 5)
    seen = 0

    for term in sub_cfg.get("query_terms") or []:
        if seen >= max_datasets:
            break
        print(f"Searching Socrata catalog for: '{term}'")
        try:
            resp = requests.get(
                endpoint,
                params={"q": term, "only": "datasets", "limit": max_datasets},
                timeout=REQUEST_TIMEOUT_S,
            )
            resp.raise_for_status()
        except requests.RequestException as e:  # noqa: BLE001
            print(f"  search failed: {e}")
            continue

        results = resp.json().get("results") or []
        for item in results:
            if seen >= max_datasets:
                break
                
            resource = item.get("resource") or {}
            metadata = item.get("metadata") or {}
            
            domain = metadata.get("domain") or resource.get("domain")
            dataset_id = resource.get("id")
            name = resource.get("name", "unknown")
            
            if not domain or not dataset_id:
                continue

            # Socrata provides a standardized CSV export endpoint for all platform datasets
            csv_url = f"https://{domain}/api/views/{dataset_id}/rows.csv?accessType=DOWNLOAD"
            
            fname = safe_filename(f"{name}_{dataset_id}") + ".csv"
            dest = staging_dir / safe_filename(domain) / fname
            time.sleep(REQUEST_DELAY_S)
            
            if not _download_resource(csv_url, dest):
                continue
                
            append_manifest(
                manifest_path,
                {
                    "source": "gov_data",
                    "source_id": f"socrata/{domain}/{dataset_id}",
                    "license": metadata.get("license") or resource.get("license", "unknown"),
                    "local_path": str(dest.resolve()),
                    "url": csv_url,
                },
            )
            seen += 1
            print(f"  [{seen}/{max_datasets}] {name}")
            

# --- Our World in Data (GitHub) ------------------------------------------

def _scrape_owid(sub_cfg: dict, staging_dir: Path, manifest_path: str) -> None:
    datasets = sub_cfg.get("datasets") or []
    if not datasets:
        print("No OWID datasets configured, skipping")
        return
        
    for ds in datasets:
        name = ds.get("name", "unknown_owid")
        url = ds.get("url")
        if not url:
            continue
            
        print(f"Fetching Our World in Data: {name}")
        
        fname = f"{safe_filename(name)}.csv"
        dest = staging_dir / fname
        time.sleep(REQUEST_DELAY_S)
        if not _download_resource(url, dest):
            continue

        append_manifest(
            manifest_path,
            {
                "source": "gov_data",
                "source_id": f"owid/{name}",
                "license": "CC-BY-4.0", 
                "local_path": str(dest.resolve()),
                "url": url,
            },
        )
        print(f"  saved {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/data_sources.yaml")
    args = ap.parse_args()
    cfg = load_yaml_config(args.config)
    scrape_gov_data(cfg)


if __name__ == "__main__":
    main()

