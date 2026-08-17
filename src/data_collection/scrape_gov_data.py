"""
Pulls tabular resources from government/public open-data portals: data.gov
(CKAN API via api.gsa.gov), data.norge.no (Elasticsearch-based search API),
Eurostat (SDMX bulk download), Socrata-powered portals (SODA Discovery
API), and Our World in Data (direct GitHub CSV downloads).

data.gov requires an API key (x-api-key header) -- register free at
https://api.data.gov/signup/ and set DATA_GOV_API_KEY in your environment.
Falls back to the shared "DEMO_KEY" if unset, which works for a smoke test
but is rate-limited across everyone using it, not suitable for a real run.
Disabled by default in config/data_sources.yaml -- flip data_gov.enabled
to true and set the env var if you want this source; the other four don't
need any registration.

data.norge.no's search API needs no auth but is rate limited to 10 req/min
(burst 20) -- see REQUEST_DELAY_S_DATA_NORGE below.

Socrata's Discovery API and OWID's GitHub-hosted CSVs need no auth.

Usage:
    python scrape_gov_data.py --config config/data_sources.yaml
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from common import append_manifest, load_yaml_config, safe_filename

REQUEST_TIMEOUT_S = 30
REQUEST_DELAY_S = 0.5  # be a polite scraper; data.gov, Eurostat, Socrata, OWID aren't rate-limit-documented
REQUEST_DELAY_S_DATA_NORGE = 6.5  # 10 req/min documented limit -> >=6s between requests, with margin


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
    # Resume-friendly: skip re-downloading a file that already succeeded on
    # a previous run. Only trusts a non-empty file -- a zero-byte file means
    # a prior attempt was interrupted before writing anything real, so it's
    # still worth retrying rather than treated as "already done".
    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"    skipping, already downloaded: {dest_path.name}")
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
            dest_path.unlink()  # remove partial file on failure
        return False


# --- data.gov (CKAN, via api.gsa.gov) -----------------------------------

def _scrape_data_gov(sub_cfg: dict, staging_dir: Path, manifest_path: str) -> None:
    endpoint = sub_cfg["endpoint"]
    max_datasets = sub_cfg["max_datasets"]
    seen = 0

    api_key = os.environ.get("DATA_GOV_API_KEY", "DEMO_KEY")
    if api_key == "DEMO_KEY":
        print(
            "  WARNING: using shared DEMO_KEY for data.gov -- rate limited across "
            "everyone using it. Register a free key at https://api.data.gov/signup/ "
            "and set DATA_GOV_API_KEY for a real scrape run."
        )
    headers = {"x-api-key": api_key}

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
                    headers=headers,
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


# --- data.norge.no (search.api.fellesdatakatalog.digdir.no) -------------

def _scrape_data_norge(sub_cfg: dict, staging_dir: Path, manifest_path: str) -> None:
    """Uses the current POST-based Elasticsearch search API
    (https://data.norge.no/en/technical/api/search), which replaced the old
    GET /api/dcat/datasets endpoint. No auth required.

    Response-shape caveat: the documented request format is confirmed
    against the official docs (query + pagination in a JSON body), but the
    exact response JSON schema for individual hits was NOT independently
    verified against a live response. The parsing below defends against a
    couple of plausible field-name variants (mirroring the old DCAT-AP-NO
    field names, since the search index is built from that same underlying
    data) -- confirm against an actual response and adjust field names if
    dataset/distribution info doesn't come through."""
    endpoint = sub_cfg["endpoint"]  # expected: .../search/datasets
    max_datasets = sub_cfg["max_datasets"]
    query_terms = sub_cfg.get("query_terms") or [""]  # empty string = match-all-ish
    seen = 0

    for ds_summary in hits:
        if seen >= max_datasets:
            break
            
        if not isinstance(ds_summary, dict):
            continue

        dataset_id = ds_summary.get("id")
        if not dataset_id:
            continue
            
        ds_title = _first_text(ds_summary.get("title")) or dataset_id
        
        # --- THE TWO-STEP FETCH ---
        # Fellesdatakatalog Resource API endpoint for full DCAT records
        resource_url = f"https://resource.api.fellesdatakatalog.digdir.no/v1/datasets/{dataset_id}"
        
        time.sleep(REQUEST_DELAY_S_DATA_NORGE)  # Respect rate limit for the secondary GET request
        try:
            detail_resp = requests.get(
                resource_url,
                headers={"Accept": "application/json"},
                timeout=REQUEST_TIMEOUT_S,
            )
            detail_resp.raise_for_status()
        except requests.RequestException as e:  # noqa: BLE001
            print(f"  Failed to fetch detailed record for {dataset_id}: {e}")
            continue
            
        full_ds = detail_resp.json()
        
        # Extract metadata from the COMPLETE record, not the summary
        access_rights = full_ds.get("accessRights", {})
        license_info = access_rights.get("code") or access_rights.get("uri")

        # The full record will contain the actual distributions array
        distributions = full_ds.get("distribution") or []
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

            # Check FDK's normalized format list if available on the distribution
            fdk_formats = dist.get("fdkFormat") or []
            if isinstance(fdk_formats, list):
                for f_obj in fdk_formats:
                    if isinstance(f_obj, dict):
                        fmt += f" {str(f_obj.get('code', ''))} {str(f_obj.get('type', ''))}".lower()

            is_tabular = (
                any(k in fmt for k in ("csv", "comma", "tsv", "xls", "json")) or
                any(k in media_type for k in ("csv", "comma", "tsv", "xls", "json")) or
                any(url_lower.endswith(ext) for ext in (".csv", ".tsv", ".xlsx", ".xls", ".json"))
            )

            if not is_tabular:
                continue

            parsed_path = urlparse(url_str).path
            suffix = Path(parsed_path).suffix or ".csv"

            fname = safe_filename(ds_title) + suffix
            dest = staging_dir / safe_filename(ds_title) / fname
            
            time.sleep(REQUEST_DELAY_S_DATA_NORGE) # Be polite before downloading

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
    """DCAT fields are sometimes plain strings, sometimes {lang: text}
    dicts (multi-language), sometimes lists of either. Best-effort
    extraction of a single display string."""
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


# --- Socrata (SODA Discovery API) ----------------------------------------

def _scrape_socrata(sub_cfg: dict, staging_dir: Path, manifest_path: str) -> None:
    """Searches Socrata's cross-portal Discovery API (covers hundreds of US
    federal/state/city open-data portals running on the Socrata platform --
    data.cdc.gov, data.cityofnewyork.us, data.texas.gov, etc. in one query)
    and downloads matches via each portal's standard CSV export endpoint.
    No auth required.

    `only=dataset` (singular) is confirmed correct against the R
    `socratadata` package, a direct wrapper around this same API -- the
    plural form is a plausible-looking but wrong guess that would likely
    silently return zero/wrong results."""
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
                params={"q": term, "only": "dataset", "limit": max_datasets},
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

            # Socrata's standard CSV export endpoint, present for every
            # platform dataset regardless of which portal it's hosted on.
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


# --- Our World in Data (direct GitHub CSV downloads) ---------------------

def _scrape_owid(sub_cfg: dict, staging_dir: Path, manifest_path: str) -> None:
    """Direct downloads from OWID's own GitHub repos rather than an API --
    simpler and more robust than scraping their site, at the cost of only
    covering the specific datasets listed in config. All three configured
    URLs verified live (HTTP 200, real CSV content, tens of thousands of
    rows) as of writing.

    License note: "CC-BY-4.0" reflects OWID's own stated reuse policy for
    their processed/compiled datasets, but some underlying source data
    they aggregate (e.g. COVID case data originally from Johns Hopkins
    CSSE) may carry different upstream attribution requirements OWID's
    license doesn't fully capture. Fine for this pipeline's purposes, but
    worth knowing if you need precise upstream attribution."""
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
