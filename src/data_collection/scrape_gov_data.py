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

# Some open-data portals (Socrata, data.gov edge, GitHub) throttle or block
# on the default "python-requests" user agent. This applies to every
# download request, whichever source issued it.
DOWNLOAD_HEADERS = {
    "User-Agent": "open-data-gov-scraper (public-data pipeline; polite requests)",
}

# If a whole page of search results yields zero downloadable tabular data,
# stop after this many consecutive empty pages so the loop cannot crawl an
# entire (huge) result set forever just because nothing was downloadable.
MAX_EMPTY_PAGES_PER_TERM = 10


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
        resp = requests.get(url, timeout=REQUEST_TIMEOUT_S, stream=True, headers=DOWNLOAD_HEADERS)
        resp.raise_for_status()

        # 1. Guard against HTML landing pages served with HTTP 200
        content_type = resp.headers.get("content-type", "").lower()
        if "text/html" in content_type or "application/xhtml+xml" in content_type:
            print(f"    SKIPPED {url}: Received HTML webpage instead of tabular data (Content-Type: {content_type})")
            return False

        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                f.write(chunk)

        # 2. Safeguard against servers returning HTML disguised as application/octet-stream
        with open(dest_path, "rb") as f:
            header_bytes = f.read(200).lower()
            if b"<html" in header_bytes or b"<!doctype html" in header_bytes:
                print(f"    SKIPPED {url}: Downloaded file content is an HTML document.")
                dest_path.unlink()  # Clean up the bad file
                return False

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
        empty_pages = 0
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

            page_this_downloadable = False

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
                    ext = ".csv" if any(k in fmt for k in ("csv", "comma")) or url_lower.endswith(".csv") else                           ".json" if "json" in fmt or url_lower.endswith(".json") else                           ".xlsx" if "xlsx" in fmt or url_lower.endswith(".xlsx") else                           ".tsv" if "tsv" in fmt or url_lower.endswith(".tsv") else                           ".xls" if "xls" in fmt or url_lower.endswith(".xls") else ".csv"

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
                    page_this_downloadable = True
                    print(f"  [{seen}/{max_datasets}] {pkg_name}")

            # If a whole page produced nothing downloadable, stop iterating
            # once too many come back empty rather than crawling indefinitely.
            if not page_this_downloadable:
                empty_pages += 1
                if empty_pages >= MAX_EMPTY_PAGES_PER_TERM:
                    print(f"  stopping: {empty_pages} consecutive pages with no downloadable data for '{term}'")
                    break
            else:
                empty_pages = 0

            start += rows


# --- data.norge.no (search.api.fellesdatakatalog.digdir.no) -------------
#
# Two-step fetch, confirmed against a real response (not a guess):
#   1. POST to the search API for a lightweight `hits` list (id + title only,
#      not full distribution/format info).
#   2. GET https://resource.api.fellesdatakatalog.digdir.no/v1/datasets/{id}
#      per hit for the full DCAT record (accessRights, distribution list).
# The resource.api hostname is confirmed real (shows up across multiple
# official EU Open Data Portal harvest records), but its own rate limit
# is unconfirmed -- REQUEST_DELAY_S_DATA_NORGE (paced to the SEARCH API's
# documented 10 req/min) is used for it too as a conservative default,
# not because that limit is confirmed to apply here specifically.

def _scrape_data_norge(sub_cfg: dict, staging_dir: Path, manifest_path: str) -> None:
    endpoint = sub_cfg["endpoint"]
    max_datasets = sub_cfg["max_datasets"]
    query_terms = sub_cfg.get("query_terms") or [""]
    seen = 0

    for term in query_terms:
        if seen >= max_datasets:
            break
        print(f"Searching data.norge.no for: '{term}'")
        page = 0
        page_size = 100

        while seen < max_datasets:
            time.sleep(REQUEST_DELAY_S_DATA_NORGE)

            payload = {"pagination": {"size": page_size, "page": page}}
            if term:
                payload["query"] = term

            try:
                resp = requests.post(
                    endpoint,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=REQUEST_TIMEOUT_S,
                )
                resp.raise_for_status()
            except requests.RequestException as e:  # noqa: BLE001
                print(f"  search failed: {e}")
                break

            body = resp.json()

            # Explicitly define hits from the search response body
            hits = body.get("hits") if isinstance(body, dict) else []
            if not hits or not isinstance(hits, list):
                break

            for ds_summary in hits:
                if seen >= max_datasets:
                    break

                if not isinstance(ds_summary, dict):
                    continue

                # The search API may expose the id directly or as an ES doc
                # (`_id` + `_source`). Prefer a direct `id` if present.
                dataset_id = ds_summary.get("id") or ds_summary.get("_id")
                if not dataset_id:
                    continue

                ds_title = _first_text(ds_summary.get("title")) or dataset_id

                # Two-step fetch: query the Resource API for the full DCAT record containing distributions
                resource_url = f"https://resource.api.fellesdatakatalog.digdir.no/v1/datasets/{dataset_id}"

                time.sleep(REQUEST_DELAY_S_DATA_NORGE)
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

                access_rights = full_ds.get("accessRights", {})
                license_info = access_rights.get("code") or access_rights.get("uri")

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

                    # NOT REQUEST_DELAY_S_DATA_NORGE here -- that delay is
                    # calibrated to fellesdatakatalog.digdir.no's documented
                    # 10 req/min limit, but this request goes to the
                    # distribution's own accessURL, which is frequently a
                    # completely different host. Pacing it against a rate
                    # limit that doesn't apply to this request just makes
                    # the run slower for no protective benefit -- at
                    # max_datasets=1500 the difference is roughly 5.4 hours
                    # vs 2.9 hours worst-case.
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

            page_info = body.get("page") or {}
            total_pages = page_info.get("totalPages", 1)

            if page + 1 >= total_pages:
                break

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
    covering the specific datasets listed in config.

    License note: "CC-BY-4.0" reflects OWID's own stated reuse policy for
    their processed/compiled datasets, but some underlying source data
    they aggregate (e.g. COVID case data originally from Johns Hopkins
    CSSE) may carry different upstream attribution requirements OWID's
    license doesn't fully capture."""
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
