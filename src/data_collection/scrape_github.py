"""
Searches GitHub's code search API for standalone .csv files and downloads
them. Registers per-file license via the repository's detected license
(GitHub's license detection API), since individual files don't carry
license metadata themselves.

Requires a GITHUB_TOKEN environment variable -- GitHub's code search API
is authenticated-only (no anonymous access) and has a low unauthenticated
rate limit that makes this impractical without one. Create a
read-only/public-repo-scope personal access token at
https://github.com/settings/tokens.

Usage:
    GITHUB_TOKEN=ghp_xxx python scrape_github.py --config config/data_sources.yaml
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import requests

from common import append_manifest, license_allowed, load_yaml_config, safe_filename

GITHUB_API = "https://api.github.com"
REQUEST_TIMEOUT_S = 30
REQUEST_DELAY_S = 2.0  # GitHub code search is rate-limited to 10 req/min even authenticated

# GitHub's code search API hard-caps at 1,000 results per query (10 pages of
# 100) -- page 11 returns HTTP 422, not more results. This isn't a crash (the
# existing except block below catches it and moves to the next query term),
# but it DOES silently cap you at ~1,000 files per term even if max_files is
# set higher. Stop cleanly at the documented limit instead of hitting the
# API's error response, and log it plainly so a low file count isn't mistaken
# for a bug -- add more, more specific query_terms in config/data_sources.yaml
# (e.g. combining extension:csv with size:, language:, or path: qualifiers)
# if you need more than ~1,000 files from this source.
MAX_PAGES_PER_QUERY = 10

_repo_license_cache: dict[str, str | None] = {}


def _headers() -> dict[str, str]:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError(
            "GITHUB_TOKEN environment variable is required for GitHub code search "
            "(unauthenticated rate limits are too low to be usable here)."
        )
    return {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}


def _get_repo_license(repo_full_name: str) -> str | None:
    if repo_full_name in _repo_license_cache:
        return _repo_license_cache[repo_full_name]

    try:
        resp = requests.get(
            f"{GITHUB_API}/repos/{repo_full_name}",
            headers=_headers(),
            timeout=REQUEST_TIMEOUT_S,
        )
        resp.raise_for_status()
        license_info = resp.json().get("license")
        spdx_id = license_info.get("spdx_id") if license_info else None
    except requests.RequestException:
        spdx_id = None

    _repo_license_cache[repo_full_name] = spdx_id
    return spdx_id


def scrape_github(cfg: dict) -> None:
    src_cfg = cfg["sources"]["github_csv_search"]
    if not src_cfg.get("enabled", True):
        print("github_csv_search source disabled in config, skipping")
        return

    staging_dir = Path(cfg["output"]["staging_dir"]) / "github"
    manifest_path = cfg["output"]["sources_manifest_path"]
    max_files = src_cfg["max_files"]
    min_stars = src_cfg.get("min_stars", 0)

    seen = 0
    for term in src_cfg["query_terms"]:
        if seen >= max_files:
            break
        query = term if min_stars == 0 else f"{term} stars:>={min_stars}"
        print(f"Searching GitHub code for: '{query}'")

        page = 1
        while seen < max_files and page <= MAX_PAGES_PER_QUERY:
            time.sleep(REQUEST_DELAY_S)
            try:
                resp = requests.get(
                    f"{GITHUB_API}/search/code",
                    headers=_headers(),
                    params={"q": query, "per_page": 100, "page": page},
                    timeout=REQUEST_TIMEOUT_S,
                )
                resp.raise_for_status()
            except requests.RequestException as e:  # noqa: BLE001
                print(f"  search failed: {e}")
                break

            items = resp.json().get("items", [])
            if not items:
                break

            for item in items:
                if seen >= max_files:
                    break
                repo_full_name = item["repository"]["full_name"]
                license_id = _get_repo_license(repo_full_name)

                if not license_allowed(license_id, "github_csv_search", cfg["license_policy"]):
                    continue

                download_url = item.get("html_url", "").replace(
                    "github.com", "raw.githubusercontent.com"
                ).replace("/blob/", "/")
                if not download_url:
                    continue

                fname = safe_filename(f"{repo_full_name}_{item['name']}")
                dest = staging_dir / fname
                time.sleep(REQUEST_DELAY_S)
                if not _download_file(download_url, dest):
                    continue

                append_manifest(
                    manifest_path,
                    {
                        "source": "github_csv_search",
                        "source_id": f"{repo_full_name}/{item['path']}",
                        "license": license_id,
                        "local_path": str(dest.resolve()),
                        "url": item.get("html_url"),
                    },
                )
                seen += 1
                print(f"  [{seen}/{max_files}] {repo_full_name}/{item['path']} (license={license_id})")

            page += 1

        if page > MAX_PAGES_PER_QUERY:
            print(
                f"  reached GitHub's 1,000-result cap for query '{query}' "
                f"({seen} total files collected so far across all terms) -- "
                f"add more/narrower query_terms in config if you need more from this source"
            )


def _download_file(url: str, dest_path: Path) -> bool:
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT_S)
        resp.raise_for_status()
    except requests.RequestException as e:  # noqa: BLE001
        print(f"    FAILED download {url}: {e}")
        return False

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_bytes(resp.content)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/data_sources.yaml")
    args = ap.parse_args()
    cfg = load_yaml_config(args.config)
    scrape_github(cfg)


if __name__ == "__main__":
    main()
