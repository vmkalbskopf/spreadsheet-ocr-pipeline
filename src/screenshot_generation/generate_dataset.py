"""
Orchestrates screenshot generation across all source CSVs assigned to this
shard.

Two fixes vs. the first draft, both worth calling out since they shape the
control flow:

1. `cfg.resolution` is sampled per-screenshot in variation_sampler.py, but
   Xvfb's resolution is fixed at process start -- it can't change mid-run.
   Restarting Xvfb per screenshot would work but is wasteful (Xvfb startup
   is ~1s, and most CSVs get several variants). Instead: pre-sample every
   (csv, variant) work item's config up front, GROUP by resolution, and
   run one Xvfb+soffice session per resolution group. Screenshot order is
   no longer CSV-major -- it's resolution-major -- which is irrelevant to
   correctness (the manifest records everything needed downstream) but
   worth knowing if you're tailing progress output expecting CSV order.

2. Long-running soffice/UNO sessions are prone to memory leaks and socket
   hangs (LibreOffice bug reports on this are long-standing). Restarting
   soffice every RESTART_EVERY_N_DOCS documents bounds the damage instead
   of hoping a multi-hour shard doesn't hit it.

Usage:
    python generate_dataset.py \
        --csv-dir data/raw_csv --out-dir data/screenshots \
        --variation-config config/screenshot_variation.yaml \
        --shard-index 0 --n-shards 1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from capture import VirtualDisplay, capture_screenshot
from render_libreoffice import apply_config, close_doc, connect_to_soffice, load_csv
from variation_sampler import ScreenshotConfig, load_variation_config, sample_config

SOFFICE_PORT = 2002
SOFFICE_STARTUP_TIMEOUT_S = 30
RESTART_EVERY_N_DOCS = 50


@dataclass
class WorkItem:
    csv_path: Path
    variant_idx: int
    cfg: ScreenshotConfig


def build_work_items(csv_paths: list[Path], variation_cfg: dict, n_variants: int) -> list[WorkItem]:
    items = []
    for csv_path in csv_paths:
        for variant_idx in range(n_variants):
            seed = hash((str(csv_path), variant_idx)) & 0xFFFFFFFF
            cfg = sample_config(variation_cfg, seed=seed)
            if cfg.software != "libreoffice_calc":
                # Excel path not yet implemented (needs Windows/COM automation)
                # -- fall back to LibreOffice so the pipeline stays runnable.
                cfg.software = "libreoffice_calc"
            items.append(WorkItem(csv_path=csv_path, variant_idx=variant_idx, cfg=cfg))
    return items


def group_by_resolution(items: list[WorkItem]) -> dict[str, list[WorkItem]]:
    groups: dict[str, list[WorkItem]] = defaultdict(list)
    for item in items:
        groups[item.cfg.resolution].append(item)
    return groups


def start_soffice(display: str, port: int = SOFFICE_PORT) -> subprocess.Popen:
    return subprocess.Popen(
        [
            "soffice",
            f"--accept=socket,host=localhost,port={port};urp;",
            "--norestore",
            "--nologo",
        ],
        env={"DISPLAY": display},
    )


def wait_for_soffice(port: int = SOFFICE_PORT, timeout_s: int = SOFFICE_STARTUP_TIMEOUT_S):
    start = time.time()
    last_err = None
    while time.time() - start < timeout_s:
        try:
            return connect_to_soffice(port=port)
        except Exception as e:  # noqa: BLE001 -- broad by design, we're polling
            last_err = e
            time.sleep(1.0)
    raise RuntimeError(f"soffice did not accept connections within {timeout_s}s: {last_err}")


def restart_soffice(old_proc: subprocess.Popen, display: str):
    old_proc.terminate()
    try:
        old_proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        old_proc.kill()
    new_proc = start_soffice(display)
    desktop, ctx = wait_for_soffice()
    return new_proc, desktop, ctx


def process_item(item: WorkItem, out_dir: Path, desktop, display: str, manifest_f) -> None:
    doc = load_csv(desktop, str(item.csv_path.resolve()))
    try:
        apply_config(doc, item.cfg)
        time.sleep(0.2)  # let the UI settle before capture

        stem = item.csv_path.stem
        raw_path = out_dir / f"{stem}_v{item.variant_idx}_raw.png"
        final_path = out_dir / f"{stem}_v{item.variant_idx}.png"
        capture_screenshot(item.cfg, display, "LibreOffice Calc", raw_path, final_path)
        raw_path.unlink(missing_ok=True)

        manifest_f.write(
            json.dumps(
                {
                    "csv_path": str(item.csv_path),
                    "screenshot_path": str(final_path),
                    "variant_index": item.variant_idx,
                    "config": item.cfg.to_dict(),
                }
            )
            + "\n"
        )
    finally:
        close_doc(doc)


def process_resolution_group(
    resolution: str, items: list[WorkItem], out_dir: Path, manifest_f
) -> None:
    print(f"-- resolution {resolution}: {len(items)} screenshots --")
    with VirtualDisplay(resolution=resolution) as display:
        soffice_proc = start_soffice(display)
        try:
            desktop, _ctx = wait_for_soffice()
            docs_since_restart = 0

            for i, item in enumerate(items):
                if docs_since_restart >= RESTART_EVERY_N_DOCS:
                    print(f"  restarting soffice after {docs_since_restart} docs")
                    soffice_proc, desktop, _ctx = restart_soffice(soffice_proc, display)
                    docs_since_restart = 0

                print(f"  [{i+1}/{len(items)}] {item.csv_path.name} v{item.variant_idx}")
                try:
                    process_item(item, out_dir, desktop, display, manifest_f)
                except Exception as e:  # noqa: BLE001
                    # Log and continue -- one malformed/slow document shouldn't
                    # kill a multi-hour shard. A hung soffice call specifically
                    # (vs. a clean exception) isn't caught by this try/except --
                    # if you see a shard silently stall rather than error out,
                    # that's the failure mode a hard per-doc timeout via a
                    # watchdog thread or subprocess-based worker would catch;
                    # not implemented here, flagged as a follow-up.
                    print(f"    FAILED: {e}")
                docs_since_restart += 1
        finally:
            soffice_proc.terminate()
            try:
                soffice_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                soffice_proc.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--variation-config", type=Path, default=Path("config/screenshot_variation.yaml")
    )
    ap.add_argument("--n-variants", type=int, default=None, help="override n_variants_per_csv")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    variation_cfg = load_variation_config(args.variation_config)
    n_variants = args.n_variants or variation_cfg["n_variants_per_csv"]

    all_csvs = sorted(args.csv_dir.glob("*.csv"))
    shard_csvs = all_csvs[args.shard_index :: args.n_shards]
    print(f"Shard {args.shard_index}/{args.n_shards}: {len(shard_csvs)} CSVs to process")

    work_items = build_work_items(shard_csvs, variation_cfg, n_variants)
    groups = group_by_resolution(work_items)
    print(f"{len(work_items)} total screenshots across {len(groups)} resolution groups")

    manifest_path = args.out_dir / f"manifest_shard{args.shard_index}.jsonl"
    with open(manifest_path, "w") as manifest_f:
        for resolution, items in groups.items():
            process_resolution_group(resolution, items, args.out_dir, manifest_f)


if __name__ == "__main__":
    main()
