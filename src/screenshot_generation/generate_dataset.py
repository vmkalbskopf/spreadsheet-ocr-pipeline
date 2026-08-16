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

3. A per-document hard timeout (SIGALRM) guards against a specific
   observed failure mode: an interactive LibreOffice dialog (e.g. the
   Text Import wizard, if FilterName is ever missing/wrong -- see
   render_libreoffice.load_csv's docstring) blocking forever waiting for
   a click that will never come under Xvfb. This presents as near-zero
   CPU usage with no further log output -- easy to mistake for something
   else. On timeout, soffice is force-restarted and the shard continues
   rather than hanging indefinitely.

4. Work items are grouped by (software, resolution), not just resolution.
   "excel"-sampled items route to render_onlyoffice.py (OnlyOffice Desktop
   Editors, driven via xdotool since there's no UNO-equivalent scripting
   API for it) rather than silently falling back to LibreOffice. See that
   module's docstring for important caveats -- it was written without
   access to a running OnlyOffice instance to verify against, so window
   titles and keyboard shortcuts are documented best-guesses, not
   confirmed-correct.

Usage:
    python generate_dataset.py \
        --csv-dir data/raw_csv --out-dir data/screenshots \
        --variation-config config/screenshot_variation.yaml \
        --shard-index 0 --n-shards 1
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from capture import VirtualDisplay, capture_screenshot
from render_libreoffice import apply_config, close_doc, connect_to_soffice, load_csv
from render_onlyoffice import (
    WINDOW_TITLE_HINT,
    apply_config_onlyoffice,
    close_onlyoffice,
    find_onlyoffice_window,
    launch_onlyoffice,
)
from variation_sampler import ScreenshotConfig, load_variation_config, sample_config

SOFFICE_PORT = 2002
SOFFICE_STARTUP_TIMEOUT_S = 30
RESTART_EVERY_N_DOCS = 50
PER_DOC_TIMEOUT_S = 60


class DocumentTimeoutError(Exception):
    pass


def _alarm_handler(signum, frame):
    raise DocumentTimeoutError(f"Document processing exceeded {PER_DOC_TIMEOUT_S}s")


# SIGALRM-based timeouts only work on the main thread on Unix -- fine here
# since this script is single-threaded, but don't lift process_libreoffice_group
# or process_onlyoffice_group into a worker thread without switching to a
# subprocess- or multiprocessing-based timeout instead.


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
            items.append(WorkItem(csv_path=csv_path, variant_idx=variant_idx, cfg=cfg))
    return items


def group_by_software_and_resolution(items: list[WorkItem]) -> dict[tuple[str, str], list[WorkItem]]:
    groups: dict[tuple[str, str], list[WorkItem]] = defaultdict(list)
    for item in items:
        groups[(item.cfg.software, item.cfg.resolution)].append(item)
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


def process_libreoffice_item(item: WorkItem, out_dir: Path, desktop, display: str, manifest_f) -> None:
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


def process_libreoffice_group(
    resolution: str, items: list[WorkItem], out_dir: Path, manifest_f
) -> None:
    print(f"-- LibreOffice, resolution {resolution}: {len(items)} screenshots --")
    with VirtualDisplay(resolution=resolution) as display:
        soffice_proc = start_soffice(display)
        try:
            desktop, _ctx = wait_for_soffice()
            docs_since_restart = 0
            signal.signal(signal.SIGALRM, _alarm_handler)

            for i, item in enumerate(items):
                if docs_since_restart >= RESTART_EVERY_N_DOCS:
                    print(f"  restarting soffice after {docs_since_restart} docs")
                    soffice_proc, desktop, _ctx = restart_soffice(soffice_proc, display)
                    docs_since_restart = 0

                print(f"  [{i+1}/{len(items)}] {item.csv_path.name} v{item.variant_idx}")
                try:
                    signal.alarm(PER_DOC_TIMEOUT_S)
                    process_libreoffice_item(item, out_dir, desktop, display, manifest_f)
                    signal.alarm(0)
                except DocumentTimeoutError as e:
                    # The specific failure mode this guards against: a blocked
                    # interactive dialog under Xvfb (near-zero CPU, no further
                    # log output). The stuck soffice process can't be reasoned
                    # with -- kill and restart it, then move on to the next doc.
                    print(f"    TIMEOUT: {e} -- restarting soffice")
                    signal.alarm(0)
                    soffice_proc, desktop, _ctx = restart_soffice(soffice_proc, display)
                    docs_since_restart = 0
                    continue
                except Exception as e:  # noqa: BLE001
                    # Log and continue -- one malformed document shouldn't
                    # kill a multi-hour shard.
                    signal.alarm(0)
                    print(f"    FAILED: {e}")
                docs_since_restart += 1
        finally:
            signal.alarm(0)
            soffice_proc.terminate()
            try:
                soffice_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                soffice_proc.kill()


def process_onlyoffice_item(item: WorkItem, out_dir: Path, display: str, manifest_f) -> None:
    """No long-lived session to reuse here -- each document gets its own
    OnlyOffice process, launched fresh and killed after capture. See
    render_onlyoffice.py's module docstring for why."""
    proc = launch_onlyoffice(display, str(item.csv_path.resolve()))
    try:
        window_id = find_onlyoffice_window(display)
        apply_config_onlyoffice(display, window_id, item.cfg)
        time.sleep(0.2)  # let the UI settle before capture

        stem = item.csv_path.stem
        raw_path = out_dir / f"{stem}_v{item.variant_idx}_raw.png"
        final_path = out_dir / f"{stem}_v{item.variant_idx}.png"
        # WINDOW_TITLE_HINT is reused here too, so capture.py finds the same
        # window it just configured rather than searching by a second,
        # possibly-inconsistent name.
        capture_screenshot(item.cfg, display, WINDOW_TITLE_HINT, raw_path, final_path)
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
        close_onlyoffice(proc)


def process_onlyoffice_group(
    resolution: str, items: list[WorkItem], out_dir: Path, manifest_f
) -> None:
    print(f"-- OnlyOffice, resolution {resolution}: {len(items)} screenshots --")
    with VirtualDisplay(resolution=resolution) as display:
        signal.signal(signal.SIGALRM, _alarm_handler)
        for i, item in enumerate(items):
            print(f"  [{i+1}/{len(items)}] {item.csv_path.name} v{item.variant_idx}")
            try:
                signal.alarm(PER_DOC_TIMEOUT_S)
                process_onlyoffice_item(item, out_dir, display, manifest_f)
                signal.alarm(0)
            except DocumentTimeoutError as e:
                # No shared session to restart here (each doc is its own
                # process already) -- just log and move to the next item.
                # If this fires often, it's a strong signal WINDOW_TITLE_HINT
                # or the keyboard-shortcut assumptions in render_onlyoffice.py
                # need correcting for your installed version, not that the
                # timeout itself is too short.
                print(f"    TIMEOUT: {e}")
                signal.alarm(0)
            except Exception as e:  # noqa: BLE001
                signal.alarm(0)
                print(f"    FAILED: {e}")


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
    groups = group_by_software_and_resolution(work_items)
    print(f"{len(work_items)} total screenshots across {len(groups)} (software, resolution) groups")

    manifest_path = args.out_dir / f"manifest_shard{args.shard_index}.jsonl"
    with open(manifest_path, "w") as manifest_f:
        for (software, resolution), items in groups.items():
            if software == "libreoffice_calc":
                process_libreoffice_group(resolution, items, args.out_dir, manifest_f)
            elif software == "excel":
                process_onlyoffice_group(resolution, items, args.out_dir, manifest_f)
            else:
                print(f"Unknown software '{software}', skipping {len(items)} items")


if __name__ == "__main__":
    main()
