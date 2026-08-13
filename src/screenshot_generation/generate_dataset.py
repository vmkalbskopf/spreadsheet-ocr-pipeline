"""
Orchestrates screenshot generation across all source CSVs.

For each source CSV, generates `n_variants_per_csv` screenshots, each with
an independently sampled ScreenshotConfig. Manages the soffice --accept
process lifecycle (start once per worker, reuse across many documents --
starting a fresh soffice process per screenshot is far too slow at scale).

Designed to run as one shard of a Slurm array job (see
slurm/generate_screenshots.sbatch) -- pass --shard-index/--n-shards to
process a disjoint slice of the CSV list, so the array job is fully
parallel with no coordination needed between workers.

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
from pathlib import Path

from capture import VirtualDisplay, capture_screenshot
from render_libreoffice import apply_config, close_doc, connect_to_soffice, load_csv
from variation_sampler import load_variation_config, sample_config

SOFFICE_PORT = 2002
SOFFICE_STARTUP_TIMEOUT_S = 30


def start_soffice(display: str, port: int = SOFFICE_PORT) -> subprocess.Popen:
    proc = subprocess.Popen(
        [
            "soffice",
            "--accept=socket,host=localhost,port={};urp;".format(port),
            "--norestore",
            "--nologo",
        ],
        env={"DISPLAY": display},
    )
    return proc


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


def process_csv(
    csv_path: Path,
    out_dir: Path,
    desktop,
    display: str,
    variation_cfg: dict,
    n_variants: int,
    manifest_f,
) -> None:
    for variant_idx in range(n_variants):
        seed = hash((str(csv_path), variant_idx)) & 0xFFFFFFFF
        cfg = sample_config(variation_cfg, seed=seed)

        if cfg.software != "libreoffice_calc":
            # Excel rendering path (COM automation on a Windows box/VM) is
            # a separate, not-yet-implemented module -- see README follow-up.
            # Re-sample as LibreOffice for now so the pipeline stays runnable.
            cfg.software = "libreoffice_calc"

        doc = load_csv(desktop, str(csv_path.resolve()))
        try:
            apply_config(doc, cfg)
            time.sleep(0.2)  # let the UI settle before capture

            stem = csv_path.stem
            raw_path = out_dir / f"{stem}_v{variant_idx}_raw.png"
            final_path = out_dir / f"{stem}_v{variant_idx}.png"
            capture_screenshot(cfg, display, "LibreOffice Calc", raw_path, final_path)
            raw_path.unlink(missing_ok=True)

            manifest_f.write(
                json.dumps(
                    {
                        "csv_path": str(csv_path),
                        "screenshot_path": str(final_path),
                        "variant_index": variant_idx,
                        "config": cfg.to_dict(),
                    }
                )
                + "\n"
            )
        finally:
            close_doc(doc)


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

    manifest_path = args.out_dir / f"manifest_shard{args.shard_index}.jsonl"

    with VirtualDisplay(resolution="1920x1080") as display:
        soffice_proc = start_soffice(display)
        try:
            desktop, _ctx = wait_for_soffice()
            with open(manifest_path, "w") as manifest_f:
                for i, csv_path in enumerate(shard_csvs):
                    print(f"[{i+1}/{len(shard_csvs)}] {csv_path.name}")
                    try:
                        process_csv(
                            csv_path, args.out_dir, desktop, display,
                            variation_cfg, n_variants, manifest_f,
                        )
                    except Exception as e:  # noqa: BLE001
                        # Log and continue -- one malformed CSV shouldn't kill
                        # a multi-hour array job shard.
                        print(f"  FAILED: {e}")
        finally:
            soffice_proc.terminate()
            soffice_proc.wait(timeout=10)


if __name__ == "__main__":
    main()
