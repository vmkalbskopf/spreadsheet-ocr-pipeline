"""
Thin wrapper around `docker run` so assets don't repeat volume-mount and
GPU-flag boilerplate. Deliberately just shells out to the docker CLI rather
than using the docker Python SDK -- simpler to debug on a single workstation,
and avoids a docker-in-docker complication if Dagster itself ever runs
containerized.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from dagster import ConfigurableResource


class DockerRunner(ConfigurableResource):
    project_root: str  # absolute path to spreadsheet-ocr-pipeline/ on the host

    def _base_mounts(self) -> list[str]:
        root = Path(self.project_root)
        return [
            "-v", f"{root / 'data'}:/app/data",
            "-v", f"{root / 'checkpoints'}:/app/checkpoints",
        ]

    def run_screenshot_gen(self, shard_index: int, n_shards: int) -> None:
        cmd = [
            "docker", "run", "--rm",
            *self._base_mounts(),
            "spreadsheet-ocr/screenshot-gen:latest",
            "--csv-dir", "/app/data/raw_csv",
            "--out-dir", "/app/data/screenshots",
            "--variation-config", "/app/config/screenshot_variation.yaml",
            "--shard-index", str(shard_index),
            "--n-shards", str(n_shards),
        ]
        subprocess.run(cmd, check=True)

    def run_training(self, entrypoint_override: list[str] | None = None) -> None:
        """entrypoint_override lets the same image run prepare_dataset.py,
        train_qlora.py, export_awq.py, or teds_eval.py -- they all live in
        the same container, only the invoked script differs."""
        cmd = [
            "docker", "run", "--rm", "--gpus", "all",
            *self._base_mounts(),
        ]
        if entrypoint_override:
            cmd += ["--entrypoint", entrypoint_override[0]]
            cmd += ["spreadsheet-ocr/training:latest", *entrypoint_override[1:]]
        else:
            cmd += ["spreadsheet-ocr/training:latest"]
        subprocess.run(cmd, check=True)
