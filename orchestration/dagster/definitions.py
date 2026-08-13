"""
Entrypoint for `dagster dev` / `dagster asset materialize`.

Usage:
    cd orchestration/dagster
    DAGSTER_HOME=$(pwd)/.dagster_home dagster dev

Set PROJECT_ROOT to the absolute path of spreadsheet-ocr-pipeline/ on your
workstation before running -- the DockerRunner resource uses it to bind-mount
data/ and checkpoints/ into each container.
"""

from __future__ import annotations

import os

from dagster import Definitions

from assets import awq_export, eval_results, screenshot_shard, trained_model, training_manifests
from resources import DockerRunner

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.abspath("../.."))

defs = Definitions(
    assets=[screenshot_shard, training_manifests, trained_model, awq_export, eval_results],
    resources={
        "docker_runner": DockerRunner(project_root=PROJECT_ROOT),
    },
)
