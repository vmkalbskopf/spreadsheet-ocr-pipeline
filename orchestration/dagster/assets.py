"""
Asset graph for the pipeline, replacing the two Slurm .sbatch scripts.

screenshot_shard (partitioned, N_SHARDS partitions, runs concurrently)
    -> training_manifests (waits for ALL shard partitions, merges + splits)
        -> trained_model (single GPU run, QLoRA)
            -> awq_export (single GPU run, quantize for laptop)
            -> eval_results (single GPU run, TEDS scoring)

Concurrency for screenshot_shard partitions is controlled by Dagster's run
queue (see dagster.yaml / workspace config: max_concurrent_runs), not by
anything in this file -- set it based on how many LibreOffice+Xvfb
instances your workstation's CPU/RAM can comfortably run at once (each
instance is fairly light: ~1 core, a few hundred MB, so most workstations
can run 4-8 concurrently even while the GPU is idle waiting).

Retries: set `retry_policy` on screenshot_shard so a single flaky shard
(e.g. soffice failed to start in time) retries in isolation, instead of
requiring you to notice and manually rerun it -- this is the capability
that was missing without Slurm's array-job retry.
"""

from dagster import (
    AssetExecutionContext,
    ResourceParam,
    RetryPolicy,
    StaticPartitionsDefinition,
    asset,
)

from resources import DockerRunner

N_SHARDS = 8  # tune to your workstation's CPU core count / RAM

shard_partitions = StaticPartitionsDefinition([str(i) for i in range(N_SHARDS)])


@asset(
    partitions_def=shard_partitions,
    retry_policy=RetryPolicy(max_retries=2),
)
def screenshot_shard(context: AssetExecutionContext, docker_runner: ResourceParam[DockerRunner]) -> None:
    shard_index = int(context.partition_key)
    context.log.info(f"Generating screenshots for shard {shard_index}/{N_SHARDS}")
    docker_runner.run_screenshot_gen(shard_index=shard_index, n_shards=N_SHARDS)


@asset(deps=[screenshot_shard])
def training_manifests(context: AssetExecutionContext, docker_runner: ResourceParam[DockerRunner]) -> None:
    """Depends on ALL screenshot_shard partitions (Dagster resolves this
    automatically from the partitioned upstream asset -- this asset only
    materializes once every shard has completed)."""
    docker_runner.run_training(
        entrypoint_override=[
            "python3", "src/training/prepare_dataset.py",
            "--shard-manifests-glob", "/app/data/screenshots/manifest_shard*.jsonl",
            "--out-dir", "/app/data/manifests",
            "--val-frac", "0.05",
        ]
    )


@asset(deps=[training_manifests])
def trained_model(context: AssetExecutionContext, docker_runner: ResourceParam[DockerRunner]) -> None:
    docker_runner.run_training()  # default entrypoint: train_qlora.py


@asset(deps=[trained_model])
def awq_export(context: AssetExecutionContext, docker_runner: ResourceParam[DockerRunner]) -> None:
    docker_runner.run_training(
        entrypoint_override=[
            "python3", "src/training/export_awq.py",
            "--checkpoint", "/app/checkpoints/qwen25vl-3b-spreadsheet-ocr/final",
            "--calib-manifest", "/app/data/manifests/val.jsonl",
            "--out-dir", "/app/checkpoints/qwen25vl-3b-spreadsheet-ocr-awq",
        ]
    )


@asset(deps=[trained_model])
def eval_results(context: AssetExecutionContext, docker_runner: ResourceParam[DockerRunner]) -> None:
    docker_runner.run_training(
        entrypoint_override=[
            "python3", "src/eval/teds_eval.py",
            "--checkpoint", "/app/checkpoints/qwen25vl-3b-spreadsheet-ocr/final",
            "--val-manifest", "/app/data/manifests/val.jsonl",
        ]
    )
