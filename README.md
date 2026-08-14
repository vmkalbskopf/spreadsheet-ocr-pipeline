# Spreadsheet Screenshot OCR Pipeline

Trains a vision-language model (Qwen2.5-VL-3B) to reconstruct structured CSV
data from realistic screenshots of spreadsheet software (LibreOffice Calc,
Excel). Designed for a single 40GB-VRAM training workstation, with inference
targeting a laptop with a small discrete GPU (4-6GB VRAM, AWQ 4-bit).

## Pipeline stages

```
1. data_collection/     Scrape + dedup + normalize real public spreadsheets to CSV
2. screenshot_generation/  Render each CSV in Calc/Excel with randomized UI state,
                            capture real screenshots (not synthetic mockups)
3. training/             QLoRA fine-tune Qwen2.5-VL-3B on (screenshot, csv) pairs,
                            then export to AWQ 4-bit for laptop inference
4. eval/                 TEDS-based structural accuracy, not raw string match
```

## Orchestration (Dagster + Docker, not Slurm)

Two Docker images, no shared host processes:

- `docker/Dockerfile.screenshot-gen` — CPU-only: LibreOffice + Xvfb + UNO.
  Each container instance runs one shard in isolation, which means the
  hardcoded Xvfb display (`:99`) and soffice port (`2002`) never collide
  across concurrently-running shards — isolation is free from
  containerization, no port/display parameterization needed.
- `docker/Dockerfile.training` — GPU: PyTorch + QLoRA/AWQ stack. Runs
  `prepare_dataset.py`, `train_qlora.py`, `export_awq.py`, and
  `teds_eval.py` via entrypoint overrides (same image, different script).

Build both:
```bash
docker build -f docker/Dockerfile.screenshot-gen -t spreadsheet-ocr/screenshot-gen:latest .
docker build -f docker/Dockerfile.training -t spreadsheet-ocr/training:latest .
```

Dagster (`orchestration/dagster/`) replaces the two `.sbatch` scripts:
- `screenshot_shard` is a partitioned asset (one partition per shard),
  each partition launching its own `screenshot-gen` container. Dagster's
  run queue controls concurrency (set `max_concurrent_runs` based on your
  workstation's cores — each shard is light, ~1 core / a few hundred MB).
- Per-partition retries (`RetryPolicy`) replace what Slurm's array-job
  retry gave you for free — without this, a flaky shard silently needs a
  manual rerun.
- `training_manifests` depends on *all* shard partitions completing, then
  the rest of the graph (`trained_model` → `awq_export` / `eval_results`)
  runs as single GPU container invocations.

```bash
cd orchestration/dagster
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
mkdir -p .dagster_home

PROJECT_ROOT=/absolute/path/to/spreadsheet-ocr-pipeline \
DAGSTER_HOME=$(pwd)/.dagster_home dagster dev -f definitions.py
```
The `-f definitions.py` tells Dagster where the `Definitions` object lives —
newer Dagster versions no longer infer this automatically and error with
`No arguments given and no [tool.dagster] block in pyproject.toml found`
without it. `dagster asset materialize` needs the same flag:
```bash
dagster asset materialize -f definitions.py --select "*"
```

**Why local, not containerized:** `DockerRunner` (`resources.py`) works by shelling out to `docker run` on the host. If Dagster itself ran inside a container, it would need either the host's Docker socket mounted in (`-v /var/run/docker.sock:/var/run/docker.sock`) to launch sibling containers, or a full docker-in-docker setup — both add real complexity (the socket-mount approach in particular gives that container root-equivalent access to the host) for no benefit on a single workstation. Dagster only needs the `dagster` Python package and the `docker` CLI on the host; it never needs LibreOffice, CUDA, or any of the heavier dependencies those live inside the two images.

## Known limitations / deferred decisions

- **Large-table token explosion**: end-to-end auto-regressive CSV generation
  means a ~100x15 table is 3,000-6,000+ tokens, which is slow to generate
  on a laptop GPU and prone to cascading errors from a single dropped
  delimiter. Deferred intentionally: `src/eval/teds_eval.py` now reports
  accuracy broken down by row-count bucket, and `train_qlora.py` logs a
  warning when truncation fully masks an example's labels. Use those
  signals to decide whether a windowed-tiling or 2-stage
  detector-then-VLM architecture is actually justified, rather than
  building it preemptively.
- **AWQ calibration is text-only** (`src/training/export_awq.py`): doesn't
  exercise the vision tower during calibration, which is a real quality
  gap for a model whose job is reading images. `llm-compressor` has more
  consistent multimodal calibration support if this proves to matter.
- **Excel rendering path is unimplemented** -- screenshots currently fall
  back to LibreOffice Calc even when Excel is sampled in
  `screenshot_variation.yaml`. Needs Windows/COM automation (`xlwings`)
  or a VM, which is a meaningfully different automation stack from the
  Linux/UNO path already built.
- **GitHub scraping requires `GITHUB_TOKEN`**; Kaggle scraping requires
  `~/.kaggle/kaggle.json` or `KAGGLE_USERNAME`/`KAGGLE_KEY`. Neither
  scraper has been run against live APIs in this environment (network
  access here is restricted to package registries) -- the normalize and
  dedup/binning stages *have* been tested end-to-end against synthetic
  staged files, including exact-dup, near-dup, and license-gate cases.

## Directory layout

```
config/
  data_sources.yaml         # source APIs, license allowlist, row/col binning targets
  screenshot_variation.yaml # ranges for all randomized rendering parameters
  train_qlora.yaml          # LoRA rank, quantization, image resolution, hyperparams

src/
  data_collection/
    common.py                # shared: manifests, hashing, license gate, near-dup signatures
    scrape_kaggle.py         # Kaggle API, license-filtered at search time
    scrape_gov_data.py       # data.gov (CKAN) + data.norge.no (DCAT) + Eurostat (SDMX)
    scrape_github.py         # GitHub code search for standalone .csv files
    normalize_to_csv.py      # xlsx/xls/ods/tsv/json -> canonical CSV, computes row/col counts
    dedup_and_bin.py         # license gate + exact/near dedup + shape-bin balancing -> data/raw_csv/

  screenshot_generation/
    variation_sampler.py    # samples one random UI configuration per screenshot
    render_libreoffice.py   # UNO automation: load CSV, apply config, save state
    capture.py              # Xvfb + window capture -> PNG
    generate_dataset.py     # orchestrates sampler + render + capture per CSV

  training/
    prepare_dataset.py      # builds HF Dataset of (image, target_csv_text) pairs
    train_qlora.py          # QLoRA fine-tune entrypoint
    export_awq.py           # post-training AWQ quantization for laptop inference

  eval/
    teds_eval.py            # tree-edit-distance table similarity scoring

data/
  raw_csv/                  # canonical ground-truth CSVs, binned by size
  screenshots/              # rendered screenshots, mirrors raw_csv structure
  manifests/                # JSONL manifests linking screenshot <-> csv <-> config

docker/
  Dockerfile.screenshot-gen    # CPU image: LibreOffice + Xvfb + UNO
  Dockerfile.training          # GPU image: PyTorch + QLoRA/AWQ stack
  requirements-*.txt

orchestration/dagster/
  assets.py                    # partitioned screenshot_shard -> training graph
  resources.py                 # DockerRunner: wraps `docker run` per asset
  definitions.py               # Dagster entrypoint
```

## Suggested run order

```bash
# 1. Data collection (writes to data/staging/, then data/raw_csv/)
python src/data_collection/scrape_kaggle.py --config config/data_sources.yaml
python src/data_collection/scrape_gov_data.py --config config/data_sources.yaml
GITHUB_TOKEN=ghp_xxx python src/data_collection/scrape_github.py --config config/data_sources.yaml
python src/data_collection/normalize_to_csv.py --config config/data_sources.yaml
python src/data_collection/dedup_and_bin.py --config config/data_sources.yaml
# install: pip install -r src/data_collection/requirements.txt

# 2. Build containers
docker build -f docker/Dockerfile.screenshot-gen -t spreadsheet-ocr/screenshot-gen:latest .
docker build -f docker/Dockerfile.training -t spreadsheet-ocr/training:latest .

# 3. Run everything else via Dagster (install: orchestration/dagster/requirements.txt)
cd orchestration/dagster && mkdir -p .dagster_home
PROJECT_ROOT=$(cd ../.. && pwd) DAGSTER_HOME=$(pwd)/.dagster_home dagster dev -f definitions.py
# then materialize assets from the UI, or:
dagster asset materialize -f definitions.py --select "*"
```
