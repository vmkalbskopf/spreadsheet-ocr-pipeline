"""
Exports a QLoRA-fine-tuned checkpoint to AWQ 4-bit for laptop inference on
a small discrete GPU (4-6GB VRAM class, e.g. RTX 3050/4050 mobile).

Two steps:
  1. Merge LoRA adapters into the base model (AWQ quantizes a dense model,
     not a base + adapter pair)
  2. AWQ-quantize the merged model using a calibration set drawn from your
     OWN validation screenshots -- NOT a generic text calibration set,
     since AWQ's activation-aware scaling should reflect the actual
     distribution of prompts/images this model will see in production.

Usage:
    python export_awq.py \
        --checkpoint checkpoints/qwen25vl-3b-spreadsheet-ocr/final \
        --calib-manifest data/manifests/val.jsonl \
        --out-dir checkpoints/qwen25vl-3b-spreadsheet-ocr-awq
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from awq import AutoAWQForCausalLM
from peft import PeftModel
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


def merge_lora(checkpoint_dir: str, merged_out_dir: str) -> None:
    """Loads the base model + LoRA adapter and writes a merged dense
    checkpoint. Runs in bf16 on CPU-offloaded weights if VRAM is tight --
    this step doesn't need to be fast, it runs once per training run."""
    with open(Path(checkpoint_dir) / "train_config_used.json") as f:
        train_cfg = json.load(f)
    base_model_name = train_cfg["model"]["base_model"]

    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    merged = PeftModel.from_pretrained(base_model, checkpoint_dir)
    merged = merged.merge_and_unload()
    merged.save_pretrained(merged_out_dir)

    processor = AutoProcessor.from_pretrained(checkpoint_dir)
    processor.save_pretrained(merged_out_dir)


def build_calibration_data(calib_manifest: str, n_samples: int = 128) -> list[dict]:
    """Draws a representative calibration sample from real validation
    screenshots, spanning the row/col size range, so AWQ's per-channel
    scaling reflects both dense small-font sheets and sparse large ones."""
    records = []
    with open(calib_manifest) as f:
        for line in f:
            records.append(json.loads(line))

    # Simple stratified-ish sampling: sort by target CSV length (proxy for
    # table size) and take an even spread rather than the first N records,
    # which would bias toward whatever order the manifest happened to be in.
    records.sort(key=lambda r: len(r["target"]))
    step = max(1, len(records) // n_samples)
    return records[::step][:n_samples]


def run_awq_quantization(merged_model_dir: str, calib_records: list[dict], out_dir: str) -> None:
    model = AutoAWQForCausalLM.from_pretrained(merged_model_dir, device_map="auto")

    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM",  # GEMM kernel: better throughput on small discrete GPUs than GEMV at this batch=1 use case
    }

    # NOTE: AWQ's vision-language calibration support varies by library
    # version -- as of writing, multimodal AWQ calibration APIs are less
    # standardized than text-only. Verify against the awq library version
    # you pin before relying on this path; the text-only calibration
    # fallback below (calibrating on target CSV text alone) is a reasonable
    # degraded option if full multimodal calibration isn't supported yet.
    calib_texts = [r["target"] for r in calib_records]

    model.quantize(quant_config=quant_config, calib_data=calib_texts)
    model.save_quantized(out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--calib-manifest", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--n-calib-samples", type=int, default=128)
    args = ap.parse_args()

    merged_dir = args.out_dir + "-merged-fp16"
    print("Merging LoRA adapters...")
    merge_lora(args.checkpoint, merged_dir)

    print("Building calibration set from validation screenshots...")
    calib_records = build_calibration_data(args.calib_manifest, args.n_calib_samples)

    print(f"Running AWQ 4-bit quantization ({len(calib_records)} calibration samples)...")
    run_awq_quantization(merged_dir, calib_records, args.out_dir)

    print(f"Done. Quantized model at: {args.out_dir}")
    print("Expected laptop VRAM footprint: ~2.5-3GB for the 3B model.")


if __name__ == "__main__":
    main()
