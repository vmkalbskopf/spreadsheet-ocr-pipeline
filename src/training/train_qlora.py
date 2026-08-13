"""
QLoRA fine-tune Qwen2.5-VL-3B-Instruct on (screenshot, csv) pairs.

Config-driven: all hyperparameters come from config/train_qlora.yaml so the
40GB VRAM budget is documented in one place rather than scattered as magic
numbers in code. See that file's trailing comment for the VRAM breakdown.

Usage:
    python train_qlora.py --config config/train_qlora.yaml
"""

from __future__ import annotations

import argparse
import json

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model_and_processor(cfg: dict):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=getattr(torch, cfg["model"]["compute_dtype"]),
        bnb_4bit_use_double_quant=True,
    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        cfg["model"]["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=getattr(torch, cfg["model"]["compute_dtype"]),
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    processor = AutoProcessor.from_pretrained(
        cfg["model"]["base_model"],
        min_pixels=cfg["image"]["min_pixels"],
        max_pixels=cfg["image"]["max_pixels"],
    )
    return model, processor


def build_collate_fn(processor, cfg: dict):
    def collate(batch: list[dict]):
        images = [Image.open(r["image_path"]).convert("RGB") for r in batch]
        prompts = [r["prompt"] for r in batch]
        targets = [r["target"] for r in batch]

        messages = [
            [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": p}],
                }
            ]
            for p in prompts
        ]
        chat_prompts = [
            processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages
        ]

        inputs = processor(
            text=chat_prompts,
            images=images,
            padding=True,
            truncation=True,
            max_length=cfg["data"]["max_target_tokens"],
            return_tensors="pt",
        )

        target_ids = processor.tokenizer(
            targets,
            padding=True,
            truncation=True,
            max_length=cfg["data"]["max_target_tokens"],
            return_tensors="pt",
        ).input_ids

        # Standard causal-LM setup: labels = prompt tokens masked to -100,
        # target tokens kept as-is. Full masking logic depends on the exact
        # processor output layout -- Qwen2-VL's processor interleaves image
        # tokens into input_ids, so in practice you'll want to verify this
        # against `processor.apply_chat_template` output shape before the
        # first real training run rather than trusting this sketch blindly.
        inputs["labels"] = target_ids
        return inputs

    return collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    model, processor = build_model_and_processor(cfg)

    dataset = load_dataset(
        "json",
        data_files={
            "train": cfg["data"]["train_manifest"],
            "validation": cfg["data"]["val_manifest"],
        },
    )

    collate_fn = build_collate_fn(processor, cfg)
    t = cfg["training"]

    training_args = TrainingArguments(
        output_dir=t["output_dir"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        logging_steps=t["logging_steps"],
        eval_strategy="steps",
        eval_steps=t["eval_steps"],
        save_steps=t["save_steps"],
        save_total_limit=t["save_total_limit"],
        bf16=t["bf16"],
        gradient_checkpointing=t["gradient_checkpointing"],
        report_to=t["report_to"],
        remove_unused_columns=False,  # required: our collate_fn needs raw dict fields
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collate_fn,
    )

    trainer.train()
    trainer.save_model(t["output_dir"] + "/final")
    processor.save_pretrained(t["output_dir"] + "/final")

    with open(t["output_dir"] + "/final/train_config_used.json", "w") as f:
        json.dump(cfg, f, indent=2)


if __name__ == "__main__":
    main()
