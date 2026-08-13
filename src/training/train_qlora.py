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
    """
    Builds (input_ids, labels) with the FULL user+assistant turn encoded
    together, then masks prompt+image tokens in labels to -100 so loss is
    only computed on the target CSV tokens.

    This requires two processor passes per example:
      1. prompt-only (user turn, add_generation_prompt=True), UNPADDED,
         run per-example with that example's own image -- this gives the
         exact prompt token count *including* the image tokens Qwen2-VL
         interleaves into input_ids, which varies per image depending on
         the number of patches at the sampled resolution. There's no
         shortcut here: image token count isn't known without running the
         processor on that specific image.
      2. full text (user turn + assistant target), BATCHED with padding --
         this produces the actual training tensors.
    Given (1), we know exactly how many leading tokens in each row of (2)
    belong to the prompt, so we mask that prefix (plus any right-padding)
    to -100 and leave the target CSV tokens as the loss target.
    """
    max_len = cfg["data"]["max_target_tokens"]

    def _prompt_len(image, prompt_text: str) -> int:
        prompt_only = processor(
            text=[prompt_text], images=[image], padding=False, return_tensors="pt"
        )
        return prompt_only.input_ids.shape[1]

    def collate(batch: list[dict]):
        images = [Image.open(r["image_path"]).convert("RGB") for r in batch]
        prompts = [r["prompt"] for r in batch]
        targets = [r["target"] for r in batch]

        prompt_messages = [
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": p}]}]
            for p in prompts
        ]
        prompt_texts = [
            processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in prompt_messages
        ]

        full_messages = [
            [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": p}]},
                {"role": "assistant", "content": [{"type": "text", "text": t}]},
            ]
            for p, t in zip(prompts, targets)
        ]
        full_texts = [
            processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in full_messages
        ]

        # Pass 1: per-example prompt length (unpadded, includes image tokens)
        prompt_lens = [
            _prompt_len(img, ptext) for img, ptext in zip(images, prompt_texts)
        ]

        # Pass 2: batched, padded, full user+assistant sequence
        inputs = processor(
            text=full_texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )

        labels = inputs["input_ids"].clone()
        pad_token_id = processor.tokenizer.pad_token_id
        for i, plen in enumerate(prompt_lens):
            # Mask the prompt+image-token prefix
            labels[i, :plen] = -100
            # Mask right-padding (default HF padding side)
            labels[i][inputs["attention_mask"][i] == 0] = -100
        # Belt-and-braces: mask any literal pad tokens that slipped through
        labels[labels == pad_token_id] = -100

        # Surface truncation now rather than discovering it later as an
        # unexplained accuracy cliff on large tables -- this is exactly the
        # token-explosion failure mode being deferred to post-v1, so make it
        # visible in training logs instead of silent.
        n_fully_masked = int((labels == -100).all(dim=1).sum())
        if n_fully_masked:
            print(
                f"[collate] WARNING: {n_fully_masked}/{len(batch)} examples in this "
                f"batch were truncated so severely their labels are entirely masked "
                f"(zero loss contribution). Likely large tables exceeding "
                f"max_target_tokens={max_len}. Track this rate -- rising over training "
                f"is the signal to prioritize the tiling/windowing follow-up."
            )

        inputs["labels"] = labels
        return inputs

    return collate

# NOTE: for very large tables, `full_texts` can exceed max_target_tokens
# before the assistant's target CSV even starts, leaving that row's labels
# entirely -100 (contributes zero loss, silently). This is the same
# underlying issue as the token-explosion/tiling discussion for inference --
# raising max_target_tokens helps short-term, but a tiling/windowed
# architecture is the real fix for large sheets. Worth logging a warning
# here (count of all-masked rows per batch) once you're running real
# training, to catch it happening silently.


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
