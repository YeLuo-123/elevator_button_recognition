#!/usr/bin/env python3
"""QLoRA fine-tuning of Qwen3-VL for button ON/OFF/UNCERTAIN output."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


ROOT = Path(__file__).resolve().parents[1]
SYSTEM_PROMPT = (
    "你是电梯按钮亮暗状态分类器。只判断图中按钮是否主动发光。"
    "金属反光、白色表面和环境照明不算点亮。无法可靠判断时选择UNCERTAIN。"
    "只输出严格JSON，不要解释。"
)
USER_PROMPT = (
    "判断这个电梯按钮的状态。只允许输出"
    '{"state":"ON"}、{"state":"OFF"}或{"state":"UNCERTAIN"}。'
)


class StateDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], root: Path) -> None:
        self.rows = rows
        self.root = root

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        answer = row["conversations"][-1]["value"]
        return {"image": self.root / row["image"], "answer": answer}


class VLMCollator:
    def __init__(self, processor: Any) -> None:
        self.processor = processor

    def _encode(self, item: dict[str, Any]) -> dict[str, torch.Tensor]:
        with Image.open(item["image"]) as opened:
            image = opened.convert("RGB")
        user_messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": USER_PROMPT},
            ]},
        ]
        full_messages = user_messages + [
            {"role": "assistant", "content": [{"type": "text", "text": item["answer"]}]}
        ]
        prefix = self.processor.apply_chat_template(
            user_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        encoded = self.processor.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )
        labels = encoded["input_ids"].clone()
        labels[:, : prefix["input_ids"].shape[1]] = -100
        encoded["labels"] = labels
        return {
            key: value.squeeze(0) if key in ("input_ids", "attention_mask", "labels", "token_type_ids") else value
            for key, value in encoded.items() if torch.is_tensor(value)
        }

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        encoded = [self._encode(item) for item in items]
        pad_id = self.processor.tokenizer.pad_token_id
        max_length = max(item["input_ids"].shape[0] for item in encoded)
        batch: dict[str, list[torch.Tensor]] = {}
        for item in encoded:
            for key, value in item.items():
                batch.setdefault(key, []).append(value)
        result: dict[str, torch.Tensor] = {}
        for key, values in batch.items():
            if key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
                result[key] = torch.cat(values, dim=0)
            elif values[0].ndim == 1:
                fill = -100 if key == "labels" else (pad_id if key == "input_ids" else 0)
                padded = [
                    torch.nn.functional.pad(value, (0, max_length - value.shape[0]), value=fill)
                    for value in values
                ]
                result[key] = torch.stack(padded)
            else:
                result[key] = torch.stack(values)
        return result


def oversample(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    by_state: dict[str, list[dict[str, Any]]] = {state: [] for state in ("ON", "OFF", "UNCERTAIN")}
    for row in rows:
        answer = json.loads(row["conversations"][-1]["value"])
        by_state[answer["state"]].append(row)
    largest = max(len(values) for values in by_state.values())
    targets = {"OFF": largest, "ON": largest // 2, "UNCERTAIN": min(largest // 4, len(by_state["UNCERTAIN"]) * 5)}
    rng = random.Random(seed)
    balanced: list[dict[str, Any]] = []
    for state, values in by_state.items():
        if not values:
            continue
        balanced.extend(values)
        balanced.extend(rng.choices(values, k=max(0, targets[state] - len(values))))
    rng.shuffle(balanced)
    print("Original:", {state: len(values) for state, values in by_state.items()})
    print("After oversampling:", Counter(json.loads(row["conversations"][-1]["value"])["state"] for row in balanced))
    return balanced


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--data", type=Path, default=ROOT / "datasets/button_state/vlm_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/button_state/qwen3_vl_2b_lora_v1")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--dataloader-workers", type=int, default=2)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=-1, help="positive value for a smoke run")
    parser.add_argument("--resume-from-checkpoint", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.output / "final_adapter").exists():
        raise FileExistsError(f"Completed adapter already exists in {args.output}")
    set_seed(args.seed)
    data_root = args.data.expanduser().resolve()
    train_rows = json.loads((data_root / "train.json").read_text(encoding="utf-8"))
    val_rows = json.loads((data_root / "val.json").read_text(encoding="utf-8"))
    train_rows = oversample(train_rows, args.seed)

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        quantization_config=quantization,
        device_map={"": 0},
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    )
    use_gradient_checkpointing = not args.no_gradient_checkpointing
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=use_gradient_checkpointing
    )
    model = get_peft_model(model, LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    # Keep the visual encoder fixed; language-side LoRA learns the state policy.
    for name, parameter in model.named_parameters():
        if ".visual." in name or name.startswith("base_model.model.visual"):
            parameter.requires_grad_(False)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0,
        fp16=True,
        gradient_checkpointing=use_gradient_checkpointing,
        eval_strategy="epoch" if args.max_steps < 0 else "no",
        save_strategy="epoch" if args.max_steps < 0 else "steps",
        save_steps=1 if args.max_steps > 0 else 500,
        logging_steps=5,
        load_best_model_at_end=args.max_steps < 0,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_workers,
        dataloader_persistent_workers=args.dataloader_workers > 0,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=StateDataset(train_rows, data_root),
        eval_dataset=StateDataset(val_rows, data_root),
        data_collator=VLMCollator(processor),
    )
    result = trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
    trainer.save_metrics("train", result.metrics)
    trainer.save_state()
    trainer.save_model(args.output / "final_adapter")
    processor.save_pretrained(args.output / "final_adapter")


if __name__ == "__main__":
    main()
