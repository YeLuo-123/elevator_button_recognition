#!/usr/bin/env python3
"""Generate strict state labels and report metrics for a Qwen3-VL adapter."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from train_qwen_vl_button_state import SYSTEM_PROMPT, USER_PROMPT


ROOT = Path(__file__).resolve().parents[1]
STATES = ("ON", "OFF", "UNCERTAIN")


def parse_state(text: str) -> str:
    match = re.search(
        r'\{\s*"state"\s*:\s*"(ON|OFF|UNCERTAIN)"\s*\}',
        text,
        flags=re.IGNORECASE,
    )
    return match.group(1).upper() if match else "INVALID"


def metrics(rows: list[dict]) -> dict:
    confusion = {actual: Counter() for actual in STATES}
    for row in rows:
        confusion[row["actual"]][row["predicted"]] += 1
    per_class = {}
    for state in STATES:
        support = sum(confusion[state].values())
        predicted = sum(confusion[actual][state] for actual in STATES)
        true_positive = confusion[state][state]
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else None
        per_class[state] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if recall is not None and precision + recall else 0.0,
        }
    valid = [row for row in rows if row["actual"] in ("ON", "OFF")]
    correct = sum(row["actual"] == row["predicted"] for row in valid)
    all_correct = sum(row["actual"] == row["predicted"] for row in rows)
    class_f1 = [per_class[state]["f1"] for state in STATES if per_class[state]["support"]]
    return {
        "accuracy": all_correct / len(rows) if rows else None,
        "binary_accuracy": correct / len(valid) if valid else None,
        "macro_f1": sum(class_f1) / len(class_f1) if class_f1 else None,
        "invalid_rate": sum(row["predicted"] == "INVALID" for row in rows) / len(rows),
        "per_class": per_class,
        "confusion": {state: dict(confusion[state]) for state in STATES},
        "mean_latency_ms": sum(row["latency_ms"] for row in rows) / len(rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--data", type=Path, default=ROOT / "datasets/button_state/vlm_v1")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/button_state/evaluation.json")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data.expanduser().resolve()
    rows = json.loads((data_root / f"{args.split}.json").read_text(encoding="utf-8"))
    if args.limit:
        rows = rows[: args.limit]
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    processor_path = str(args.adapter) if args.adapter else args.model
    processor = AutoProcessor.from_pretrained(processor_path)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        quantization_config=quantization,
        device_map={"": 0},
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    predictions = []
    for index, row in enumerate(rows, start=1):
        actual = json.loads(row["conversations"][-1]["value"])["state"]
        with Image.open(data_root / row["image"]) as opened:
            image = opened.convert("RGB")
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": USER_PROMPT},
            ]},
        ]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=16, do_sample=False)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - started) * 1000
        new_tokens = generated[:, inputs["input_ids"].shape[1]:]
        raw = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
        predictions.append({
            "image": row["image"], "actual": actual,
            "predicted": parse_state(raw), "raw": raw, "latency_ms": elapsed,
        })
        if index % 20 == 0 or index == len(rows):
            print(f"Evaluated {index}/{len(rows)}")
    result = {"summary": metrics(predictions), "predictions": predictions}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
