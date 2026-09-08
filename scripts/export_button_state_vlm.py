#!/usr/bin/env python3
"""Freeze state annotations and export Qwen-VL crop/SFT datasets.

Samples derived from the same Roboflow source image are always assigned to the
same split.  This prevents augmented variants and multiple buttons in one panel
from leaking between train, validation, and test sets.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.annotate_button_state import Candidate, load_candidates  # noqa: E402


DEFAULT_ANNOTATIONS = ROOT / "datasets" / "button_state" / "annotations.json"
DEFAULT_DATA = ROOT / "configs" / "dataset.yaml"
DEFAULT_OUTPUT = ROOT / "datasets" / "button_state" / "vlm_v1"
VALID_STATES = {"ON", "OFF", "UNCERTAIN"}
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def choose_group_splits(records: list[dict], seed: int) -> dict[str, str]:
    by_group: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_group[record["source_group"]].append(record)
    groups = sorted(by_group)
    if len(groups) < 3:
        raise ValueError("At least three source groups are required")

    totals = Counter(record["state"] for record in records)
    target_groups = {
        "train": round(len(groups) * RATIOS["train"]),
        "val": round(len(groups) * RATIOS["val"]),
    }
    target_groups["test"] = len(groups) - target_groups["train"] - target_groups["val"]
    best_score = float("inf")
    best: dict[str, str] | None = None

    # Search deterministic group-level shuffles for a split with representative
    # ON/OFF counts. UNCERTAIN has too few source groups to require all splits.
    for trial in range(4000):
        shuffled = groups.copy()
        random.Random(seed + trial).shuffle(shuffled)
        assignment: dict[str, str] = {}
        cursor = 0
        for split in ("train", "val", "test"):
            end = cursor + target_groups[split]
            for group in shuffled[cursor:end]:
                assignment[group] = split
            cursor = end
        counts = {split: Counter() for split in RATIOS}
        for group, rows in by_group.items():
            counts[assignment[group]].update(row["state"] for row in rows)
        if any(counts[split][state] == 0 for split in RATIOS for state in ("ON", "OFF")):
            continue
        if totals["UNCERTAIN"] and counts["train"]["UNCERTAIN"] == 0:
            continue
        score = 0.0
        for split, ratio in RATIOS.items():
            for state in ("ON", "OFF", "UNCERTAIN"):
                if totals[state] == 0:
                    continue
                expected = totals[state] * ratio
                score += abs(counts[split][state] - expected) / max(1.0, expected)
        if score < best_score:
            best_score, best = score, assignment
    if best is None:
        raise RuntimeError("Could not produce a group split containing ON and OFF in every partition")
    return best


def crop_candidate(candidate: Candidate, destination: Path, padding_ratio: float) -> None:
    with Image.open(candidate.image_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    x, y, width, height = candidate.bbox_xywhn
    x1, y1 = (x - width / 2) * image.width, (y - height / 2) * image.height
    x2, y2 = (x + width / 2) * image.width, (y + height / 2) * image.height
    padding = max(x2 - x1, y2 - y1) * padding_ratio
    box = (
        max(0, round(x1 - padding)),
        max(0, round(y1 - padding)),
        min(image.width, round(x2 + padding)),
        min(image.height, round(y2 + padding)),
    )
    crop = image.crop(box)
    destination.parent.mkdir(parents=True, exist_ok=True)
    crop.save(destination, "JPEG", quality=95, subsampling=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export button states for Qwen-VL SFT")
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--padding", type=float, default=0.25)
    parser.add_argument("--replace", action="store_true", help="replace an existing export")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        if not args.replace:
            raise FileExistsError(f"Output exists: {output}. Use --replace to rebuild it.")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    annotation_path = args.annotations.expanduser().resolve()
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    annotations = payload.get("annotations", {})
    candidates = {candidate.key: candidate for candidate in load_candidates(args.data)}
    records: list[dict] = []
    for key, annotation in annotations.items():
        state = str(annotation.get("state", ""))
        candidate = candidates.get(key)
        if state not in VALID_STATES or candidate is None:
            continue
        records.append({
            "key": key,
            "state": state,
            "source_group": candidate.source_group,
            "candidate": candidate,
        })
    if not records:
        raise RuntimeError("No ON/OFF/UNCERTAIN annotations found")

    assignment = choose_group_splits(records, args.seed)
    exported: dict[str, list[dict]] = {split: [] for split in RATIOS}
    manifest_rows: list[dict] = []
    for index, record in enumerate(records):
        candidate: Candidate = record["candidate"]
        split = assignment[candidate.source_group]
        filename = f"{index:06d}_{candidate.source_group}_{candidate.line_index}.jpg"
        relative_image = Path("images") / split / filename
        crop_candidate(candidate, output / relative_image, args.padding)
        answer = json.dumps({"state": record["state"]}, ensure_ascii=False, separators=(",", ":"))
        exported[split].append({
            "image": str(relative_image),
            "conversations": [
                {
                    "from": "human",
                    "value": (
                        "<image>\n判断图中电梯按钮是否主动发光。"
                        "金属反光和白色表面不算点亮；无法可靠判断时选UNCERTAIN。"
                        "只输出JSON：{\"state\":\"ON|OFF|UNCERTAIN\"}"
                    ),
                },
                {"from": "gpt", "value": answer},
            ],
        })
        manifest_rows.append({
            "key": record["key"],
            "split": split,
            "state": record["state"],
            "source_group": candidate.source_group,
            "button_label": candidate.label,
            "image": str(relative_image),
        })

    for split, rows in exported.items():
        (output / f"{split}.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    with (output / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)
    shutil.copy2(annotation_path, output / "annotations.snapshot.json")
    summary = {
        split: {
            "samples": len(rows),
            "groups": len({row["source_group"] for row in manifest_rows if row["split"] == split}),
            "states": dict(Counter(row["state"] for row in manifest_rows if row["split"] == split)),
        }
        for split, rows in exported.items()
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Exported to: {output}")


if __name__ == "__main__":
    main()
