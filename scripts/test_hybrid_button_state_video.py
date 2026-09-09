#!/usr/bin/env python3
"""Fast button-state video inference with tracking, event sampling, and Qwen review."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image
from peft import PeftModel
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inference.temporal_state import iou
from tools.annotate_button_state import is_button_label
from evaluate_qwen_vl_button_state import parse_state
from train_qwen_vl_button_state import SYSTEM_PROMPT, USER_PROMPT


@dataclass
class Track:
    track_id: int
    box: list[float]
    label: str
    last_seen: float
    last_review: float = -1e9
    feature_ema: float | None = None
    followup_at: float | None = None
    rows: list[dict] = field(default_factory=list)


def center_distance(a: list[float], b: list[float]) -> float:
    ac = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2)
    bc = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    scale = max(8.0, ((a[2] - a[0]) + (a[3] - a[1]) + (b[2] - b[0]) + (b[3] - b[1])) / 4)
    return ((ac[0] - bc[0]) ** 2 + (ac[1] - bc[1]) ** 2) ** 0.5 / scale


def deduplicate(rows: list[dict], overlap: float = 0.75) -> list[dict]:
    kept: list[dict] = []
    for row in sorted(rows, key=lambda item: item["detection_confidence"], reverse=True):
        if all(iou(row["bbox"], other["bbox"]) < overlap for other in kept):
            kept.append(row)
    return kept


def brightness_feature(crop: np.ndarray) -> float:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    inner = gray[max(0, int(.15 * height)):max(1, int(.85 * height)),
                 max(0, int(.15 * width)):max(1, int(.85 * width))]
    return float(np.percentile(inner, 95) - np.median(inner))


def assign_tracks(by_frame: list[list[dict]], fps: float, max_gap: float, review_interval: float,
                  brightness_delta: float, followup_delay: float, yolo_trigger: float) -> tuple[dict[int, Track], list[dict]]:
    tracks: dict[int, Track] = {}
    next_id = 1
    review_rows: list[dict] = []
    for frame_rows in by_frame:
        if not frame_rows:
            continue
        timestamp = frame_rows[0]["time_s"]
        active = {track_id: track for track_id, track in tracks.items() if timestamp - track.last_seen <= max_gap}
        candidates = []
        for index, row in enumerate(frame_rows):
            for track_id, track in active.items():
                overlap = iou(row["bbox"], track.box)
                distance = center_distance(row["bbox"], track.box)
                if overlap >= .18 or (row["label"] == track.label and distance <= .9):
                    score = overlap + .20 * (row["label"] == track.label) - .04 * distance
                    candidates.append((score, index, track_id))
        assignments: dict[int, int] = {}
        used: set[int] = set()
        for _, index, track_id in sorted(candidates, reverse=True):
            if index not in assignments and track_id not in used:
                assignments[index] = track_id
                used.add(track_id)
        for index, row in enumerate(frame_rows):
            if index in assignments:
                track = tracks[assignments[index]]
            else:
                track = Track(next_id, row["bbox"], row["label"], timestamp)
                tracks[next_id] = track
                next_id += 1
            row["track_id"] = track.track_id
            delta = 0.0 if track.feature_ema is None else row["brightness_feature"] - track.feature_ema
            row["brightness_delta"] = delta
            reasons = []
            if not track.rows:
                reasons.append("new_track")
            if timestamp - track.last_review >= review_interval:
                reasons.append("periodic")
            if abs(delta) >= brightness_delta:
                reasons.append("brightness_rise" if delta > 0 else "brightness_fall")
                track.followup_at = timestamp + followup_delay
            if row["yolo_on_probability"] >= yolo_trigger:
                reasons.append("yolo_on")
            if track.followup_at is not None and timestamp >= track.followup_at:
                reasons.append("brightness_followup")
                track.followup_at = None
            # Around a real light transition, adjacent crops can differ sharply in
            # blur/occlusion. Review every available frame in that short burst so
            # one unlucky crop cannot hide the event. Stable periods stay sparse.
            is_brightness_event = any(reason.startswith("brightness_") for reason in reasons)
            cooldown = .025 if is_brightness_event else min(.12, followup_delay / 2)
            if reasons and timestamp - track.last_review >= cooldown:
                row["review_reasons"] = reasons
                review_rows.append(row)
                track.last_review = timestamp
            else:
                row["review_reasons"] = []
            alpha = .04
            track.feature_ema = row["brightness_feature"] if track.feature_ema is None else (1-alpha) * track.feature_ema + alpha * row["brightness_feature"]
            track.box = row["bbox"]
            track.label = row["label"]
            track.last_seen = timestamp
            track.rows.append(row)
    return tracks, review_rows


def load_qwen(base: str, adapter: Path):
    processor = AutoProcessor.from_pretrained(adapter)
    processor.tokenizer.padding_side = "left"
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForImageTextToText.from_pretrained(base,
        local_files_only=Path(base).is_dir(), quantization_config=quant,
        device_map={"": 0}, dtype=torch.float16, attn_implementation="sdpa")
    return processor, PeftModel.from_pretrained(model, adapter).eval()


def qwen_review(rows: list[dict], frames: list[np.ndarray], processor, model, batch: int) -> None:
    started = time.monotonic()
    for start in range(0, len(rows), batch):
        part = rows[start:start + batch]
        chats = []
        for row in part:
            left, top, right, bottom = row["crop_bbox"]
            crop = frames[row["frame"]][top:bottom, left:right]
            image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            chats.append([
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": USER_PROMPT}]},
            ])
        inputs = processor.apply_chat_template(chats, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        outputs = processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        for row, raw in zip(part, outputs):
            row["qwen_state"] = parse_state(raw)
            row["qwen_raw"] = raw.strip()
        done = min(start + batch, len(rows))
        if start == 0 or done == len(rows) or done % 40 == 0:
            elapsed = time.monotonic() - started
            print(f"Qwen events {done}/{len(rows)}; {done/max(elapsed, .01):.2f} events/s", flush=True)


def propagate_states(tracks: dict[int, Track], off_delay: float, fast_off_delay: float,
                     brightness_hold_margin: float) -> None:
    for track in tracks.values():
        state = "UNCERTAIN"
        negative_since: float | None = None
        negative_reviews = 0
        confirmed_off_features: list[float] = []
        for row in track.rows:
            reviewed = "qwen_state" in row
            if reviewed:
                raw = row["qwen_state"]
                if raw == "ON":
                    state = "ON"
                    negative_since = None
                    negative_reviews = 0
                elif raw == "OFF":
                    if state != "ON":
                        state = "OFF"
                        confirmed_off_features.append(row["brightness_feature"])
                    else:
                        off_baseline = (float(np.median(confirmed_off_features))
                                        if confirmed_off_features else None)
                        visibly_above_off = (off_baseline is not None and
                            row["brightness_feature"] >= off_baseline + brightness_hold_margin)
                        if visibly_above_off:
                            negative_since = None
                            negative_reviews = 0
                        else:
                            negative_since = row["time_s"] if negative_since is None else negative_since
                            negative_reviews += 1
                            fast_change = ("brightness_fall" in row["review_reasons"] or
                                           "brightness_followup" in row["review_reasons"])
                            delay = fast_off_delay if fast_change and negative_reviews >= 2 else off_delay
                            if row["time_s"] - negative_since >= delay:
                                state = "OFF"
                                confirmed_off_features.append(row["brightness_feature"])
                                negative_since = None
                                negative_reviews = 0
                elif state == "UNCERTAIN":
                    state = "UNCERTAIN"
            row["state"] = state
            row["state_source"] = "qwen" if reviewed else "track_cache"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detector", type=Path, default=ROOT / "models/best.pt")
    parser.add_argument("--classifier", type=Path, default=ROOT / "models/button_state_yolo26n_cls.pt")
    parser.add_argument("--base", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--adapter", type=Path, default=ROOT / "models/button_state_lora")
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--conf", type=float, default=.25)
    parser.add_argument("--review-interval", type=float, default=.75)
    parser.add_argument("--track-max-gap", type=float, default=1.0)
    parser.add_argument("--brightness-delta", type=float, default=10.0)
    parser.add_argument("--followup-delay", type=float, default=.25)
    parser.add_argument("--off-delay", type=float, default=1.2)
    parser.add_argument("--fast-off-delay", type=float, default=.20)
    parser.add_argument("--brightness-hold-margin", type=float, default=8.0)
    parser.add_argument("--yolo-trigger", type=float, default=.25)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output exists: {args.output}")
    args.output.mkdir(parents=True)
    total_started = time.monotonic()
    cap = cv2.VideoCapture(str(args.source))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames or fps <= 0:
        raise RuntimeError(f"Cannot decode {args.source}")
    height, width = frames[0].shape[:2]

    phase = time.monotonic()
    detector = YOLO(args.detector)
    by_frame: list[list[dict]] = [[] for _ in frames]
    crops = []
    rows = []
    for frame_index, frame in enumerate(frames):
        result = detector.predict(frame, device=args.device, conf=args.conf, imgsz=640, verbose=False)[0]
        candidates = []
        for box in result.boxes:
            label = str(detector.names[int(box.cls.item())])
            if not is_button_label(label):
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            pad = .25 * max(x2 - x1, y2 - y1)
            left, top = max(0, round(x1-pad)), max(0, round(y1-pad))
            right, bottom = min(width, round(x2+pad)), min(height, round(y2+pad))
            if right <= left or bottom <= top:
                continue
            crop = frame[top:bottom, left:right].copy()
            candidates.append({"frame": frame_index, "time_s": frame_index / fps, "label": label,
                "detection_confidence": float(box.conf.item()), "bbox": [x1, y1, x2, y2],
                "crop_bbox": [left, top, right, bottom], "brightness_feature": brightness_feature(crop),
                "_crop": crop})
        for row in deduplicate(candidates):
            crops.append(row.pop("_crop"))
            rows.append(row)
            by_frame[frame_index].append(row)
    del detector
    torch.cuda.empty_cache()
    detection_s = time.monotonic() - phase

    phase = time.monotonic()
    classifier = YOLO(args.classifier)
    fast_results = classifier.predict(crops, imgsz=224, batch=128, device=args.device, verbose=False)
    on_index = next(index for index, name in classifier.names.items() if name == "ON")
    for row, result in zip(rows, fast_results):
        row["yolo_on_probability"] = float(result.probs.data[on_index].item())
    del classifier
    torch.cuda.empty_cache()
    fast_classifier_s = time.monotonic() - phase

    tracks, review_rows = assign_tracks(by_frame, fps, args.track_max_gap, args.review_interval,
        args.brightness_delta, args.followup_delay, args.yolo_trigger)
    phase = time.monotonic()
    processor, model = load_qwen(args.base, args.adapter)
    qwen_review(review_rows, frames, processor, model, args.batch)
    qwen_s = time.monotonic() - phase
    propagate_states(tracks, args.off_delay, args.fast_off_delay, args.brightness_hold_margin)

    predictions_path = args.output / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    raw_video = args.output / "annotated_raw.mp4"
    writer = cv2.VideoWriter(str(raw_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    colors = {"ON": (30, 220, 30), "OFF": (220, 160, 40), "UNCERTAIN": (0, 210, 255)}
    for frame_index, original in enumerate(frames):
        frame = original.copy()
        for row in by_frame[frame_index]:
            x1, y1, x2, y2 = map(round, row["bbox"])
            color = colors[row["state"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            review_mark = "*" if row["state_source"] == "qwen" else ""
            label = f'{row["label"]} {row["state"]}{review_mark} T{row["track_id"]}'
            cv2.putText(frame, label, (max(0, x1), max(18, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, .48, color, 2, cv2.LINE_AA)
        cv2.rectangle(frame, (0, height-44), (width, height), (20, 20, 20), -1)
        cv2.putText(frame, f"Hybrid YOLO + Qwen | {frame_index/fps:.2f}s | * = Qwen review",
                    (10, height-15), cv2.FONT_HERSHEY_SIMPLEX, .52, (255,255,255), 1, cv2.LINE_AA)
        writer.write(frame)
    writer.release()
    output_video = args.output / "annotated.mp4"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(raw_video), "-i", str(args.source), "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "libx264", "-preset", "fast", "-crf", "19", "-pix_fmt", "yuv420p",
        "-c:a", "copy", "-movflags", "+faststart", str(output_video)], check=True)

    per_label = defaultdict(Counter)
    for row in rows:
        per_label[row["label"]][row["state"]] += 1
    total_s = time.monotonic() - total_started
    summary = {
        "source": str(args.source.resolve()), "output_video": str(output_video.resolve()),
        "frames": len(frames), "fps": fps, "duration_s": len(frames)/fps,
        "detections_after_deduplication": len(rows), "tracks": len(tracks),
        "qwen_calls": len(review_rows), "qwen_call_reduction_vs_per_crop": 1-len(review_rows)/max(1,len(rows)),
        "state_counts": dict(Counter(row["state"] for row in rows)),
        "qwen_raw_counts": dict(Counter(row.get("qwen_state") for row in review_rows)),
        "per_label": {label: dict(counts) for label, counts in per_label.items()},
        "timing_s": {"detection": round(detection_s,3), "fast_classifier": round(fast_classifier_s,3),
                     "qwen_load_and_review": round(qwen_s,3), "total": round(total_s,3)},
        "parameters": {key: getattr(args,key) for key in ("review_interval", "track_max_gap", "brightness_delta",
            "followup_delay", "off_delay", "fast_off_delay", "brightness_hold_margin", "yolo_trigger")},
        "note": "No frame-level ground truth is available; counts describe predictions, not accuracy.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
