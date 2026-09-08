"""Detect buttons and classify each crop with the trained Qwen adapter."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import cv2
from PIL import Image
import torch
from peft import PeftModel
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.annotate_button_state import is_button_label
from train_qwen_vl_button_state import SYSTEM_PROMPT, USER_PROMPT
from evaluate_qwen_vl_button_state import parse_state


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--base', default='Qwen/Qwen3-VL-2B-Instruct')
    p.add_argument('--batch', type=int, default=4)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    cap = cv2.VideoCapture(str(a.source))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames or fps <= 0:
        raise RuntimeError('Video cannot be decoded')
    height, width = frames[0].shape[:2]
    detector = YOLO(ROOT / 'models/best.pt')
    detections, crops = [], []
    for fi, frame in enumerate(frames):
        result = detector.predict(frame, device=0, conf=0.25, imgsz=640, verbose=False)[0]
        for box in result.boxes:
            label = str(detector.names[int(box.cls.item())])
            if not is_button_label(label):
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            pad = 0.25 * max(x2-x1, y2-y1)
            left, top = max(0, round(x1-pad)), max(0, round(y1-pad))
            right, bottom = min(width, round(x2+pad)), min(height, round(y2+pad))
            if right <= left or bottom <= top:
                continue
            crops.append(Image.fromarray(cv2.cvtColor(frame[top:bottom,left:right], cv2.COLOR_BGR2RGB)))
            detections.append(dict(frame=fi, time_s=fi/fps, label=label,
                                   detection_confidence=float(box.conf.item()),
                                   bbox=[x1,y1,x2,y2], crop_bbox=[left,top,right,bottom]))
        if fi % 40 == 0:
            print(f'Detection {fi+1}/{len(frames)}; crops {len(crops)}', flush=True)
    del detector
    torch.cuda.empty_cache()
    print(f'Classifying {len(crops)} crops from {len(frames)} frames', flush=True)
    processor = AutoProcessor.from_pretrained(a.adapter)
    processor.tokenizer.padding_side = 'left'
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                              bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForImageTextToText.from_pretrained(a.base, quantization_config=quant,
        device_map={'':0}, dtype=torch.float16, attn_implementation='sdpa')
    model = PeftModel.from_pretrained(model, a.adapter).eval()
    started = time.monotonic()
    with (a.output/'predictions.jsonl').open('w') as log:
        for start in range(0, len(crops), a.batch):
            conversations = [[
                {'role':'system','content':[{'type':'text','text':SYSTEM_PROMPT}]},
                {'role':'user','content':[{'type':'image','image':im},{'type':'text','text':USER_PROMPT}]}
            ] for im in crops[start:start+a.batch]]
            inputs = processor.apply_chat_template(conversations, tokenize=True,
                add_generation_prompt=True, return_dict=True, return_tensors='pt', padding=True).to(model.device)
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=16, do_sample=False)
            texts = processor.batch_decode(generated[:,inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            for offset, text in enumerate(texts):
                row = detections[start+offset]
                row.update(state=parse_state(text), raw=text.strip())
                log.write(json.dumps(row, ensure_ascii=False)+'\n')
            log.flush()
            if start % 80 == 0 or start+a.batch >= len(crops):
                done = min(start+a.batch,len(crops))
                elapsed = time.monotonic()-started
                print(f'Qwen {done}/{len(crops)}; elapsed {elapsed:.0f}s; ETA {elapsed/done*(len(crops)-done):.0f}s', flush=True)
    by_frame = [[] for _ in frames]
    for row in detections:
        by_frame[row['frame']].append(row)
    colors = {'ON':(30,220,30), 'OFF':(220,160,40), 'UNCERTAIN':(0,210,255), 'INVALID':(40,40,240)}
    output = a.output/'annotated.mp4'
    writer = cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
    if not writer.isOpened():
        raise RuntimeError('Cannot create video writer')
    for fi, frame in enumerate(frames):
        for row in by_frame[fi]:
            x1,y1,x2,y2 = map(round,row['bbox'])
            color = colors[row['state']]
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            label = f"{row['label']} {row['state']}"
            (tw,th),_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.55,2)
            tx = min(max(x1,0),max(0,width-tw-6))
            ty = max(th+6,y1)
            cv2.rectangle(frame,(tx,ty-th-5),(tx+tw+5,ty+4),(20,20,20),-1)
            cv2.putText(frame,label,(tx+2,ty),cv2.FONT_HERSHEY_SIMPLEX,0.55,color,2,cv2.LINE_AA)
        cv2.rectangle(frame,(0,0),(width,58),(20,20,20),-1)
        cv2.putText(frame,f'YOLO + Qwen LoRA | {fi/fps:.2f}s',(12,23),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(frame,'ON: green | OFF: blue | UNCERTAIN: yellow',(12,46),cv2.FONT_HERSHEY_SIMPLEX,0.48,(255,255,255),1,cv2.LINE_AA)
        writer.write(frame)
        if fi in [0,len(frames)//2,len(frames)-1]:
            cv2.imwrite(str(a.output/f'preview_{fi:04d}.jpg'),frame)
    writer.release()
    counts = Counter(r['state'] for r in detections)
    summary = dict(source=str(a.source.resolve()), adapter=str(a.adapter.resolve()),
        frames=len(frames), fps=fps, duration_s=len(frames)/fps,
        detections=len(detections), state_counts=dict(counts),
        note='Counts are per-frame detections, not unique buttons. No ground truth or accuracy measured.',
        frame_on_counts=[sum(r['state']=='ON' for r in rows) for rows in by_frame])
    (a.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    print(f'Saved {output}',flush=True)


if __name__ == '__main__':
    main()
