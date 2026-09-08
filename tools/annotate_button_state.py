#!/usr/bin/env python3
"""Local browser UI for ON/OFF/UNCERTAIN labels on existing YOLO boxes.

Source YOLO labels are read-only. State labels are atomically saved to a
separate JSON file, so closing and reopening the program resumes the job.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import threading
import webbrowser
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import yaml
from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "configs" / "dataset.yaml"
DEFAULT_OUTPUT = ROOT / "datasets" / "button_state" / "annotations.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
STATES = {"ON", "OFF", "UNCERTAIN", "SKIP"}
NON_BUTTON_LABELS = {
    "blur", "bt_keyhole", "bt_switch", "empty", "hat", "indicator",
    "key", "keyhole", "led", "speaker", "switch", "text", "unknown",
}


@dataclass(frozen=True)
class Candidate:
    id: int
    key: str
    split: str
    image_path: str
    image_name: str
    source_group: str
    line_index: int
    class_id: int
    label: str
    bbox_xywhn: tuple[float, float, float, float]


def source_group(image_name: str) -> str:
    """Group Roboflow derivatives by the name before its .rf. suffix."""
    return Path(image_name).stem.split(".rf.", 1)[0]


def is_button_label(label: str) -> bool:
    return not label.startswith("text_") and label not in NON_BUTTON_LABELS


def load_candidates(
    data_yaml: Path,
    splits: tuple[str, ...] = ("train", "val"),
    include_all_boxes: bool = False,
) -> list[Candidate]:
    data_yaml = data_yaml.expanduser().resolve()
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    raw_names = config.get("names", [])
    names = (
        {int(key): str(value) for key, value in raw_names.items()}
        if isinstance(raw_names, dict)
        else {index: str(value) for index, value in enumerate(raw_names)}
    )
    configured_root = Path(str(config.get("path", "."))).expanduser()
    dataset_root = (
        configured_root.resolve()
        if configured_root.is_absolute()
        else (data_yaml.parent / configured_root).resolve()
    )

    candidates: list[Candidate] = []
    for requested_split in splits:
        config_split = "val" if requested_split == "valid" else requested_split
        relative_images = config.get(config_split)
        if not relative_images:
            raise ValueError(f"Dataset config has no '{config_split}' split: {data_yaml}")
        image_dir = (dataset_root / str(relative_images)).resolve()
        label_dir = image_dir.parent / "labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"Missing images or labels: {image_dir}, {label_dir}")
        image_by_stem = {
            path.stem: path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        }
        split = "valid" if config_split == "val" else config_split
        for label_path in sorted(label_dir.glob("*.txt")):
            image_path = image_by_stem.get(label_path.stem)
            if image_path is None:
                continue
            for line_index, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
                fields = line.split()
                if len(fields) != 5:
                    continue
                try:
                    class_id = int(fields[0])
                    bbox = tuple(float(value) for value in fields[1:])
                except ValueError:
                    continue
                label = names.get(class_id, f"class_{class_id}")
                if not include_all_boxes and not is_button_label(label):
                    continue
                if any(value < 0.0 or value > 1.0 for value in bbox):
                    continue
                image_name = image_path.name
                candidates.append(Candidate(
                    id=len(candidates),
                    key=f"{split}:{image_name}:{line_index}",
                    split=split,
                    image_path=str(image_path),
                    image_name=image_name,
                    source_group=source_group(image_name),
                    line_index=line_index,
                    class_id=class_id,
                    label=label,
                    bbox_xywhn=bbox,  # type: ignore[arg-type]
                ))
    return candidates


class AnnotationStore:
    def __init__(self, output_path: Path, data_yaml: Path) -> None:
        self.output_path = output_path.expanduser().resolve()
        self.data_yaml = data_yaml.expanduser().resolve()
        self.lock = threading.Lock()
        self.annotations: dict[str, dict[str, Any]] = {}
        self.history: list[tuple[str, dict[str, Any] | None]] = []
        if self.output_path.exists():
            payload = json.loads(self.output_path.read_text(encoding="utf-8"))
            self.annotations = dict(payload.get("annotations", {}))

    def _save_locked(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "dataset": str(self.data_yaml),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "labels": ["ON", "OFF", "UNCERTAIN"],
            "annotations": self.annotations,
        }
        temporary = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, self.output_path)

    def set(self, candidate: Candidate, state: str) -> None:
        if state not in STATES:
            raise ValueError(f"Invalid state: {state}")
        with self.lock:
            previous = self.annotations.get(candidate.key)
            self.history.append((candidate.key, dict(previous) if previous else None))
            self.annotations[candidate.key] = {
                "state": state,
                "split": candidate.split,
                "image_name": candidate.image_name,
                "source_group": candidate.source_group,
                "line_index": candidate.line_index,
                "class_id": candidate.class_id,
                "button_label": candidate.label,
                "bbox_xywhn": list(candidate.bbox_xywhn),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            self._save_locked()

    def undo(self) -> str | None:
        with self.lock:
            if not self.history:
                return None
            key, previous = self.history.pop()
            if previous is None:
                self.annotations.pop(key, None)
            else:
                self.annotations[key] = previous
            self._save_locked()
            return key

    def state_for(self, key: str) -> str | None:
        annotation = self.annotations.get(key)
        return str(annotation["state"]) if annotation else None


def _bbox_pixels(candidate: Candidate, width: int, height: int) -> tuple[int, int, int, int]:
    x, y, box_width, box_height = candidate.bbox_xywhn
    x1 = max(0, round((x - box_width / 2) * width))
    y1 = max(0, round((y - box_height / 2) * height))
    x2 = min(width, round((x + box_width / 2) * width))
    y2 = min(height, round((y + box_height / 2) * height))
    return x1, y1, max(x1 + 1, x2), max(y1 + 1, y2)


def render_candidate(candidate: Candidate, kind: str) -> bytes:
    with Image.open(candidate.image_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    bbox = _bbox_pixels(candidate, *image.size)
    if kind == "crop":
        x1, y1, x2, y2 = bbox
        padding = round(max(x2 - x1, y2 - y1) * 0.25)
        crop_box = (
            max(0, x1 - padding), max(0, y1 - padding),
            min(image.width, x2 + padding), min(image.height, y2 + padding),
        )
        rendered = image.crop(crop_box)
        rendered.thumbnail((720, 720), Image.Resampling.LANCZOS)
    else:
        rendered = image.copy()
        width = max(3, round(max(rendered.size) / 220))
        ImageDraw.Draw(rendered).rectangle(bbox, outline=(255, 74, 84), width=width)
        rendered.thumbnail((960, 720), Image.Resampling.LANCZOS)
    output = io.BytesIO()
    rendered.save(output, format="JPEG", quality=92, optimize=True)
    return output.getvalue()


HTML = r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>电梯按钮亮暗标注</title>
<style>
:root{color-scheme:dark;--bg:#0c111b;--panel:#151d2b;--line:#29364b;--text:#f3f6fb;--muted:#93a2b8;--accent:#6da8ff}*{box-sizing:border-box}
body{margin:0;background:radial-gradient(circle at 20% 0,#17233a 0,var(--bg) 38%);color:var(--text);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
header{min-height:64px;display:flex;align-items:center;justify-content:space-between;padding:12px 24px;border-bottom:1px solid var(--line);background:#0c111bd9;position:sticky;top:0;z-index:3;backdrop-filter:blur(10px)}h1{font-size:18px;margin:0}.stats{color:var(--muted)}.stats b{color:var(--text)}
main{padding:18px 24px 28px;max-width:1500px;margin:auto}.toolbar,.actions,.nav{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.toolbar{justify-content:space-between;margin-bottom:14px}.meta{color:var(--muted)}select,label.toggle{background:var(--panel);border:1px solid var(--line);color:var(--text);border-radius:8px;padding:8px 10px}
.viewer{display:grid;grid-template-columns:minmax(340px,1fr) minmax(340px,1fr);gap:14px}.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden;min-height:520px;display:flex;flex-direction:column}.card-title{padding:9px 12px;color:var(--muted);border-bottom:1px solid var(--line)}.image-wrap{flex:1;display:grid;place-items:center;padding:12px;background:#080b11}.image-wrap img{max-width:100%;max-height:670px;object-fit:contain}
.bottom{display:flex;justify-content:space-between;gap:14px;margin-top:14px;align-items:center}.actions{flex:1}.btn{border:1px solid transparent;border-radius:10px;padding:13px 18px;color:#fff;font-weight:700;cursor:pointer;min-width:120px}.btn:hover{filter:brightness(1.12)}.btn:disabled{opacity:.4}.on{background:#087f5b}.off{background:#475569}.uncertain{background:#a16207}.skip{background:#9f3431}.secondary{background:#1d293b;border-color:var(--line);min-width:auto}.key{font-size:12px;opacity:.7;margin-left:6px}.status{padding:8px 12px;border-radius:8px;background:#101827;color:var(--muted)}.status strong{color:var(--accent)}
@media(max-width:850px){.viewer{grid-template-columns:1fr}.card{min-height:360px}.bottom{align-items:stretch;flex-direction:column}.actions .btn{flex:1}header{align-items:flex-start;gap:8px;flex-direction:column}main{padding:12px}}
</style></head><body>
<header><h1>电梯按钮亮暗标注</h1><div class="stats" id="stats">正在读取…</div></header><main>
<div class="toolbar"><div class="meta" id="meta">—</div><div class="toolbar"><select id="classFilter"><option value="">全部按钮类别</option></select><label class="toggle"><input type="checkbox" id="unlabeledOnly" checked> 只看待标注</label></div></div>
<div class="viewer"><section class="card"><div class="card-title">按钮裁剪（含 25% 上下文）</div><div class="image-wrap"><img id="crop" alt="按钮裁剪"></div></section><section class="card"><div class="card-title">原图与已有标注框</div><div class="image-wrap"><img id="context" alt="原图"></div></section></div>
<div class="bottom"><div class="actions"><button class="btn on" data-state="ON">ON <span class="key">1</span></button><button class="btn off" data-state="OFF">OFF <span class="key">2</span></button><button class="btn uncertain" data-state="UNCERTAIN">UNCERTAIN <span class="key">3</span></button><button class="btn skip" data-state="SKIP">跳过 <span class="key">0</span></button></div><div class="nav"><button class="btn secondary" id="previous">← 上一个</button><button class="btn secondary" id="undo">撤销 Z</button><button class="btn secondary" id="next">下一个 →</button><span class="status" id="currentState">当前：<strong>未标注</strong></span></div></div>
</main><script>
let current=null;const $=id=>document.getElementById(id);async function jf(url,opt){const r=await fetch(url,opt);if(!r.ok)throw new Error(await r.text());return r.json()}
function qp(dir){const p=new URLSearchParams({direction:String(dir),unlabeled:$('unlabeledOnly').checked?'1':'0',label:$('classFilter').value});if(current)p.set('current',current.id);return p}
async function nav(dir=1,reset=false){if(reset)current=null;const data=await jf('/api/navigate?'+qp(dir));current=data.item;render(data)}
function render(data){const s=data.stats;$('stats').innerHTML=`总计 <b>${s.total}</b>　已标 <b>${s.annotated}</b>　ON <b>${s.ON}</b>　OFF <b>${s.OFF}</b>　不确定 <b>${s.UNCERTAIN}</b>　跳过 <b>${s.SKIP}</b>`;document.querySelectorAll('.actions button').forEach(b=>b.disabled=!current);if(!current){$('meta').textContent='当前筛选条件下没有更多样本';$('crop').removeAttribute('src');$('context').removeAttribute('src');$('currentState').innerHTML='当前：<strong>—</strong>';return}$('meta').textContent=`类别 ${current.label} · ${current.split} · ${current.image_name} · 框 #${current.line_index+1} · 全局 ${current.position}/${s.total}`;const t=Date.now();$('crop').src=`/api/image/${current.id}?kind=crop&t=${t}`;$('context').src=`/api/image/${current.id}?kind=context&t=${t}`;$('currentState').innerHTML=`当前：<strong>${current.state||'未标注'}</strong>`}
async function mark(state){if(!current)return;await jf('/api/annotate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:current.id,state})});await nav(1)}
async function undo(){const d=await jf('/api/undo',{method:'POST'});if(d.item_id!==null){const x=await jf('/api/item?id='+d.item_id);current=x.item;render(x)}}
document.querySelectorAll('[data-state]').forEach(b=>b.onclick=()=>mark(b.dataset.state));$('previous').onclick=()=>nav(-1);$('next').onclick=()=>nav(1);$('undo').onclick=undo;$('classFilter').onchange=()=>nav(1,true);$('unlabeledOnly').onchange=()=>nav(1,true);
document.addEventListener('keydown',e=>{if(['SELECT','INPUT'].includes(e.target.tagName))return;if(e.key==='1')mark('ON');else if(e.key==='2')mark('OFF');else if(e.key==='3')mark('UNCERTAIN');else if(e.key==='0')mark('SKIP');else if(e.key==='ArrowLeft')nav(-1);else if(e.key==='ArrowRight'||e.key===' ')nav(1);else if(e.key.toLowerCase()==='z')undo()});
(async()=>{const b=await jf('/api/bootstrap');for(const label of b.labels){const o=document.createElement('option');o.value=label;o.textContent=label;$('classFilter').appendChild(o)}await nav(1,true)})().catch(e=>$('meta').textContent='加载失败：'+e.message);
</script></body></html>"""


class AnnotationApp:
    def __init__(self, candidates: list[Candidate], store: AnnotationStore) -> None:
        self.candidates = candidates
        self.by_id = {candidate.id: candidate for candidate in candidates}
        self.position = {candidate.id: index for index, candidate in enumerate(candidates)}
        self.store = store

    def stats(self) -> dict[str, int]:
        counts = {state: 0 for state in STATES}
        keys = {candidate.key for candidate in self.candidates}
        for key, annotation in self.store.annotations.items():
            state = str(annotation.get("state"))
            if key in keys and state in counts:
                counts[state] += 1
        return {"total": len(self.candidates), "annotated": sum(counts.values()), **counts}

    def serialize(self, candidate: Candidate) -> dict[str, Any]:
        value = asdict(candidate)
        value.pop("image_path")
        value["state"] = self.store.state_for(candidate.key)
        value["position"] = self.position[candidate.id] + 1
        return value

    def navigate(self, current_id: int | None, direction: int, unlabeled: bool, label: str) -> Candidate | None:
        if not self.candidates:
            return None
        if current_id is None or current_id not in self.position:
            indices = range(len(self.candidates)) if direction >= 0 else range(len(self.candidates) - 1, -1, -1)
        else:
            start = self.position[current_id] + (1 if direction >= 0 else -1)
            indices = range(start, len(self.candidates)) if direction >= 0 else range(start, -1, -1)
        for index in indices:
            candidate = self.candidates[index]
            if label and candidate.label != label:
                continue
            if unlabeled and self.store.state_for(candidate.key) is not None:
                continue
            return candidate
        return None


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "ButtonStateAnnotator/1.0"

    @property
    def app(self) -> AnnotationApp:
        return self.server.app  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_item(self, candidate: Candidate | None) -> None:
        self.send_json({"item": self.app.serialize(candidate) if candidate else None, "stats": self.app.stats()})

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if parsed.path == "/":
            body = HTML.encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif parsed.path == "/api/bootstrap":
            self.send_json({"labels": sorted({c.label for c in self.app.candidates}), "stats": self.app.stats()})
        elif parsed.path == "/api/navigate":
            raw_current = query.get("current", [None])[0]
            current = int(raw_current) if raw_current is not None else None
            self.send_item(self.app.navigate(
                current,
                int(query.get("direction", ["1"])[0]),
                query.get("unlabeled", ["1"])[0] == "1",
                query.get("label", [""])[0],
            ))
        elif parsed.path == "/api/item":
            try:
                self.send_item(self.app.by_id[int(query["id"][0])])
            except (KeyError, ValueError, IndexError):
                self.send_json({"error": "Unknown candidate"}, HTTPStatus.NOT_FOUND)
        elif parsed.path.startswith("/api/image/"):
            try:
                candidate = self.app.by_id[int(parsed.path.rsplit("/", 1)[1])]
            except (KeyError, ValueError):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            body = render_candidate(candidate, "crop" if query.get("kind", ["crop"])[0] == "crop" else "context")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/annotate":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                candidate = self.app.by_id[int(payload["id"])]
                self.app.store.set(candidate, str(payload["state"]).upper())
            except (KeyError, ValueError, json.JSONDecodeError) as error:
                self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
                return
            self.send_json({"ok": True, "stats": self.app.stats()})
        elif parsed.path == "/api/undo":
            key = self.app.store.undo()
            candidate_id = next((c.id for c in self.app.candidates if c.key == key), None)
            self.send_json({"ok": key is not None, "item_id": candidate_id})
        else:
            self.send_error(HTTPStatus.NOT_FOUND)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate ON/OFF/UNCERTAIN states on existing YOLO boxes")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="YOLO dataset YAML")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="state annotation JSON")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help="splits to load")
    parser.add_argument("--all-boxes", action="store_true", help="also include text/component boxes")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = load_candidates(args.data, tuple(args.splits), args.all_boxes)
    if not candidates:
        raise RuntimeError("No candidate boxes found. Check --data and --splits.")
    store = AnnotationStore(args.output, args.data)
    app = AnnotationApp(candidates, store)
    server = ThreadingHTTPServer((args.host, args.port), RequestHandler)
    server.app = app  # type: ignore[attr-defined]
    url = f"http://{args.host}:{args.port}"
    print(f"Loaded {len(candidates)} candidate boxes")
    print(f"Annotations: {store.output_path}")
    print(f"Open: {url}")
    print("Keys: 1=ON, 2=OFF, 3=UNCERTAIN, 0=skip, Z=undo")
    if not args.no_browser:
        threading.Timer(0.4, partial(webbrowser.open, url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped. All annotations were saved.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
