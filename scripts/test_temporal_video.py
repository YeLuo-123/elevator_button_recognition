"""Replay saved model predictions through a causal temporal filter and render comparison."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import subprocess
import sys
import cv2
import imageio_ffmpeg

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from inference.temporal_state import TemporalButtonStates


def draw(frame,rows,field,title,timestamp):
    out=frame.copy(); h,w=out.shape[:2]
    colors={'ON':(30,220,30),'OFF':(220,160,40),'UNCERTAIN':(0,210,255),'INVALID':(40,40,240)}
    for row in rows:
        state=row[field]; x1,y1,x2,y2=map(round,row['bbox'])
        color=colors[state];cv2.rectangle(out,(x1,y1),(x2,y2),color,2)
        held=field=='temporal_state' and row['held_from_history']
        label=f"{row['label']} {state}"+(' [HOLD]' if held else '')
        (tw,th),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,.55,2)
        x=min(max(0,x1),w-tw-6);y=max(th+6,y1)
        cv2.rectangle(out,(x,y-th-5),(x+tw+5,y+4),(20,20,20),-1)
        cv2.putText(out,label,(x+2,y),cv2.FONT_HERSHEY_SIMPLEX,.55,color,2,cv2.LINE_AA)
    cv2.rectangle(out,(0,h-70),(w,h),(20,20,20),-1)
    cv2.putText(out,f'{title} | {timestamp:.2f}s',(12,h-44),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(out,'ON green | OFF blue | HOLD = recent ON evidence',(12,h-16),cv2.FONT_HERSHEY_SIMPLEX,.49,(255,255,255),1,cv2.LINE_AA)
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,default=ROOT/'test.mp4')
    p.add_argument('--predictions',type=Path,default=ROOT/'runs/predict/test_qwen_state/predictions.jsonl')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--off-delay',type=float,default=.35)
    p.add_argument('--sustained-off-delay',type=float,default=1.2)
    p.add_argument('--baseline-predictions',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    raw=[json.loads(line) for line in a.predictions.read_text().splitlines()]
    grouped=defaultdict(list)
    for row in raw:grouped[row['frame']].append(row)
    baseline=defaultdict(list)
    if a.baseline_predictions:
        for line in a.baseline_predictions.read_text().splitlines():
            row=json.loads(line);baseline[row['frame']].append(row)
    cap=cv2.VideoCapture(str(a.source));fps=cap.get(5);w,h=int(cap.get(3)),int(cap.get(4))
    f=TemporalButtonStates(off_delay=a.off_delay,sustained_off_delay=a.sustained_off_delay)
    writers={name:cv2.VideoWriter(str(a.output/f'{name}_raw.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w*scale,h)) for name,scale in [('temporal',1),('comparison',2)]}
    assert all(writer.isOpened() for writer in writers.values())
    rows=[];index=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        filtered=f.update(grouped[index],index/fps);rows.extend(filtered)
        before=(draw(frame,baseline[index],'temporal_state','BEFORE: fixed OFF delay',index/fps)
                if a.baseline_predictions else draw(frame,filtered,'raw_state','BEFORE: independent frames',index/fps))
        after=draw(frame,filtered,'temporal_state','AFTER: confirmed ON / longer hold',index/fps)
        writers['temporal'].write(after);writers['comparison'].write(cv2.hconcat([before,after]))
        if index in (48,55,60,80,85,110,130):cv2.imwrite(str(a.output/f'comparison_{index:04d}.jpg'),cv2.hconcat([before,after]))
        index+=1
    cap.release()
    for writer in writers.values():writer.release()
    (a.output/'predictions.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
    summaries={}
    for label in ('2','12'):
        subset=[r for r in rows if r['label']==label]
        info={'detection_instances':len(subset),'detected_frames':len({r['frame'] for r in subset})}
        for field in ('raw_state','temporal_state'):
            on=sorted({r['frame'] for r in subset if r[field]=='ON'})
            info[field]={'counts':dict(Counter(r[field] for r in subset)),
                'on_frames':on,'first_on_s':on[0]/fps if on else None}
        summaries[label]=info
    summary=dict(frames=index,fps=fps,off_delay_s=a.off_delay,max_gap_s=.2,
        sustained_off_delay_s=a.sustained_off_delay,on_confirmation_window_s=.4,
        trigger='One raw ON triggers immediately; no state confidence is available in saved JSON predictions.',
        changed_detections=sum(r['raw_state']!=r['temporal_state'] for r in rows),
        labels=summaries,note='Causal replay of identical trained-model outputs. No ground truth; counts are not accuracy.')
    (a.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    for name in writers:
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-nostdin','-hide_banner','-loglevel','error',
            '-i',str(a.output/f'{name}_raw.mp4'),'-i',str(a.source),'-map','0:v:0','-map','1:a:0?',
            '-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p','-c:a','copy','-movflags','+faststart',
            str(a.output/f'{name}.mp4')],check=True)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
