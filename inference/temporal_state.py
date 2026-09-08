"""Causal button tracking and asymmetric state debounce."""
from dataclasses import dataclass


def iou(a, b):
    area = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area
    return area/union if union > 0 else 0.0


@dataclass
class Track:
    box: list
    label: str
    last_seen: float
    state: str = 'UNCERTAIN'
    negative_since: float | None = None
    negative_state: str | None = None
    last_on: float | None = None
    sustained_on: bool = False


class TemporalButtonStates:
    def __init__(self, off_delay=0.35, max_gap=0.20, iou_threshold=0.35,
                 sustained_off_delay=1.2, on_confirmation_window=0.4):
        self.off_delay = off_delay
        self.max_gap = max_gap
        self.iou_threshold = iou_threshold
        self.sustained_off_delay = max(off_delay, sustained_off_delay)
        self.on_confirmation_window = on_confirmation_window
        self.tracks = {}
        self.next_id = 1

    def update(self, detections, timestamp):
        self.tracks = {k:t for k,t in self.tracks.items() if timestamp-t.last_seen <= self.max_gap}
        candidates = []
        for i,row in enumerate(detections):
            for tid,t in self.tracks.items():
                overlap = iou(row['bbox'],t.box)
                if overlap >= self.iou_threshold:
                    candidates.append((overlap + .05*(row['label']==t.label),i,tid))
        assignments, used = {}, set()
        for _,i,tid in sorted(candidates,reverse=True):
            if i not in assignments and tid not in used:
                assignments[i]=tid;used.add(tid)
        # Missing observations break continuous OFF evidence; no new boxes are fabricated.
        for tid,t in self.tracks.items():
            if tid not in used:
                t.negative_since=None;t.negative_state=None
        result=[]
        for i,row in enumerate(detections):
            raw=row['state']
            if i in assignments:
                tid=assignments[i]; t=self.tracks[tid]
            else:
                tid=self.next_id;self.next_id+=1
                t=Track(row['bbox'],row['label'],timestamp)
                self.tracks[tid]=t
            if raw=='ON':
                if t.last_on is not None and timestamp-t.last_on <= self.on_confirmation_window:
                    t.sustained_on=True
                t.last_on=timestamp
                t.state='ON';t.negative_since=None;t.negative_state=None
            elif t.state=='ON':
                negative='OFF' if raw=='OFF' else 'UNCERTAIN'
                if t.negative_since is None or t.negative_state != negative:
                    t.negative_since=timestamp;t.negative_state=negative
                # Repeated ON evidence justifies a longer OFF confirmation interval.
                # Unknown observations keep the short deadline rather than latching ON.
                delay=self.sustained_off_delay if t.sustained_on and negative=='OFF' else self.off_delay
                if timestamp-t.negative_since >= delay:
                    t.state=negative;t.negative_since=None;t.negative_state=None
                    t.sustained_on=False;t.last_on=None
            else:
                t.state=raw if raw in ('ON','OFF','UNCERTAIN') else 'UNCERTAIN'
            t.box=row['bbox'];t.label=row['label'];t.last_seen=timestamp
            result.append(dict(row,track_id=tid,raw_state=raw,temporal_state=t.state,
                sustained_on=t.sustained_on,
                held_from_history=t.state=='ON' and raw!='ON'))
        return result
