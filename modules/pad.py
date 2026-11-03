"""Experimental RGB PAD. Scores are not calibrated human probabilities."""
from pathlib import Path
from collections import deque
import hashlib,json
import cv2
import numpy as np

class Evidence:
    def __init__(self): self.clear()
    def clear(self): self.samples=deque(maxlen=8);self.last=None
    def add(self,score,now):
        if not np.isfinite(score) or not 0<=score<=1: self.clear();return
        if self.last is not None and (now<=self.last or now-self.last>1.5):self.clear()
        self.samples.append((now,float(score)));self.last=now
    def state(self,now):
        if self.last is None or now-self.last>1.5 or now<self.last:return 'pending'
        recent=[v for t,v in self.samples if now-t<=3]
        if len(recent)<4:return 'pending'
        # Require every recent sample to agree; intermediate scores remain inconclusive.
        if all(v>=.8 for v in recent):return 'live'
        if all(v<=.2 for v in recent):return 'attack'
        return 'pending'
    def summary(self,now):
        return {'state':self.state(now),'samples':len(self.samples),'scores':[round(v,4) for _,v in self.samples]}

def crop_patch(frame,box,scale):
    h,w=frame.shape[:2];x,y,bw,bh=box
    if min(bw,bh)<=0 or x<0 or y<0 or x+bw>w or y+bh>h:raise ValueError('Invalid face box')
    scale=min((h-1)/bh,(w-1)/bw,scale)
    nw,nh=bw*scale,bh*scale;cx,cy=x+bw/2,y+bh/2
    left,top,right,bottom=cx-nw/2,cy-nh/2,cx+nw/2,cy+nh/2
    if left<0:right-=left;left=0
    if top<0:bottom-=top;top=0
    if right>w-1:left-=right-w+1;right=w-1
    if bottom>h-1:top-=bottom-h+1;bottom=h-1
    patch=frame[int(top):int(bottom)+1,int(left):int(right)+1]
    return cv2.resize(patch,(80,80))

class PAD:
    def __init__(self,directory):
        self.evidence=Evidence();self.last_run=-1e9;self.error='';self.nets=[]
        try:
            folder=Path(directory);meta=json.loads((folder/'manifest.json').read_text('utf8'))
            for entry in meta['models']:
                path=folder/entry['file']
                if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:raise ValueError('Model integrity mismatch')
                self.nets.append((cv2.dnn.readNetFromONNX(str(path)),entry['scale']))
            if len(self.nets)!=2:raise ValueError('Two PAD models required')
        except Exception as exc:self.error=type(exc).__name__;self.nets=[]
    def clear(self):self.evidence.clear();self.last_run=-1e9
    def predict(self,frame,box):
        values=[]
        for net,scale in self.nets:
            patch=crop_patch(frame,box,scale)
            net.setInput(cv2.dnn.blobFromImage(patch,1.0,(80,80),swapRB=False))
            logits=net.forward().reshape(-1)
            if logits.size!=3 or not np.isfinite(logits).all():raise ValueError('Invalid PAD output')
            probs=np.exp(logits-logits.max());probs/=probs.sum();values.append(float(probs[1]))
        # Both models must support live. Disagreement must not become an attack decision.
        if min(values)>=.8:return min(values)
        if max(values)<=.2:return max(values)
        return .5
    def update(self,frame,box,now,eligible=True):
        if self.error:return 'unavailable'
        if not eligible:self.evidence.clear();return 'pending'
        if now-self.last_run>=.3:
            self.last_run=now
            try:self.evidence.add(self.predict(frame,box),now)
            except Exception as exc:self.error=type(exc).__name__;self.evidence.clear();return 'unavailable'
        return self.evidence.state(now)

def secure_decision(base_pass,pad_state):
    return bool(base_pass and pad_state=='live')
