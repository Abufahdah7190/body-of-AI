"""Temporary motion tracks; a track ID is never a person's identity."""
from collections import deque
import math
import uuid

class MotionTracker:
    def __init__(self):
        self.tracks = {}

    def update(self, boxes, now):
        # Treat ambiguous multi-face scenes as new observations, never link identities.
        old = list(self.tracks.values())
        updated = {}
        for box in boxes:
            x,y,w,h = box
            center = (x+w/2,y+h/2)
            track = None
            if len(boxes)==1 and len(old)==1:
                prev = old[0]
                distance = math.dist(center,prev["center"])
                if now-prev["last"] < .7 and distance < max(w,h)*.6:
                    track = prev
            if track is None:
                track = {"id":uuid.uuid4().hex[:12], "points":deque(maxlen=80), "since":now}
            track.update(box=box,center=center,last=now)
            track["points"].append((now,*center))
            while track["points"] and now-track["points"][0][0] > 5:
                track["points"].popleft()
            updated[track["id"]] = track
        self.tracks = updated
        return list(updated.values())


def face_box(landmarks,width,height):
    xs=[lm.x*width for lm in landmarks]; ys=[lm.y*height for lm in landmarks]
    x=max(0,int(min(xs))); y=max(0,int(min(ys)))
    right=min(width,int(max(xs))); bottom=min(height,int(max(ys)))
    return x,y,max(0,right-x),max(0,bottom-y)
