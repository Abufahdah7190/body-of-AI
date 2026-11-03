"""Persistent local throttling; user with file/code access can bypass it."""
import json
import time
from pathlib import Path

class SessionGuard:
    def __init__(self,path,clock=time.time):
        self.path=Path(path); self.clock=clock
        self.failures=0; self.locked_until=0; self.starts=[]
        if self.path.exists():
            try:
                data=json.loads(self.path.read_text(encoding="utf-8"))
                self.failures=int(data["failures"]); self.locked_until=float(data["locked_until"])
                self.starts=[float(t) for t in data["starts"]]
                if not 0<=self.failures<=3: raise ValueError("Invalid failure count")
            except (ValueError,KeyError,TypeError):
                self.failures=3; self.locked_until=clock()+30; self.starts=[]; self.save()
    def save(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        tmp=self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(failures=self.failures,locked_until=self.locked_until,starts=self.starts)),encoding="utf-8")
        tmp.replace(self.path)
    def remaining(self):
        return max(0,self.locked_until-self.clock())
    def start(self):
        now=self.clock()
        if self.remaining()>0: return False
        if self.locked_until:
            self.locked_until=0; self.failures=0
        self.starts=[t for t in self.starts if now-t<60]
        if len(self.starts)>=6:
            self.locked_until=now+30; self.save(); return False
        self.starts.append(now); self.save(); return True
    def failure(self):
        self.failures+=1
        if self.failures>=3: self.failures=3; self.locked_until=self.clock()+30
        self.save()
    def success(self):
        self.failures=0; self.save()
