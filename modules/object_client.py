"""Optional isolated detector: frames travel in memory, never via image files."""
import threading
import subprocess
import struct
import json
import time
from pathlib import Path
import cv2
from modules.object_filter import ApproachTracker, StableObjects

class ObjectClient:
    def __init__(self,root):
        self.root=Path(root); self.pending=None; self.result=[]; self.detections=[]; self.stamp=0
        self.status="كشف الأشياء غير مفعّل"; self.lock=threading.Lock()
        self.stable=StableObjects()
        self.approach=ApproachTracker()
        self.stop=threading.Event(); self.process=None; self.thread=None
    def start(self):
        python=self.root/".objects/Scripts/python.exe"
        if not python.exists() or not (self.root/"yolov8n.pt").exists():
            self.status="شغّل INSTALL_OBJECTS.cmd لتفعيل كشف الأشياء"; return
        self.status="جاري تحميل كاشف الأشياء"
        self.thread=threading.Thread(target=self._loop,args=(python,),daemon=True); self.thread.start()
    def submit(self,frame):
        if self.thread is not None and self.thread.is_alive():
            with self.lock: self.pending=frame.copy()
    def snapshot(self):
        with self.lock:
            return list(self.result) if time.monotonic()-self.stamp<1.5 else []
    def details(self):
        with self.lock:
            return list(self.detections) if time.monotonic()-self.stamp<1.5 else []
    def _loop(self,python):
        try:
            process=subprocess.Popen([str(python),str(self.root/"object_worker.py")],cwd=str(self.root),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
            self.process=process
            while not self.stop.is_set():
                with self.lock: frame=self.pending; self.pending=None
                if frame is None:
                    self.stop.wait(.1); continue
                _,encoded=cv2.imencode(".jpg",frame,[cv2.IMWRITE_JPEG_QUALITY,85])
                data=encoded.tobytes(); captured=time.monotonic()
                process.stdin.write(struct.pack("<I",len(data))+data); process.stdin.flush()
                line=process.stdout.readline()
                if not line: raise RuntimeError("Detector stopped")
                result=json.loads(line)
                if "error" in result: raise RuntimeError(result["error"])
                stable=self.stable.update(result.get("detections",[]),captured)
                tracked=self.approach.update(stable,captured)
                with self.lock:
                    self.result=result["objects"]
                    self.detections=tracked
                    self.stamp=captured
                self.status="كشف الأشياء يعمل محليًا"
                self.stop.wait(.25)
        except Exception:
            self.status="تعذر كشف الأشياء — أعد تثبيت الإضافة"
        finally:
            if self.process is not None:
                if self.process.poll() is None: self.process.terminate()
                self.process.wait(timeout=5)
    def close(self):
        self.stop.set()
        if self.process is not None and self.process.poll() is None: self.process.terminate()
        if self.thread is not None: self.thread.join(timeout=2)
