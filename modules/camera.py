"""Keep only the latest camera frame instead of building a stale queue."""
import threading
import time
import cv2

class LatestCamera:
    def __init__(self,index=0):
        self.cap=cv2.VideoCapture(index)
        self.stop=threading.Event(); self.lock=threading.Lock()
        self.frame=None; self.stamp=0
        if not self.cap.isOpened():
            self.cap.release(); raise RuntimeError("تعذر فتح الكاميرا")
        for prop,value in [(cv2.CAP_PROP_FRAME_WIDTH,1280),(cv2.CAP_PROP_FRAME_HEIGHT,720),(cv2.CAP_PROP_FPS,30),(cv2.CAP_PROP_BUFFERSIZE,1)]:
            self.cap.set(prop,value)
        ok,frame=self.cap.read()
        if not ok:
            self.cap.release(); raise RuntimeError("تعذر قراءة الكاميرا")
        self.frame=frame; self.stamp=time.monotonic()
        self.thread=threading.Thread(target=self._loop,daemon=True); self.thread.start()
    def isOpened(self): return not self.stop.is_set()
    def _loop(self):
        try:
            while not self.stop.is_set():
                ok,frame=self.cap.read()
                if not ok: break
                with self.lock: self.frame=frame; self.stamp=time.monotonic()
                self.stop.wait(.005)
        finally:
            self.cap.release()
    def read(self):
        with self.lock:
            if self.frame is None or time.monotonic()-self.stamp>1.5: return False,None
            return True,self.frame.copy()
    def release(self):
        self.stop.set(); self.thread.join(timeout=1)
