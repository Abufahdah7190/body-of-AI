"""Experimental periodic face-motion signal. NOT respiration or liveness evidence."""
from collections import deque
import math
import numpy as np

class BreathingEstimator:
    WINDOW=25.0
    def __init__(self): self.clear()
    def clear(self):
        self.samples=deque(maxlen=2000)
        self.last_time=None
        self.cached=(None,"جارٍ تجميع حركة الوجه — ليست قياس تنفس")
        self.last_compute=-float('inf')
    def update(self,landmarks,width,height,now):
        if landmarks is None or width<=0 or height<=0 or not math.isfinite(now):
            self.clear();return None,"يلزم وجه ثابت وواضح"
        try:
            a,b,n=landmarks[33],landmarks[263],landmarks[1]
            span=math.hypot((a.x-b.x)*width,(a.y-b.y)*height)
            signal=n.y*height/span
            if span<10 or not math.isfinite(signal):raise ValueError('invalid landmarks')
        except (IndexError,AttributeError,ValueError,ZeroDivisionError):
            self.clear();return None,"نقاط الوجه غير صالحة للتحليل"
        if self.last_time is not None and (now<=self.last_time or now-self.last_time>0.5):self.clear()
        # Large displacement or scale change invalidates the previous window.
        if self.samples and abs(signal-self.samples[-1][1])>.05:self.clear()
        self.last_time=now;self.samples.append((now,signal))
        while self.samples and now-self.samples[0][0]>self.WINDOW:self.samples.popleft()
        if len(self.samples)<80 or now-self.samples[0][0]<18:
            return None,"جارٍ تجميع حركة الوجه — ليست قياس تنفس"
        if now-self.last_compute<1:return self.cached
        self.last_compute=now
        t,v=np.asarray(self.samples).T
        regular=np.arange(t[0],t[-1],.1)
        y=np.interp(regular,t,v)
        x=regular-regular.mean();y=y-np.polyval(np.polyfit(x,y,1),x)
        amplitude=float(np.std(y))
        unknown=(None,"حركة غير كافية أو غير منتظمة — لا يمكن استنتاج التنفس")
        if amplitude<.0005 or amplitude>.02:self.cached=unknown;return unknown
        spectrum=np.abs(np.fft.rfft(y*np.hanning(len(y))))**2
        freq=np.fft.rfftfreq(len(y),.1);band=(freq>=.1)&(freq<=.5)
        inds=np.flatnonzero(band);k=inds[np.argmax(spectrum[band])]
        concentration=float(spectrum[max(1,k-1):k+2].sum()/max(spectrum[1:].sum(),1e-12))
        if concentration<.7:self.cached=unknown;return unknown
        rate=round(float(freq[k])*60)
        self.cached=(rate,f"حركة وجه دورية: نحو {rate} دورة/دقيقة — لا تثبت التنفس أو الحيوية")
        return self.cached
