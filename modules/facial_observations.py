"""Transient visible face geometry. No disease, identity, or emotion inference."""
import math
import numpy as np
import cv2

class FacialObservations:
    def __init__(self): self.clear()
    def clear(self):
        self.last_time=None;self.pending='';self.since=None
    def update(self,landmarks,frame,now,eligible=True):
        if landmarks is None or not eligible:
            self.clear();return ''
        unknown='تعذر وصف تفاصيل الوجه — حسّن الوضوح والإضاءة'
        try:
            if not math.isfinite(now):raise ValueError('time')
            h,w=frame.shape[:2]
            def pt(i):
                q=landmarks[i]
                if not math.isfinite(q.x+q.y) or not (0<=q.x<=1 and 0<=q.y<=1):raise ValueError('point')
                return np.array([q.x*w,q.y*h],dtype=float)
            a,b=pt(33),pt(263);axis=b-a;span=float(np.linalg.norm(axis))
            if span<45:raise ValueError('small face')
            u=axis/span;v=np.array([-u[1],u[0]])
            nose=pt(1);center=(a+b)/2
            if abs(float(np.dot(nose-center,u)))/span>.22:
                self.clear();return 'وجّه وجهك للأمام لوصف حركة الجانبين'
            points=np.array([pt(i) for i in [234,454,10,152]])
            x1,y1=np.floor(points.min(0)).astype(int);x2,y2=np.ceil(points.max(0)).astype(int)
            roi=frame[y1:y2,x1:x2]
            if roi.size==0:raise ValueError('roi')
            gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
            halves=np.array_split(gray,2,axis=1)
            if gray.mean()<40 or gray.mean()>215 or np.mean(gray>=250)>.20 or abs(halves[0].mean()-halves[1].mean())>55:
                self.clear();return 'إضاءة الوجه غير مناسبة للمقارنة'
            def ear(ids):
                p=[pt(i) for i in ids];den=np.linalg.norm(p[0]-p[3])
                if den<8:raise ValueError('eye unresolved')
                return float((np.linalg.norm(p[1]-p[5])+np.linalg.norm(p[2]-p[4]))/(2*den))
            le=ear([33,160,158,133,153,144]);re=ear([362,385,387,263,373,380])
            if not (0<=le<=.65 and 0<=re<=.65):raise ValueError('geometry')
            mouth_diff=abs(float(np.dot(pt(61)-pt(291),v)))/span
        except (ValueError,IndexError,AttributeError,TypeError,cv2.error):
            self.clear();return unknown
        if self.last_time is not None and (now<=self.last_time or now-self.last_time>.5):self.clear()
        self.last_time=now
        observations=[]
        if abs(le-re)>.10:observations.append('اختلاف ظاهر في اتساع العينين')
        if mouth_diff>.12:observations.append('اختلاف ظاهر في ارتفاع زاويتي الفم')
        candidate=' | '.join(observations)
        if not candidate:
            self.pending='';self.since=None;return ''
        if candidate!=self.pending:self.pending=candidate;self.since=now
        if now-self.since<.8:return ''
        return candidate+' — وصف بصري فقط'
