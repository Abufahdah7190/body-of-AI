"""Temporal geometric expression cues, never psychological diagnoses."""
import math
from modules.liveness_utils import eye_aspect_ratio, LEFT_EYE, RIGHT_EYE

class SignalTracker:
    def __init__(self):
        self.open_since=None
        self.closed_since=None
        self.smile_since=None
        self.brow_since=None
        self.wink_since=None
        self.tilt_since=None
        self.extra_since={};self.last_time=None

    def update(self,landmarks,width,height,ear,now,loud=False):
        if self.last_time is not None and (now<self.last_time or now-self.last_time>2):
            self.__init__()
        self.last_time=now
        def xy(i): return landmarks[i].x*width,landmarks[i].y*height
        def distance(a,b): return math.dist(xy(a),xy(b))
        def since(old,condition): return (now if old is None else old) if condition else None
        mouth_width=distance(61,291)
        face_width=distance(234,454)
        mouth=distance(13,14)/max(mouth_width,1e-6)
        self.open_since=since(self.open_since,mouth>.45)
        self.closed_since=since(self.closed_since,ear<.19)
        # Corner lift relative to the mouth axis, normalized to mouth width.
        left,right=xy(61),xy(291)
        dx,dy=right[0]-left[0],right[1]-left[1]
        if dx<0: dx,dy=-dx,-dy
        nx,ny=-dy/max(mouth_width,1e-6),dx/max(mouth_width,1e-6)
        center=((landmarks[13].x+landmarks[14].x)*width/2,(landmarks[13].y+landmarks[14].y)*height/2)
        corners=((left[0]+right[0])/2,(left[1]+right[1])/2)
        lift=((center[0]-corners[0])*nx+(center[1]-corners[1])*ny)/max(mouth_width,1e-6)
        smile=face_width>1 and mouth_width/face_width>.36 and lift>.045 and mouth<.45
        brow_gap=(distance(105,159)+distance(334,386))/max(2*face_width,1e-6)
        brows=face_width>1 and brow_gap>.16 and ear>.22
        one_brow=face_width>1 and abs(distance(105,159)-distance(334,386))/face_width>.055 and not brows
        down_corners=face_width>1 and lift<-.045 and mouth<.2
        eye_dx,eye_dy=xy(263)[0]-xy(33)[0],xy(263)[1]-xy(33)[1]
        asym=abs((right[1]-left[1])-(right[0]-left[0])*eye_dy/max(abs(eye_dx),1)) / max(mouth_width,1)
        signals={"brow":one_brow,"down":down_corners,"asym":asym>.12 and mouth<.3,"narrow":.19<=ear<.24}
        for key,condition in signals.items():self.extra_since[key]=since(self.extra_since.get(key),condition)
        def held(key,duration=.7):
            start=self.extra_since.get(key)
            return start is not None and now-start>=duration
        self.smile_since=since(self.smile_since,smile)
        self.brow_since=since(self.brow_since,brows)
        left_ear=eye_aspect_ratio(landmarks,LEFT_EYE,width,height)
        right_ear=eye_aspect_ratio(landmarks,RIGHT_EYE,width,height)
        wink=min(left_ear,right_ear)<.17 and max(left_ear,right_ear)>.25
        tilt=abs(landmarks[33].y-landmarks[263].y)*height/max(distance(33,263),1e-6)>.25
        self.wink_since=since(self.wink_since,wink)
        self.tilt_since=since(self.tilt_since,tilt)
        if self.open_since is not None and now-self.open_since>=.35:
            label="نداء / صراخ محتمل — صوت مرتفع مع فتح الفم" if loud else "فتح فم واسع — كلام أو تثاؤب محتمل"
        elif self.closed_since is not None and now-self.closed_since>=1.2:
            label="إغلاق عينين ممتد — لا يكفي للحكم على التعب"
        elif self.wink_since is not None and now-self.wink_since>=.35:
            label="غمزة محتملة — إغلاق عين واحدة"
        elif self.smile_since is not None and now-self.smile_since>=.55:
            label=("ابتسامة عريضة محتملة" if mouth_width/max(face_width,1)>.48 and lift>.08 else "ابتسامة محتملة")+" — ارتفاع زوايا الفم"
        elif self.brow_since is not None and now-self.brow_since>=.6:
            label="ارتفاع حاجبين محتمل — تعبير لافت"
        elif held("brow"):
            label="ارتفاع حاجب واحد محتمل — لا يثبت الشك"
        elif held("down"):
            label="هبوط زوايا الفم / عبوس محتمل — لا يثبت الحزن"
        elif held("asym"):
            label="عدم تماثل زوايا الفم ظاهر — ليس دليل ثقة أو سخرية"
        elif held("narrow"):
            label="تضييق العينين ظاهر — لا يثبت غضبًا أو ألمًا"
        elif self.tilt_since is not None and now-self.tilt_since>=.7:
            label="ميل الرأس ظاهر — لا يحدد الحالة النفسية"
        elif mouth>.20:
            label="حركة فم ظاهرة — كلام محتمل"
        else:
            label="محايد ظاهريًا — لا يوجد تعبير بارز حاليًا"
        if self.open_since is not None and now-self.open_since>=.35 and brows:
            label+=" | رفع حاجبين — لا يكفي لإثبات المفاجأة أو الخوف"
        return label,mouth

def session_status(face_count, passed, score, suspicious):
    if face_count == 0:
        return "لا يوجد وجه", False
    if face_count != 1:
        return "أكثر من وجه — أوقف التحقق", False
    if suspicious:
        return "مؤشر جسم مستطيل — يحتاج مراجعة", False
    if passed and score >= 75:
        return "اجتاز الفحص التجريبي", True
    return "جارٍ جمع الأدلة / تنفيذ التحدي", False
