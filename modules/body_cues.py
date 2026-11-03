"""Describe visible hand and body geometry without inferring intent or emotion."""
import math

class StableCue:
    def __init__(self): self.value=""; self.since=0; self.last=None
    def update(self,value,now):
        if self.last is None or now-self.last>.8 or value!=self.value:
            self.value=value;self.since=now
        self.last=now
        return value if value and now-self.since>=.45 else ""

def hand_shape(lm,width,height):
    if len(lm)!=21: return ""
    def point(i):return lm[i].x*width,lm[i].y*height
    def angle(a,b,c):
        pa,pb,pc=point(a),point(b),point(c)
        u=(pa[0]-pb[0],pa[1]-pb[1]);v=(pc[0]-pb[0],pc[1]-pb[1])
        length=math.hypot(*u)*math.hypot(*v)
        if length<1e-6:return 0
        return math.degrees(math.acos(max(-1,min(1,(u[0]*v[0]+u[1]*v[1])/length))))
    palm=math.dist(point(0),point(9))
    if palm<15:return ""
    states=[]
    for base,joint,tip in [(5,6,8),(9,10,12),(13,14,16),(17,18,20)]:
        extended=angle(base,joint,tip)>155 and math.dist(point(0),point(tip))>math.dist(point(0),point(joint))*1.12
        states.append(extended)
    thumb=angle(2,3,4)>150 and math.dist(point(4),point(5))>palm*.55
    if all(states) and thumb:return "كف مفتوح ظاهر"
    if not any(states) and thumb and lm[4].y<lm[3].y-.025:return "إبهام مرفوع ظاهر"
    if not any(states) and not thumb:return "قبضة محتملة"
    if states==[True,False,False,False]:return "سبابة ممتدة ظاهرة"
    if states==[True,True,False,False]:return "إصبعان ممتدان ظاهران"
    return "وضعية يد غير مصنفة"

def body_shape(lm):
    if not lm or len(lm)<33:return "الجسد غير واضح"
    required=[11,12,13,14,15,16]
    if any(getattr(lm[i],"visibility",0)<.7 or not 0<=lm[i].x<=1 or not 0<=lm[i].y<=1 for i in required):
        return "أظهر الكتفين والذراعين بوضوح"
    raised=[lm[15].y<lm[11].y-.08,lm[16].y<lm[12].y-.08]
    if all(raised):return "اليدان مرفوعتان فوق مستوى الكتفين"
    if any(raised):return "يد واحدة مرفوعة فوق مستوى الكتف"
    return "الذراعان ظاهران دون رفع واضح"
