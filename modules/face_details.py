"""Observable finger extension and hand/face proximity; no mental inference."""
import math
FINGERS=['الإبهام','السبابة','الوسطى','البنصر','الخنصر']
def extended_fingers(lm,width,height):
    if len(lm)!=21:return []
    def p(i):return lm[i].x*width,lm[i].y*height
    def angle(a,b,c):
        u=(p(a)[0]-p(b)[0],p(a)[1]-p(b)[1]);v=(p(c)[0]-p(b)[0],p(c)[1]-p(b)[1]);d=math.hypot(*u)*math.hypot(*v)
        return math.degrees(math.acos(max(-1,min(1,sum(x*y for x,y in zip(u,v))/d)))) if d>1e-6 else 0
    palm=math.dist(p(0),p(9))
    if palm<15 or any(not 0<=q.x<=1 or not 0<=q.y<=1 for q in lm):return []
    values=[angle(2,3,4)>150 and math.dist(p(4),p(5))>palm*.55]
    values += [angle(a,b,c)>155 and math.dist(p(0),p(c))>1.12*math.dist(p(0),p(b)) for a,b,c in [(5,6,8),(9,10,12),(13,14,16),(17,18,20)]]
    return [name for name,yes in zip(FINGERS,values) if yes]
def finger_text(lm,w,h):
    names=extended_fingers(lm,w,h)
    if names:
        count=len(names)
        return f"أصابع اليد: {count} مرفوعة — {'، '.join(names)}"
    return 'أصابع اليد: لا يظهر إصبع مرفوع بوضوح'
def hand_face_cue(face,hands,w,h):
    if not face or not hands:return ''
    chin=(face[152].x*w,face[152].y*h)
    size=abs(face[454].x-face[234].x)*w
    if size<60:return ''
    for hand in hands:
        if any(math.dist((hand.landmark[i].x*w,hand.landmark[i].y*h),chin)<size*.23 for i in [4,8,12,16,20]):
            return 'يد قرب الذقن — لا يمكن استنتاج التفكير'
    return ''
def face_on_screen(box,objects,w,h):
    x,y,bw,bh=box;cx=(x+bw/2)/w;cy=(y+bh/2)/h
    for obj in objects:
        if obj.get('name') not in ('cell phone','tv','laptop') or obj.get('confidence',0)<.45:continue
        l,t,r,b=obj['box']
        if l<=cx<=r and t<=cy<=b and (bw*bh)/(w*h)<=max(0,(r-l)*(b-t))*1.2:return True
    return False

def bright_panel_faces(frame,boxes):
    """Conservative advisory only: bright portrait panels may also be paper/windows."""
    import cv2
    h,w=frame.shape[:2];gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    for threshold in (100,130,160,190):
        _,mask=cv2.threshold(gray,threshold,255,cv2.THRESH_BINARY)
        contours,_=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area=cv2.contourArea(c);x,y,bw,bh=cv2.boundingRect(c)
            if not (.06*w*h<area<.6*w*h and .3<bw/max(bh,1)<.9 and area/max(bw*bh,1)>.8):continue
            for fx,fy,fw,fh in boxes:
                cx,cy=fx+fw/2,fy+fh/2
                if x<=cx<=x+bw and y<=cy<=y+bh and .08<fw*fh/max(bw*bh,1)<.65:return True
    return False
