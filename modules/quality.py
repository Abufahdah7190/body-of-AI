"""Advisory capture quality checks, not presentation attack detection."""
import cv2

def quality_reason(frame,box):
    height,width=frame.shape[:2]; x,y,w,h=box
    if min(w,h)<80 or w/width<.12: return "قرّب وجهك من الكاميرا"
    if x<=1 or y<=1 or x+w>=width-1 or y+h>=height-1: return "أظهر وجهك كاملًا داخل الصورة"
    gray=cv2.cvtColor(frame[y:y+h,x:x+w],cv2.COLOR_BGR2GRAY)
    if gray.size==0: return "تعذر قراءة الوجه"
    if gray.mean()<35: return "الإضاءة ضعيفة — أضف إضاءة أمامك"
    if gray.mean()>235: return "الإضاءة شديدة — خفّف الضوء المباشر"
    if cv2.Laplacian(gray,cv2.CV_64F).var()<12: return "الصورة ضبابية — اثبت أو نظّف العدسة"
    return ""

def final_decision(face_count,state,quality,locked):
    if locked: return "توقف مؤقت للمحاولات",False
    if face_count!=1: return "يلزم وجه واحد واضح للتحقق",False
    if quality: return quality,False
    if state=="PASSED": return "اجتاز التحديات — ليس إثبات هوية أو ضمانًا ضد الانتحال",True
    if state=="FAILED": return "لم تكتمل الخطوات في المهلة",False
    if state=="INTERRUPTED": return "توقف الفحص بسبب التصوير — أعد المحاولة",False
    return "جارٍ التحقق — اتبع التعليمات",False
