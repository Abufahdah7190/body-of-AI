"""Cyber club prototype: liveness, temporary tracks and local identity candidates."""
import os
import time
import uuid
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk, filedialog
import cv2
from PIL import Image, ImageTk, ImageDraw
from modules.liveness_utils import average_ear, BlinkTracker
from modules.challenge_response import ChallengeManager
from modules.spoof_detection import detect_rectangular_device
from modules.signals import SignalTracker, session_status
from modules.audio_meter import AudioMeter
from modules.tracking import MotionTracker, face_box
from modules.identity import IdentityStore, CandidateHistory, crop_face
from modules.audit import AuditStore
from modules.camera import LatestCamera
from modules.object_client import ObjectClient
from modules.session_guard import SessionGuard
from modules.quality import quality_reason, final_decision
from modules.evaluation import record_trial
from modules.body_cues import StableCue, hand_shape, body_shape
from modules.facial_observations import FacialObservations
from modules.pad import PAD, secure_decision
from modules.object_filter import display_objects
from modules.face_details import finger_text, hand_face_cue, face_on_screen, bright_panel_faces

DEFAULT_TEST_CASE=None

BG,PANEL,TEXT,MUTED,ACCENT="#121513","#202622","#f1f4ef","#adb7ad","#b9e28c"

class App:
    def __init__(self,root):
        self.root=root; self.closed=False; self.cap=None; self.mesh=None
        self.audio=AudioMeter(); self.audit=None; self.guard=None
        self.pad=PAD(Path(__file__).resolve().parent/"models/pad"); self.pad_state="pending"
        self.pose=None; self.pose_module=None; self.last_pose=0
        self.body_enabled=tk.BooleanVar(value=False)
        self.body_text=tk.StringVar(value="")
        self.facial_observations=FacialObservations(); self.facial_text=tk.StringVar(value="")
        self.gesture_text=tk.StringVar(value="")
        self.body_stable=StableCue(); self.hand_stable=StableCue()
        self.capture_quality=""; self.auto_attempted_track=None; self.test_case=DEFAULT_TEST_CASE; self.attempt_case=None
        self.profile=tk.StringVar(value="standard")
        self.sound=tk.BooleanVar(value=False)
        self.allow_enrollment=tk.BooleanVar(value=False)
        self.hand_landmarks=[];self.session_blinks=BlinkTracker();self.hand_face_stable=StableCue();self.screen_stable=StableCue()
        self.hands=None; self.hand_count=0; self.last_hand=0; self.last_object=0
        self.objects=ObjectClient(Path(__file__).resolve().parent)
        self.visible_objects=[]
        self.observations=tk.StringVar(value="جاري فحص المشهد")
        self.last_tick=time.monotonic(); self.fps=0; self.last_auto=0
        self.enhance=tk.BooleanVar(value=False)
        self.challenge=ChallengeManager(); self.blinks=BlinkTracker(); self.signals=SignalTracker()
        self.tracker=MotionTracker(); self.history=CandidateHistory()
        self.track=None; self.candidate=None; self.last_prediction=0; self.last_motion=0
        self.attempt=None; self.enrollment=None; self.face_count=0; self.verified=False
        self.session_start=time.monotonic(); self.last_scene=None
        self.record_motion=tk.BooleanVar(value=True)
        self.status=tk.StringVar(value="جاري تهيئة الكاميرا")
        self.cue=tk.StringVar(value="بانتظار الوجه")
        self.metrics=tk.StringVar(value="—"); self.instruction=tk.StringVar(value=self.challenge.label())
        self.audio_text=tk.StringVar(value="الميكروفون متوقف")
        self.score_text=tk.StringVar(value="0 / 100")
        self.identity_text=tk.StringVar(value="غير معروف / غير مسجّل")
        self.track_text=tk.StringVar(value="لا يوجد مسار")
        self.note=tk.StringVar(value="المعالجة والتسجيل على هذا الجهاز")
        root.title("Cyber Club | CAMERA ONLY v17 OBSERVATIONS — AUTOMATIC CAMERA")
        root.geometry("960x720"); root.minsize(640,480); root.configure(bg="black")
        self.video=tk.Canvas(root,bg="black",highlightthickness=0,borderwidth=0)
        self.video.pack(fill="both",expand=True)
        self.video_image=self.video.create_image(0,0,anchor="nw")
        root.bind("<space>",lambda e:self.start() if e.widget in (root,self.video) else None)
        root.protocol("WM_DELETE_WINDOW",self.close); root.bind("<Escape>",lambda e:self.close())
        try:
            self.identities=IdentityStore("local_profiles")
            self.audit=AuditStore("security_logs")
            self.guard=SessionGuard(self.audit.directory/"attempt_limits.json")
            self.emit("SESSION_START")
            import mediapipe as mp
            if not hasattr(mp,"solutions"):
                raise RuntimeError("ثبّت إصدار MediaPipe الموجود في requirements.txt داخل بيئة مستقلة")
            self.mesh=mp.solutions.face_mesh.FaceMesh(max_num_faces=3,refine_landmarks=True,min_detection_confidence=.7,min_tracking_confidence=.7)
            self.hands=mp.solutions.hands.Hands(max_num_hands=2,model_complexity=0,min_detection_confidence=.65,min_tracking_confidence=.6)
            self.pose_module=mp.solutions.pose
            self.cap=LatestCamera(0)
            self.objects.start()
            if not self.cap.isOpened(): raise RuntimeError("تعذر فتح الكاميرا")
            self.tick()
        except Exception as exc:
            messagebox.showerror("تعذر التشغيل",str(exc)); self.close()

    def toggle_body(self):
        if self.body_enabled.get():
            self.body_enabled.set(False);self.body_text.set("");self.body_stable=StableCue()
            if self.pose is not None:self.pose.close();self.pose=None
            self.note.set("وصف حركة الجسد متوقف")
            return
        try:
            self.pose=self.pose_module.Pose(model_complexity=1,enable_segmentation=False,min_detection_confidence=.65,min_tracking_confidence=.65)
            self.body_enabled.set(True)
            self.note.set("وصف الجسد مفعّل — أظهر الكتفين والذراعين")
        except Exception as exc:
            self.body_enabled.set(False)
            self.note.set("تعذر تفعيل وصف الجسد")
            messagebox.showerror("حركة الجسد",str(exc))

    def settings_window(self):
        win=tk.Toplevel(self.root); win.title("إعدادات الكاميرا"); win.geometry("600x620"); win.configure(bg=BG)
        tk.Label(win,text="الخصوصية والصوت",bg=BG,fg=TEXT,font=("Segoe UI",17,"bold")).pack(pady=14)
        tk.Button(win,text="تشغيل / إيقاف الميكروفون",command=self.toggle_audio,bg=PANEL,fg=TEXT,font=("Segoe UI",12),relief="flat").pack(pady=10)
        tk.Label(win,textvariable=self.audio_text,bg=BG,fg=MUTED,font=("Segoe UI",11)).pack()
        for title,var in [("تسجيل الحركة محليًا بدون رسم على الوجه",self.record_motion),("تحسين سطوع العرض فقط",self.enhance),("تنبيه صوتي عند إكمال خطوة",self.sound),("السماح بتسجيل ملف وجه وصوره محليًا",self.allow_enrollment)]:
            tk.Checkbutton(win,text=title,variable=var,bg=BG,fg=TEXT,selectcolor=PANEL,activebackground=BG,font=("Segoe UI",11)).pack(anchor="e",padx=20,pady=10)
        tk.Label(win,text="مسار الحركات: standard / no_mouth / no_head",bg=BG,fg=TEXT,font=("Segoe UI",11)).pack(pady=8)
        def changed(value):
            self.invalidate("accessibility_profile_changed")
            self.auto_attempted_track=self.track
            self.note.set("تغيّر مسار الحركات — Space لبدء محاولة جديدة")
        tk.OptionMenu(win,self.profile,"standard","no_mouth","no_head",command=changed).pack()
        tk.Label(win,text="no_mouth: بدون فتح الفم | no_head: بدون حركة الرأس\nهذه بدائل تجريبية وليست طرق تحقق مكافئة مثبتة.\nلا تُحفظ صور المحاولات أو الفيديو أو الصوت.\nتسجيل ملف الوجه يحفظ عينات صور عند تفعيله وبموافقتك.\nالتعبير ليس تشخيصًا نفسيًا، والمطابقة ليست هوية مؤكدة.",bg=BG,fg=MUTED,font=("Segoe UI",11),wraplength=560,justify="right").pack(padx=15,pady=12)

    def test_window(self):
        win=tk.Toplevel(self.root);win.title("اختبار مضبوط — بدون حفظ صور");win.geometry("570x330")
        tk.Label(win,text="اختر الحقيقة المعروفة للتجربة قبل بدئها",font=("Segoe UI",13)).pack(pady=12)
        kind=tk.StringVar(value="live")
        ttk.Combobox(win,textvariable=kind,values=["live","print","screen_replay","virtual_camera","deepfake_replay"],state="readonly").pack(pady=8)
        tk.Label(win,text="live: شخص حقيقي | print: صورة | screen_replay: فيديو شاشة\nvirtual_camera: كاميرا افتراضية | deepfake_replay: فيديو مولّد",font=("Segoe UI",10)).pack()
        tk.Label(win,text="وصف الظروف: إضاءة / جهاز / مسافة (دون أسماء أشخاص)",font=("Segoe UI",11)).pack(pady=8)
        condition=tk.Entry(win,width=55);condition.insert(0,"normal_light / camera_1");condition.pack()
        def apply():
            if self.attempt is not None:
                messagebox.showinfo("المحاولة نشطة","انتظر انتهاء المحاولة الحالية قبل تغيير نوع الاختبار",parent=win);return
            self.test_case={"kind":kind.get(),"condition":condition.get().strip()[:120] or "unspecified"}
            self.note.set("وضع الاختبار مفعّل — Space لتسجيل المحاولة التالية");win.destroy()
        def stop():
            self.test_case=None;self.note.set("وضع الاختبار متوقف");win.destroy()
        tk.Button(win,text="تفعيل الاختبار",command=apply).pack(pady=8)
        tk.Button(win,text="إيقاف وضع الاختبار",command=stop).pack()

    def emit(self,event,**details):
        if self.audit is None: return
        try:
            self.audit.add(event,attempt=self.attempt,track=self.track,candidate=self.candidate,**details)
        except Exception:
            self.note.set("تعذر كتابة السجل — التسجيل غير مكتمل")

    def finish_attempt(self,event,frame=None,**details):
        if self.attempt is None: return
        self.emit(event,**details)
        if self.attempt_case is not None:
            try: record_trial(self.audit.directory/"trials.jsonl",self.attempt_case,event,self.attempt,self.challenge.profile,self.pad.evidence.summary(time.monotonic()))
            except OSError: self.note.set("تعذر حفظ نتيجة الاختبار")
            self.attempt_case=None
        if self.guard is not None:
            if event in ("CHALLENGE_TIMEOUT","PAD_REJECTED"): self.guard.failure()
            elif event=="LIVENESS_DEMO_PASSED": self.guard.success()
        self.attempt=None

    def invalidate(self,reason):
        self.finish_attempt("ATTEMPT_INTERRUPTED",reason=reason)
        self.challenge.reset(); self.blinks=BlinkTracker(); self.signals=SignalTracker()
        self.pad.clear();self.pad_state="pending"
        self.history.clear(); self.candidate=None; self.verified=False
        self.session_start=time.monotonic()
        if self.enrollment is not None:
            self.emit("ENROLLMENT_CANCELLED",reason=reason)
            self.enrollment=None; self.note.set("أُلغي التسجيل بسبب انقطاع التتبع")

    def start(self):
        if self.challenge.state=="ACTIVE":
            self.note.set("أكمل المحاولة الحالية أو انتظر انتهاءها"); return
        if self.guard is None: return
        if self.pad.error:
            self.note.set("نموذج PAD غير متاح — أصلح ملفات models/pad ثم أعد التشغيل");return
        if self.guard.remaining()>0:
            self.note.set(f"انتظر {self.guard.remaining():.0f} ثانية قبل المحاولة التالية"); return
        if self.capture_quality:
            self.note.set(self.capture_quality); return
        if self.face_count!=1 or not self.track:
            self.note.set("انتظر ظهور وجه واحد واضح"); return
        if self.enrollment is not None:
            self.note.set("انتظر انتهاء تسجيل الملف"); return
        if not self.guard.start():
            self.note.set("محاولات متقاربة — انتظر 30 ثانية"); return
        self.finish_attempt("ATTEMPT_CANCELLED",reason="restart")
        self.attempt=uuid.uuid4().hex
        self.attempt_case=dict(self.test_case) if self.test_case else None
        now=time.monotonic(); self.challenge.start(now,self.profile.get()); self.blinks=BlinkTracker()
        self.pad.clear();self.pad_state="pending"
        self.session_start=now; self.verified=False
        self.auto_attempted_track=self.track
        self.last_auto=now
        self.emit("ATTEMPT_STARTED",nonce=self.challenge.nonce,profile=self.profile.get())
        self.note.set("بدأت محاولة جديدة")

    def begin_enrollment(self):
        if not self.allow_enrollment.get():
            self.note.set("التسجيل بالصور متوقف — فعّله من الإعدادات إن أردت"); return
        if self.enrollment is not None: return
        if not self.identities.available:
            messagebox.showerror("المطابقة","يلزم تثبيت opencv-contrib-python"); return
        if not self.verified or self.face_count!=1:
            self.note.set("اجتز التحدي أولًا ثم اضغط تسجيل ملف وجه"); return
        if not messagebox.askyesno("تسجيل محلي بموافقة المشارك","سيُحفظ الاسم و15 صورة مقصوصة للوجه على هذا الجهاز دون تشفير. يمكن حذف الملف من شاشة الملفات. هل المشارك موافق؟"):
            return
        name=simpledialog.askstring("ملف المستخدم","اسم أو رمز المشارك (اسم مُدخل وليس هوية موثقة):",parent=self.root)
        if not name or not name.strip(): return
        if len(name.strip())>60:
            self.note.set("الاسم لا يتجاوز 60 حرفًا"); return
        # Dialogs can remain open longer than the challenge validity.
        if self.challenge.state!="PASSED" or time.monotonic()-self.challenge.passed_at>10:
            self.note.set("انتهت صلاحية الفحص؛ أعد التحدي قبل التسجيل"); return
        self.enrollment={"name":name.strip(),"images":[],"last":0,"track":self.track,"start":time.monotonic()}
        self.emit("ENROLLMENT_STARTED")
        self.note.set("تسجيل عينات الوجه: انظر للأمام مع حركة بسيطة")

    def collect_enrollment(self,crop,suspicious,now):
        e=self.enrollment
        if e is None: return
        if e["track"]!=self.track or suspicious or now-e["start"]>12:
            self.emit("ENROLLMENT_CANCELLED",reason="unstable_scene_or_timeout")
            self.enrollment=None; self.note.set("أُلغي التسجيل؛ أعد المحاولة في مشهد واضح"); return
        if crop is not None and now-e["last"]>.25:
            e["images"].append(crop.copy()); e["last"]=now
            self.note.set(f"عينات الملف: {len(e['images'])}/15")
            if len(e["images"])==15:
                try:
                    profile=self.identities.enroll(e["name"],e["images"])
                    self.emit("PROFILE_ENROLLED",profile=profile)
                    self.note.set("حُفظ الملف؛ الاسم المضاف غير موثّق")
                except Exception as exc:
                    self.emit("ENROLLMENT_FAILED")
                    messagebox.showerror("تعذر التسجيل",str(exc))
                self.enrollment=None; self.history.clear()

    def profiles_window(self):
        win=tk.Toplevel(self.root); win.title("الملفات المحلية"); win.geometry("600x370")
        tk.Label(win,text="الأسماء يضيفها المستخدم؛ لا تثبت الهوية الحقيقية",font=("Segoe UI",12)).pack(pady=12)
        listing=tk.Listbox(win,font=("Segoe UI",12)); listing.pack(fill="both",expand=True,padx=12)
        ids=[]
        def refresh():
            listing.delete(0,"end"); ids.clear()
            for key,value in self.identities.profiles.items():
                ids.append(key); listing.insert("end",f"{value['name']}  |  {key[:8]}")
        def delete():
            selected=listing.curselection()
            if not selected: return
            profile=ids[selected[0]]
            if messagebox.askyesno("حذف الملف","حذف عينات التسجيل والاسم؟ تبقى الأحداث السابقة برمز الملف. الملفات القديمة خارج هذا الملف لا تُحذف بهذا الزر.",parent=win):
                try:
                    self.identities.delete(profile)
                    self.emit("PROFILE_DELETED",profile=profile)
                    self.invalidate("profile_deleted"); refresh()
                except Exception as exc: messagebox.showerror("تعذر الحذف",str(exc),parent=win)
        tk.Button(win,text="حذف الملف المحدد",command=delete,font=("Segoe UI",11)).pack(pady=12)
        refresh()

    def events_window(self):
        win=tk.Toplevel(self.root); win.title("سجل الأحداث المحلي"); win.geometry("1100x500")
        tk.Label(win,text="الأوقات UTC • المرشح ليس هوية مؤكدة • السجل قابل للتعديل",font=("Segoe UI",12)).pack(pady=8)
        columns=("time","event","attempt","track","candidate","details")
        tree=ttk.Treeview(win,columns=columns,show="headings")
        for key,label in zip(columns,["الوقت UTC","الحدث","المحاولة","المسار","الاسم المرشّح","التفاصيل"]):
            tree.heading(key,text=label); tree.column(key,width=165)
        scroll=ttk.Scrollbar(win,orient="vertical",command=tree.yview); scroll.pack(side="right",fill="y")
        tree.configure(yscrollcommand=scroll.set); tree.pack(fill="both",expand=True)
        def refresh():
            for item in tree.get_children(): tree.delete(item)
            for stamp,event,attempt,track,candidate,details in self.audit.recent():
                tree.insert("","end",values=(stamp,event,(attempt or "")[:8],track or "",self.identities.name(candidate),details))
        def export():
            file=filedialog.asksaveasfilename(parent=win,defaultextension=".csv",filetypes=[("CSV","*.csv")])
            if file:
                try: self.audit.export(file)
                except Exception as exc: messagebox.showerror("تعذر التصدير",str(exc),parent=win)
        buttons=tk.Frame(win); buttons.pack(fill="x")
        tk.Button(buttons,text="تحديث آخر 200 حدث",command=refresh).pack(side="right",padx=12,pady=8)
        tk.Button(buttons,text="تصدير جميع الأحداث CSV",command=export).pack(side="right",padx=12,pady=8)
        refresh()

    def toggle_audio(self):
        if self.audio.stream is not None:
            self.audio.close(); self.audio_text.set("الميكروفون متوقف")
        else:
            try: self.audio.start()
            except Exception as exc: messagebox.showerror("الميكروفون",str(exc))

    def tick(self):
        if self.closed: return
        try: self.process_frame()
        except Exception as exc:
            self.finish_attempt("SYSTEM_ERROR",reason=type(exc).__name__)
            messagebox.showerror("توقف الفحص",str(exc)); self.close(); return
        self.root.after(10,self.tick)

    def process_frame(self):
        ok,frame=self.cap.read()
        if not ok: raise RuntimeError("انقطع تدفق الكاميرا")
        frame=cv2.flip(frame,1); h,w=frame.shape[:2]; now=time.monotonic()
        elapsed=now-self.last_tick; self.last_tick=now
        if elapsed>0: self.fps=.9*self.fps+.1/elapsed
        small=cv2.resize(frame,(640,max(1,round(h*640/w)))) if w>640 else frame
        rgb=cv2.cvtColor(small,cv2.COLOR_BGR2RGB)
        result=self.mesh.process(rgb)
        if now-self.last_hand>=.2:
            hand_result=self.hands.process(rgb)
            hand_landmarks=hand_result.multi_hand_landmarks or []
            self.hand_count=len(hand_landmarks);self.hand_landmarks=hand_landmarks
            shapes=list(dict.fromkeys(
                finger_text(hand.landmark,small.shape[1],small.shape[0])
                for hand in hand_landmarks
            ))
            hand_cue=self.hand_stable.update("\n".join(filter(None,shapes)),now)
            self.gesture_text.set(hand_cue)

            self.last_hand=now
        if self.body_enabled.get() and self.pose is not None and now-self.last_pose>=.3:
            pose_result=self.pose.process(rgb)
            pose_lm=pose_result.pose_landmarks.landmark if pose_result.pose_landmarks else None
            pose_cue=body_shape(pose_lm)
            if pose_lm is None:
                self.body_stable=StableCue();self.body_text.set(pose_cue)
            else:self.body_text.set(self.body_stable.update(pose_cue,now))
            self.last_pose=now
        if now-self.last_object>=.5:
            self.objects.submit(small); self.last_object=now
        object_names=self.objects.snapshot()
        object_details=self.objects.details()
        translations={"wallet":"محفظة", "key":"مفتاح", "charger":"شاحن / ملحق شحن", "person":"شخص ظاهر — غير متحقق", "cell phone":"هاتف", "cup":"كأس / كوب", "bottle":"زجاجة", "chair":"كرسي", "laptop":"حاسب محمول", "tv":"شاشة", "book":"كتاب", "keyboard":"لوحة مفاتيح", "mouse":"فأرة", "remote":"جهاز تحكم", "backpack":"حقيبة ظهر", "scissors":"مقص", "clock":"ساعة", "cat":"قطة", "dog":"كلب", "dining table":"طاولة", "couch":"أريكة"}
        labels=[]
        if object_details:
            self.visible_objects=display_objects(object_details)
            labels=[]
            for obj in self.visible_objects[:4]:
                label=translations.get(obj["name"],obj["name"])
                if obj["confidence"]<.70:
                    label+=" — محتمل"
                if obj.get("approaching"):
                    label+=" — يقترب بصريًا (تقديري)"
                labels.append(label)
        else:
            self.visible_objects=[]
        if self.hand_count:
            labels.append(f"يد ظاهرة: {self.hand_count} — لا تثبت الهوية أو الحيوية")
        self.observations.set("\n".join(labels) if labels else "لا يوجد غرض مدعوم بثقة كافية")

        faces=result.multi_face_landmarks or []; count=len(faces); self.face_count=count
        boxes=[face_box(face.landmark,w,h) for face in faces]
        tracks=self.tracker.update(boxes,now)
        current=tracks[0]["id"] if count==1 else None
        if current!=self.track:
            self.facial_observations.clear(); self.facial_text.set("")
            self.session_blinks=BlinkTracker()
            self.invalidate("track_changed_or_lost")
            if self.track: self.emit("TRACK_ENDED")
            self.track=current
            if current: self.emit("TRACK_STARTED")
        scene="NO_FACE" if count==0 else "MULTIPLE_FACES" if count>1 else "ONE_FACE"
        if scene!=self.last_scene:
            self.emit("SCENE_CHANGED",scene=scene,count=count); self.last_scene=scene
        suspicious,_=detect_rectangular_device(small)
        score=0
        screen_face=any(face_on_screen(b,object_details,w,h) for b in boxes)
        # Rescale boxes with the analysis frame, keeping this advisory independent of PAD.
        small_boxes=[tuple(int(v*small.shape[1]/w) for v in b) for b in boxes]
        panel=bool(self.screen_stable.update("panel" if bright_panel_faces(small,small_boxes) else "",now))
        self.capture_quality=quality_reason(frame,boxes[0]) if count==1 else ""
        if panel:self.capture_quality="اشتباه وجه داخل صورة أو شاشة — أبعدها وأعد التحقق"
        if screen_face:self.capture_quality="وجه داخل شاشة مرصودة — أبعد الشاشة للتحقق"
        if screen_face:self.observations.set("وجه ظاهر داخل شاشة — ليس حضورًا بشريًا متحققًا")
        if self.capture_quality and self.challenge.state in ("ACTIVE","PASSED"):
            self.invalidate("capture_quality")
            self.note.set(self.capture_quality)
        if count!=1:
            self.facial_observations.clear(); self.facial_text.set("")
            self.cue.set("بانتظار وجه واحد واضح"); self.metrics.set("الرمش: —")
            self.identity_text.set("المطابقة متوقفة — يلزم وجه واحد")
            self.track_text.set("لا يوجد وجه" if count==0 else "وجوه متعددة — الربط بين الإطارات متوقف")
        else:
            lm=faces[0].landmark; box=boxes[0]; crop=crop_face(frame,box)
            if now-self.last_prediction>.25:
                predicted,distance=self.identities.predict(crop)
                candidate=self.history.update(predicted)
                if candidate!=self.candidate:
                    previous=self.candidate
                    self.candidate=candidate
                    self.emit("CANDIDATE_CHANGED",previous=previous,lbph_distance=distance)
                    # A previously matched face changing invalidates active evidence.
                    if previous is not None:
                        self.finish_attempt("ATTEMPT_INTERRUPTED",reason="candidate_changed")
                        self.challenge.reset(); self.verified=False
                        if self.enrollment is not None:
                            self.emit("ENROLLMENT_CANCELLED",reason="candidate_changed")
                            self.enrollment=None
                self.last_prediction=now
            self.identity_text.set("مرشّح: "+self.identities.name(self.candidate) if self.candidate else "غير معروف / مطابقة غير كافية")
            if not self.identities.available: self.identity_text.set("مطابقة الوجه غير متاحة — ثبّت OpenCV contrib")
            ear=average_ear(lm,w,h); self.blinks.update(ear,now);self.session_blinks.update(ear,now)
            self.facial_text.set(self.facial_observations.update(lm,frame,now,eligible=not bool(self.capture_quality)))
            cue,mouth=self.signals.update(lm,w,h,ear,now,self.audio.loud()); self.cue.set(cue)
            chin=self.hand_face_stable.update(hand_face_cue(lm,self.hand_landmarks,w,h),now)
            if chin:self.cue.set(chin+" | "+cue)
            previous_step=self.challenge.step
            self.challenge.update(lm,mouth,now,ear=ear)
            if self.challenge.step>previous_step and self.sound.get(): self.root.bell()
            bpm=self.blinks.blinks_per_minute(now=now)
            self.metrics.set(f"👁 عدد الرمشات: {self.session_blinks.blink_count} | "+("تجميع 60 ثانية" if now-self.session_start<60 else f"المعدل: {bpm} / دقيقة"))
            score=round(self.challenge.step/4*100)
            center=tracks[0]["center"]
            self.track_text.set(f"TRACK {self.track}  |  X {center[0]/w:.2f}  Y {center[1]/h:.2f}")
            if self.record_motion.get() and now-self.last_motion>=1:
                self.emit("MOTION",x=round(center[0]/w,3),y=round(center[1]/h,3),width=round(box[2]/w,3),height=round(box[3]/h,3))
                self.last_motion=now
            left,right=sorted([lm[234].x,lm[454].x])
            yaw=(lm[1].x-(left+right)/2)/max(right-left,.001)
            eligible=not self.capture_quality and abs(yaw)<.12 and mouth<.2 and self.challenge.state in ("ACTIVE","PASSED")
            self.pad_state=self.pad.update(frame,box,now,eligible)
            if self.pad_state=="attack" and self.attempt:
                self.finish_attempt("PAD_REJECTED",reason="repeated_low_live_scores",pad=self.pad.evidence.summary(now))
                self.challenge.state="INTERRUPTED";self.challenge.reason="pad_rejected"
            self.collect_enrollment(crop,bool(self.capture_quality) or self.pad_state!="live",now)
        locked=self.guard.remaining()>0
        status,base_pass=final_decision(count,self.challenge.state,self.capture_quality,locked)
        self.verified=secure_decision(base_pass,self.pad_state)
        if count==1 and not locked and not self.capture_quality:
            if self.pad.error:status="تعذر فحص PAD — لن يُمنح نجاح"
            elif self.challenge.reason=="pad_rejected":status="لم يجتز فحص هجمات العرض — اشتباه يحتاج مراجعة"
            elif base_pass:
                status="تم التحقق من الحيوية — إنسان حقيقي وفق الفحص التجريبي" if self.verified else "اكتملت الحركات — انظر للأمام حتى يكتمل PAD"
        if count==0 and not locked:
            status="يد بشرية ظاهرة — لا يوجد وجه للتحقق" if self.hand_count else ("شخص ظاهر — الوجه غير واضح للتحقق" if "person" in object_names else "لا يظهر إنسان بوضوح")
        if locked: status+=f" — {self.guard.remaining():.0f} ثانية"
        if screen_face:status="وجه داخل شاشة مرصودة — التحقق متوقف";self.verified=False
        elif panel:status="اشتباه صورة أو شاشة حول الوجه — التحقق متوقف";self.verified=False
        elif count>1:status="أكثر من وجه ظاهر — أبقِ وجهًا واحدًا للتحقق"
        self.status.set(status)
        progress=100 if self.verified else min(95,round(self.challenge.step/4*80)+(15 if self.pad_state=="live" else 0))
        self.score_text.set(f"اكتمال التحقق: {progress}% | {self.challenge.step}/4")

        suffix=f" ({max(0,self.challenge.deadline-now):.0f}s)" if self.challenge.state=="ACTIVE" else ""
        self.instruction.set(("اثبت أمام الكاميرا لإكمال فحص PAD" if base_pass and not self.verified else self.challenge.label())+suffix)
        if self.audio.stream is not None:
            self.audio_text.set(f"{self.audio.db:.0f} dBFS • لا يُحفظ الصوت")
        if self.verified:
            self.finish_attempt("LIVENESS_DEMO_PASSED",frame,score=score,identity_verified=False,pad=self.pad.evidence.summary(now))
        elif self.challenge.state=="FAILED":
            self.finish_attempt("CHALLENGE_TIMEOUT",frame,score=score,not_proof_of_attack=True)
        elif self.challenge.state=="INTERRUPTED":
            self.finish_attempt("ATTEMPT_INTERRUPTED",reason=self.challenge.reason)
        elif self.challenge.state=="IDLE" and self.attempt:
            self.finish_attempt("ATTEMPT_INCONCLUSIVE",frame,reason="evidence_expired")
        if count==1 and not self.capture_quality and self.challenge.state in ("IDLE","FAILED","INTERRUPTED") and (self.auto_attempted_track!=self.track or now-self.last_auto>8) and not locked and self.enrollment is None:
            self.start()
        # Clean camera: no face box, landmark lines, trail or track identifiers.
        # Optional display-only gamma; never affects inference or evidence.
        display=frame
        if self.enhance.get():
            import numpy as np
            lut=np.array([((i/255)**.8)*255 for i in range(256)],dtype=np.uint8)
            display=cv2.LUT(frame,lut)
        image=Image.fromarray(cv2.cvtColor(display,cv2.COLOR_BGR2RGB))
        vw=max(1,self.video.winfo_width()); vh=max(1,self.video.winfo_height())
        ratio=min(vw/image.width,vh/image.height)
        image=image.resize((max(1,int(image.width*ratio)),max(1,int(image.height*ratio))),Image.Resampling.BILINEAR)
        self.photo=ImageTk.PhotoImage(image)
        ox=(vw-image.width)//2; oy=(vh-image.height)//2
        self.video.coords(self.video_image,ox,oy)
        self.video.itemconfigure(self.video_image,image=self.photo)
        self.draw_overlay(ox,oy,image.width,image.height,self.visible_objects)

    def draw_overlay(self,x,y,width,height,objects=()):
        # Tk draws Arabic text directly; camera pixels are never annotated or saved.
        self.video.delete("overlay")
        size=max(10,min(15,int(width/62)))
        gap=int(size*2.1)
        right=x+width-16
        def text(line,yy,color="#ffffff",bold=False):
            font=("Segoe UI",size,"bold" if bold else "normal")
            options=dict(anchor="ne",justify="right",font=font,width=max(100,width-32),tags="overlay")
            self.video.create_text(right+1,yy+1,text=line,fill="#111111",**options)
            self.video.create_text(right,yy,text=line,fill=color,**options)
        for obj in objects:
            box=obj.get("box")
            if not box or len(box)!=4:
                continue
            left=x+box[0]*width
            top_box=y+box[1]*height
            right_box=x+box[2]*width
            bottom_box=y+box[3]*height
            color="#ff5a5f" if obj.get("approaching") else "#83d6a0"
            self.video.create_rectangle(
                left,top_box,right_box,bottom_box,
                outline=color,width=2,tags="overlay",
            )
        top=y+height-120
        text(self.status.get(),top,"#a8f078" if self.verified else "#ffffff",True)
        text(self.instruction.get(),top+gap,"#ffffff",True)
        # النسبة تختفي بعد التأكد أنه إنسان — تُستبدل بتأكيد نهائي بدون رقم يوهم بدقة إحصائية.
        if self.verified:
            text("✓ تم التأكد أنك إنسان — اكتمل الفحص التجريبي",top+gap*2,"#a8f078",True)
        else:
            text(self.score_text.get()+" — مؤشر خطوات، وليس احتمالًا إحصائيًا",top+gap*2,"#dddddd")
        bar_y=y+height-12
        self.video.create_rectangle(x+16,bar_y,x+width-16,bar_y+4,fill="#454e46",outline="",tags="overlay")
        bar_color="#a8f078" if self.verified else "#b9e28c"
        self.video.create_rectangle(x+16,bar_y,x+16+(width-32)*min(self.challenge.step/4,1),bar_y+4,fill=bar_color,outline="",tags="overlay")
        text(self.cue.get(),y+12,"#ffffff",True)
        # عدّ الرمشات على سطر مستقل بلون مميز بدل دمجه مع قائمة الأغراض — أوضح للقراءة السريعة.
        text(self.metrics.get(),y+12+gap,"#8ecae6",True)
        detail_lines=[]
        for section in (self.observations.get(),self.gesture_text.get(),self.body_text.get()):
            detail_lines.extend(line for line in section.splitlines() if line.strip())
        if not detail_lines: detail_lines=["التعرّف والتحقق يعملان تلقائيًا"]
        for index,line in enumerate(detail_lines):
            text(line,y+12+gap*(2+index),"#dddddd")

    def close(self):
        if self.closed: return
        self.closed=True
        self.finish_attempt("ATTEMPT_CANCELLED",reason="app_closed")
        self.emit("SESSION_END")
        self.audio.close()
        self.objects.close()
        if self.pose is not None: self.pose.close()
        if self.hands is not None: self.hands.close()
        if self.cap is not None: self.cap.release()
        if self.mesh is not None: self.mesh.close()
        if self.audit is not None: self.audit.close()
        self.root.destroy()

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--trial-kind",choices=["live","print","screen_replay","virtual_camera","deepfake_replay"])
    parser.add_argument("--condition",default="unspecified")
    args=parser.parse_args()
    if args.trial_kind:DEFAULT_TEST_CASE={"kind":args.trial_kind,"condition":args.condition[:120]}
    os.chdir(Path(__file__).resolve().parent)
    root=tk.Tk(); App(root)
    root.mainloop()
