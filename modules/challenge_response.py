"""Bounded, securely randomized local challenge. Not sensor attestation."""
import secrets

PROFILES={"standard":["left","right","mouth","blink"],"no_mouth":["left","right","blink"],"no_head":["mouth","blink"]}
LABELS={"left":"لف رأسك قليلًا نحو يسار الصورة","right":"لف رأسك قليلًا نحو يمين الصورة","mouth":"افتح فمك ثم أغلقه","blink":"أغلق عينيك للحظة ثم افتحهما"}

class ChallengeManager:
    def __init__(self): self.reset()
    def reset(self):
        self.state="IDLE"; self.sequence=[]; self.step=0; self.hold=None
        self.neutral=None; self.deadline=0; self.passed_at=0; self.session_deadline=0
        self.nonce=None; self.phase="NEUTRAL"; self.ready_at=0; self.last_update=None
        self.profile="standard"; self.reason=""; self.closed_seen=False
        self.required_hold=.35
    def start(self,now,profile="standard"):
        if profile not in PROFILES: raise ValueError("Unknown challenge profile")
        self.reset(); self.profile=profile; self.nonce=secrets.token_hex(16)
        self.sequence=[secrets.choice(PROFILES[profile]) for _ in range(4)]
        self.state="ACTIVE"; self.session_deadline=now+60; self.deadline=now+12
    def label(self):
        if self.state=="IDLE": return "أظهر وجهك لبدء التحقق"
        if self.state=="PASSED": return "اكتملت 4 خطوات — فحص تجريبي صالح 10 ثوانٍ"
        if self.state=="FAILED": return "انتهت المهلة — Space لإعادة المحاولة"
        if self.state=="INTERRUPTED": return "توقف الفحص — حسّن التصوير ثم اضغط Space"
        if self.phase=="NEUTRAL": return f"{self.step+1}/4 — انظر للأمام، أغلق فمك وافتح عينيك"
        if self.phase=="WAIT": return "اثبت للحظة — انتظر التعليمات"
        if self.phase=="RETURN": return "عُد للوضع المحايد لإكمال الخطوة"
        return f"{self.step+1}/4 — {LABELS[self.sequence[self.step]]}"
    def _held(self,condition,now,duration):
        if not condition: self.hold=None; return False
        self.hold=now if self.hold is None else self.hold
        return now-self.hold>=duration
    def update(self,lm,mouth,now,ear=.3):
        if self.state=="PASSED" and now-self.passed_at>10: self.reset()
        if self.state!="ACTIVE": return
        if now>=self.session_deadline or now>=self.deadline:
            self.state="FAILED"; self.reason="session_timeout" if now>=self.session_deadline else "step_timeout"; return
        if self.last_update is not None and now-self.last_update>.8:
            self.state="INTERRUPTED"; self.reason="analysis_gap"; return
        self.last_update=now
        left,right=sorted([lm[234].x,lm[454].x])
        yaw=(lm[1].x-(left+right)/2)/max(right-left,.001)
        neutral=abs(yaw)<.12 and mouth<.2 and ear>.23
        if self.phase=="NEUTRAL":
            if self._held(neutral,now,.35):
                self.neutral=yaw; self.phase="WAIT"; self.hold=None
                self.ready_at=now+.35+secrets.randbelow(550)/1000
                self.required_hold=.25+secrets.randbelow(200)/1000
        elif self.phase=="WAIT":
            if not neutral: self.phase="NEUTRAL"; self.hold=None
            elif now>=self.ready_at: self.phase="ACTION"; self.hold=None
        elif self.phase=="ACTION":
            action=self.sequence[self.step]; delta=yaw-self.neutral
            target=(action=="left" and delta<-.13) or (action=="right" and delta>.13) or (action=="mouth" and mouth>.45) or (action=="blink" and ear<.18)
            if self._held(target,now,self.required_hold): self.phase="RETURN"; self.hold=None
        elif self.phase=="RETURN":
            if self._held(neutral,now,.3):
                self.step+=1; self.hold=None; self.neutral=None
                if self.step==len(self.sequence): self.state="PASSED"; self.passed_at=now
                else: self.phase="NEUTRAL"; self.deadline=min(now+12,self.session_deadline)
