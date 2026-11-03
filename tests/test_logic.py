import unittest
from types import SimpleNamespace as Point
from modules.signals import SignalTracker, session_status
from modules.liveness_utils import BlinkTracker
from modules.challenge_response import ChallengeManager

def landmarks():
    lm=[Point(x=.5,y=.5) for _ in range(468)]
    lm[61].x=.4; lm[291].x=.6
    lm[234].x=.2; lm[454].x=.8
    return lm

class LogicTests(unittest.TestCase):
    def test_blinks_expire(self):
        b=BlinkTracker(); b.update(.1,0); b.update(.1,.1); b.update(.3,.2)
        self.assertEqual(b.blinks_per_minute(now=1),1)
        self.assertEqual(b.blinks_per_minute(now=61),0)
    def test_no_face_never_passes(self):
        self.assertFalse(session_status(0,True,100,False)[1])
    def test_multiple_faces_never_pass(self):
        self.assertFalse(session_status(2,True,100,False)[1])
    def test_challenge_required(self):
        self.assertFalse(session_status(1,False,100,False)[1])
    def test_rectangle_blocks(self):
        self.assertFalse(session_status(1,True,100,True)[1])
    def test_eligible_pass(self):
        self.assertTrue(session_status(1,True,80,False)[1])
    def test_open_mouth_without_audio(self):
        lm=landmarks(); lm[14].y=.7; s=SignalTracker()
        s.update(lm,640,480,.3,0)
        label,_=s.update(lm,640,480,.3,.5)
        self.assertIn("فتح فم",label); self.assertNotIn("صراخ",label)
    def test_loud_open_mouth(self):
        lm=landmarks(); lm[14].y=.7; s=SignalTracker()
        s.update(lm,640,480,.3,0,True)
        label,_=s.update(lm,640,480,.3,.5,True)
        self.assertIn("صراخ محتمل",label)
    def test_challenge_timeout(self):
        c=ChallengeManager(); c.start(0); c.update(landmarks(),0,13)
        self.assertEqual(c.state,"FAILED")
    def test_pass_expires(self):
        c=ChallengeManager(); c.state="PASSED"; c.passed_at=0
        c.update(landmarks(),0,11); self.assertEqual(c.state,"IDLE")

if __name__ == "__main__": unittest.main()
