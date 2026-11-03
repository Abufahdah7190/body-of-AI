import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from modules.challenge_response import ChallengeManager
from modules.session_guard import SessionGuard
from modules.quality import quality_reason,final_decision

def face():
    lm=[SimpleNamespace(x=.5,y=.5) for _ in range(478)]
    lm[234].x=.2; lm[454].x=.8
    return lm

class ChallengeTests(unittest.TestCase):
    def test_nonce_and_profiles(self):
        c=ChallengeManager(); c.start(0,"no_head")
        self.assertEqual(len(c.nonce),32)
        self.assertTrue(set(c.sequence)<=set(["mouth","blink"]))
        self.assertEqual(len(c.sequence),4)
        nonce=c.nonce; c.start(1); self.assertNotEqual(nonce,c.nonce)
    def test_static_pose_cannot_complete(self):
        c=ChallengeManager();c.start(0);lm=face();lm[1].x=.7
        for i in range(130): c.update(lm,.7,i/10)
        self.assertEqual(c.state,"FAILED"); self.assertEqual(c.step,0)
    def test_all_steps_require_return(self):
        c=ChallengeManager();c.start(0);c.sequence=["left","right","mouth","blink"]
        lm=face();now=0
        for _ in range(550):
            mouth=0;ear=.3;lm[1].x=.5
            if c.phase=="ACTION":
                action=c.sequence[c.step]
                if action=="left":lm[1].x=.35
                if action=="right":lm[1].x=.65
                if action=="mouth":mouth=.6
                if action=="blink":ear=.1
            c.update(lm,mouth,now,ear)
            if c.state=="PASSED": break
            now+=.1
        self.assertEqual(c.state,"PASSED");self.assertEqual(c.step,4)
    def test_gap_interrupts(self):
        c=ChallengeManager();c.start(0);c.update(face(),0,0);c.update(face(),0,1)
        self.assertEqual(c.state,"INTERRUPTED")
    def test_global_deadline(self):
        c=ChallengeManager();c.start(0);c.deadline=100
        c.update(face(),0,61)
        self.assertEqual(c.reason,"session_timeout")
    def test_success_expiration(self):
        c=ChallengeManager();c.state="PASSED";c.passed_at=1;c.update(face(),0,12)
        self.assertEqual(c.state,"IDLE")
    def test_unknown_profile_rejected(self):
        with self.assertRaises(ValueError):ChallengeManager().start(0,"skip")

class GuardTests(unittest.TestCase):
    def test_failures_survive_restart_and_expire(self):
        with tempfile.TemporaryDirectory() as d:
            now=[100.];clock=lambda:now[0];path=Path(d)/"limits.json"
            g=SessionGuard(path,clock)
            for _ in range(3):self.assertTrue(g.start());g.failure()
            self.assertFalse(g.start())
            g=SessionGuard(path,clock);self.assertEqual(g.remaining(),30)
            now[0]=131;self.assertTrue(g.start());self.assertEqual(g.failures,0)
    def test_rapid_restarts_throttled(self):
        with tempfile.TemporaryDirectory() as d:
            g=SessionGuard(Path(d)/"limits.json",lambda:100)
            for _ in range(6):self.assertTrue(g.start())
            self.assertFalse(g.start())
    def test_corrupt_state_delays_not_bypasses(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"limits.json";path.write_text("bad")
            self.assertFalse(SessionGuard(path,lambda:100).start())

class QualityTests(unittest.TestCase):
    def test_dark_face(self):
        self.assertIn("الإضاءة ضعيفة",quality_reason(np.zeros((480,640,3),np.uint8),(100,100,200,200)))
    def test_cropped_face(self):
        self.assertIn("كاملًا",quality_reason(np.zeros((480,640,3),np.uint8),(0,100,200,200)))
    def test_no_quality_or_lock_bypass(self):
        self.assertFalse(final_decision(1,"PASSED","dark",False)[1])
        self.assertFalse(final_decision(1,"PASSED","",True)[1])
        self.assertFalse(final_decision(2,"PASSED","",False)[1])
        self.assertTrue(final_decision(1,"PASSED","",False)[1])
