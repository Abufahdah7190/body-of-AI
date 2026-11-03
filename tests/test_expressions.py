import unittest
from types import SimpleNamespace
from modules.signals import SignalTracker

class ExpressionTests(unittest.TestCase):
    def face(self):
        lm=[SimpleNamespace(x=.5,y=.5) for _ in range(478)]
        lm[234].x=.2; lm[454].x=.8
        lm[61].x=.35; lm[291].x=.65
        return lm
    def test_smile_needs_hold_and_clears(self):
        lm=self.face(); lm[13].y=.53; lm[14].y=.54
        s=SignalTracker()
        self.assertNotIn("ابتسامة",s.update(lm,640,480,.3,0)[0])
        self.assertIn("ابتسامة",s.update(lm,640,480,.3,.6)[0])
        lm[13].y=.5; lm[14].y=.5
        self.assertNotIn("ابتسامة",s.update(lm,640,480,.3,.7)[0])
    def test_raised_brows(self):
        lm=self.face(); lm[105].y=.3; lm[334].y=.3
        s=SignalTracker(); s.update(lm,640,480,.3,0)
        self.assertIn("حاجبين",s.update(lm,640,480,.3,.7)[0])
    def test_closed_eyes_not_emotion(self):
        lm=self.face(); s=SignalTracker(); s.update(lm,640,480,.1,0)
        self.assertIn("إغلاق عينين",s.update(lm,640,480,.1,1.3)[0])
