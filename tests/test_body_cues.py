import unittest
from types import SimpleNamespace as P
from modules.body_cues import StableCue,body_shape,hand_shape
class BodyCueTests(unittest.TestCase):
    def test_two_raised_hands(self):
        lm=[P(x=.5,y=.5,visibility=1) for _ in range(33)]
        lm[15].y=.2;lm[16].y=.2
        self.assertIn("اليدان",body_shape(lm))
        lm[16].y=.6;self.assertIn("واحدة",body_shape(lm))
    def test_missing_landmarks_unknown(self):
        self.assertEqual(body_shape(None),"الجسد غير واضح")
        self.assertEqual(hand_shape([],640,480),"")
    def test_invisible_arm_not_classified(self):
        lm=[P(x=.5,y=.5,visibility=1) for _ in range(33)]
        lm[15].visibility=.2
        self.assertIn("بوضوح",body_shape(lm))
    def test_hold_and_immediate_loss(self):
        c=StableCue();self.assertEqual(c.update("hand",0),"")
        self.assertEqual(c.update("hand",.5),"hand")
        self.assertEqual(c.update("",.6),"")
        self.assertEqual(c.update("hand",.7),"")
    def test_long_gap_cannot_count_as_hold(self):
        c=StableCue();c.update("hand",0)
        self.assertEqual(c.update("hand",2),"")
