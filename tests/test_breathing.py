import math,unittest
from types import SimpleNamespace as P
from modules.breathing import BreathingEstimator

def landmarks(y=.5):
    a=[P(x=.5,y=.5) for _ in range(468)]
    a[33]=P(x=.4,y=.4);a[263]=P(x=.6,y=.4);a[1].y=y
    return a
class MotionTests(unittest.TestCase):
    def series(self,f):
        b=BreathingEstimator()
        for i in range(750):r=b.update(landmarks(f(i/30)),640,480,i/30)
        return b,r
    def test_constant_abstains(self):self.assertIsNone(self.series(lambda t:.5)[1][0])
    def test_synthetic_period(self):
        b,r=self.series(lambda t:.5+.001*math.sin(2*math.pi*.25*t))
        self.assertIsNotNone(r[0]);self.assertLessEqual(abs(r[0]-15),2)
    def test_missing_clears(self):
        b,_=self.series(lambda t:.5);b.update(None,640,480,26);self.assertEqual(len(b.samples),0)
    def test_gap_clears(self):
        b,_=self.series(lambda t:.5);self.assertIsNone(b.update(landmarks(),640,480,27)[0]);self.assertEqual(len(b.samples),1)
    def test_clock_reversal(self):
        b,_=self.series(lambda t:.5);b.update(landmarks(),640,480,1);self.assertEqual(len(b.samples),1)
    def test_large_motion(self):
        b,_=self.series(lambda t:.5);self.assertIsNone(b.update(landmarks(.8),640,480,25)[0]);self.assertEqual(len(b.samples),1)
    def test_invalid(self):self.assertIsNone(BreathingEstimator().update([],640,480,0)[0])
    def test_nonfinite(self):self.assertIsNone(BreathingEstimator().update(landmarks(float('nan')),640,480,0)[0])
    def test_trend_abstains(self):self.assertIsNone(self.series(lambda t:.5+.0001*t)[1][0])
