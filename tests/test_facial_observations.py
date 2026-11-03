import unittest,math
from types import SimpleNamespace as P
import numpy as np
from modules.facial_observations import FacialObservations

def face(mouth_delta=0,eye=False,roll=0):
    coords={33:(220,160),133:(280,160),160:(240,150),158:(260,150),153:(260,170),144:(240,170),362:(360,160),263:(420,160),385:(380,150),387:(400,150),373:(400,170),380:(380,170),1:(320,220),61:(270,300+mouth_delta),291:(370,300),234:(160,240),454:(480,240),10:(320,80),152:(320,400)}
    if eye:
        for i in [160,158]:coords[i]=(coords[i][0],157)
        for i in [153,144]:coords[i]=(coords[i][0],163)
    lm=[P(x=.5,y=.5) for _ in range(468)]
    for i,(x,y) in coords.items():
        x,y=x-320,y-240;a=math.radians(roll);xx=x*math.cos(a)-y*math.sin(a)+320;yy=x*math.sin(a)+y*math.cos(a)+240
        lm[i]=P(x=xx/640,y=yy/480)
    return lm
class ObservationTests(unittest.TestCase):
    def setUp(self):self.frame=np.full((480,640,3),100,np.uint8);self.obs=FacialObservations()
    def stable(self,lm):
        for i in range(7):s=self.obs.update(lm,self.frame,i*.2)
        return s
    def test_symmetric_quiet(self):self.assertEqual(self.stable(face()),'')
    def test_mouth_difference(self):self.assertIn('زاويتي الفم',self.stable(face(30)))
    def test_eye_difference(self):self.assertIn('العينين',self.stable(face(eye=True)))
    def test_roll_compensated(self):self.assertEqual(self.stable(face(roll=15)),'')
    def test_short_transient(self):self.assertEqual(self.obs.update(face(30),self.frame,0),'')
    def test_gap_resets(self):self.stable(face(30));self.assertEqual(self.obs.update(face(30),self.frame,3),'')
    def test_loss_clears(self):self.stable(face(30));self.assertEqual(self.obs.update(None,self.frame,2),'');self.assertIsNone(self.obs.since)
    def test_bad_quality_clears(self):self.stable(face(30));self.assertEqual(self.obs.update(face(30),self.frame,2,False),'')
    def test_dark(self):self.assertIn('إضاءة',self.obs.update(face(),self.frame*0,0))
    def test_invalid(self):self.assertIn('تعذر',self.obs.update([],self.frame,0))
    def test_pose(self):
        lm=face();lm[1].x=.7;self.assertIn('للأمام',self.obs.update(lm,self.frame,0))
    def test_recovery_clears(self):self.stable(face(30));self.assertEqual(self.obs.update(face(),self.frame,1.4),'')
if __name__=='__main__':unittest.main()
