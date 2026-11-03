import unittest
import numpy as np
from pilot import prepare,summarize,analyze,load_model
class PilotTests(unittest.TestCase):
 def test_shapes(self):
  b,s,_=prepare(np.random.default_rng(4).uniform(1,255,(7,144,144,3)).astype('float32'))
  self.assertEqual(b.shape,(6,3,144,144));self.assertEqual(s.shape,(6,3,9,9));self.assertTrue(np.isfinite(s).all())
 def test_black(self):
  b,s,v=prepare(np.zeros((3,144,144,3)));self.assertEqual(v,0);self.assertTrue(np.isfinite(b).all());self.assertTrue(np.isfinite(s).all())
 def test_missing(self):
  with self.assertRaises(ValueError):prepare(np.zeros((0,144,144,3)))
 def test_nan(self):
  with self.assertRaises(ValueError):prepare(np.full((3,144,144,3),np.nan))
 def test_constant_abstains(self):self.assertIsNone(summarize(np.ones(750))['experimental_rate_per_minute'])
 def test_short_abstains(self):self.assertIsNone(summarize(np.arange(30))['experimental_rate_per_minute'])
 def test_nonfinite_abstains(self):self.assertIsNone(summarize(np.full(750,np.nan))['experimental_rate_per_minute'])
 def test_synthetic_rate(self):
  t=np.arange(750)/25;wave=np.sin(2*np.pi*.25*t);r=summarize(np.r_[np.diff(wave),0]);self.assertIsNotNone(r['experimental_rate_per_minute']);self.assertLessEqual(abs(r['experimental_rate_per_minute']-15),2)
 def test_static_never_infers(self):
  self.assertEqual(analyze(np.full((750,144,144,3),100,np.uint8),None)['status'],'static_input')
 def test_model(self):
  import torch
  torch.set_num_threads(2);m=load_model()
  with torch.inference_mode():y=m([torch.zeros(3,3,144,144),torch.zeros(3,3,9,9)])
  self.assertEqual(tuple(y[2].shape),(3,1));self.assertTrue(torch.isfinite(y[2]).all())

class AcquisitionTests(unittest.TestCase):
 def test_30fps_resampling(self):
  from unittest.mock import patch
  from pilot import acquire
  class Cap:
   def isOpened(self):return True
   def get(self,k):return 30
   def read(self):return True,np.full((240,320,3),100,np.uint8)
   def release(self):pass
  class Detector:
   def empty(self):return False
   def detectMultiScale(self,*args,**kwargs):return [(50,50,120,120)]
  with patch('pilot.cv2.VideoCapture',return_value=Cap()),patch('pilot.cv2.CascadeClassifier',return_value=Detector()),patch('pilot.cv2.destroyAllWindows'):
   frames=acquire('synthetic')
  self.assertGreaterEqual(len(frames),740);self.assertLessEqual(len(frames),750)
if __name__=='__main__':unittest.main()

