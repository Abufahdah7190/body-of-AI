import unittest
import numpy as np
from types import SimpleNamespace as P
from modules.face_details import extended_fingers,face_on_screen,hand_face_cue,bright_panel_faces
class DetailTests(unittest.TestCase):
 def test_no_hands_or_faces(self):self.assertEqual(hand_face_cue(None,[],640,480),'')
 def test_screen_contains_face(self):
  o=[dict(name='cell phone',confidence=.8,box=[.2,.1,.8,.9])]
  self.assertTrue(face_on_screen((250,140,100,100),o,640,480))
  self.assertFalse(face_on_screen((0,0,40,40),o,640,480))
 def test_low_confidence_not_screen(self):self.assertFalse(face_on_screen((250,140,100,100),[dict(name='cell phone',confidence=.3,box=[0,0,1,1])],640,480))
 def test_blank_not_panel(self):self.assertFalse(bright_panel_faces(np.zeros((480,640,3),np.uint8),[(200,200,100,100)]))
 def test_finger_names(self):
  # Synthetic straight middle/ring/pinky with bent remaining fingers.
  for base,joint,tip,name in [(5,6,8,'السبابة'),(9,10,12,'الوسطى'),(13,14,16,'البنصر'),(17,18,20,'الخنصر')]:
   lm=[P(x=.5,y=.6) for _ in range(21)];lm[0]=P(x=.5,y=.9);lm[9]=P(x=.5,y=.6)
   lm[base]=P(x=.5,y=.6);lm[joint]=P(x=.5,y=.45);lm[tip]=P(x=.5,y=.2)
   self.assertIn(name,extended_fingers(lm,640,480))
 def test_chin_is_not_thought(self):
  lm=[P(x=.5,y=.5) for _ in range(478)];lm[234].x=.2;lm[454].x=.8
  hands=[P(landmark=[P(x=.5,y=.5) for _ in range(21)])]
  self.assertIn('لا يمكن',hand_face_cue(lm,hands,640,480))
