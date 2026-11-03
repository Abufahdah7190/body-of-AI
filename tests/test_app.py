"""UI integration with synthetic frames; never opens a physical camera."""
import os
import time
import tempfile
import unittest
import tkinter as tk
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from main import App
from modules.audit import AuditStore
from modules.identity import IdentityStore

class Camera:
    def isOpened(self): return True
    def read(self): return True,np.zeros((480,640,3),np.uint8)
    def release(self): pass

class Hands:
    def process(self,frame): return SimpleNamespace(multi_hand_landmarks=[])
    def close(self): pass

class Mesh:
    faces=[]
    def process(self,frame): return SimpleNamespace(multi_face_landmarks=self.faces)
    def close(self): pass

def face():
    lm=[SimpleNamespace(x=.5,y=.5) for _ in range(478)]
    lm[234].x=.3; lm[454].x=.7; lm[10].y=.2; lm[152].y=.8
    lm[61].x=.4; lm[291].x=.6
    return SimpleNamespace(landmark=lm)

class AppTests(unittest.TestCase):
    def test_construct_track_attempt_and_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            mesh=Mesh(); module=SimpleNamespace(solutions=SimpleNamespace(face_mesh=SimpleNamespace(FaceMesh=lambda **kwargs:mesh),hands=SimpleNamespace(Hands=lambda **kwargs:Hands()),pose=SimpleNamespace()))
            root=tk.Tk(); root.withdraw()
            with patch("main.time.monotonic",return_value=1000.0), patch("main.PAD.update",return_value="pending"), patch.dict("sys.modules",{"mediapipe":module}), patch("main.LatestCamera",return_value=Camera()), patch("main.ObjectClient.start"), patch("main.quality_reason",return_value=""), patch("main.IdentityStore",side_effect=lambda path:IdentityStore(os.path.join(directory,path))), patch("main.AuditStore",side_effect=lambda path:AuditStore(os.path.join(directory,path))), patch("main.detect_rectangular_device",return_value=(False,None)):
                app=App(root)
                try:
                    self.assertFalse(app.closed)
                    self.assertFalse(app.video.bind("<Button-3>"))
                    self.assertEqual(app.face_count,0)
                    mesh.faces=[face()]; app.process_frame()
                    self.assertIsNotNone(app.track)
                    self.assertIsNotNone(app.attempt)  # Automatic verification starts.
                    self.assertIn("جارٍ التحقق",app.status.get())
                    app.start(); attempt=app.attempt
                    app.challenge.state="PASSED"; app.challenge.step=4; app.challenge.passed_at=time.monotonic()
                    app.process_frame()
                    self.assertFalse(app.verified)  # Challenges alone must never authorize.
                    self.assertIsNotNone(app.attempt)
                    with patch.object(app.pad,"update",return_value="live"):
                        app.process_frame()
                    self.assertTrue(app.verified, (app.challenge.state, app.challenge.reason, app.pad_state, app.capture_quality, app.guard.remaining(), app.status.get()))
                    self.assertIsNone(app.attempt)
                    rows=app.audit.recent()
                    self.assertTrue(any(r[1]=="LIVENESS_DEMO_PASSED" and r[2]==attempt for r in rows))
                    app.start(); mesh.faces=[]; app.process_frame()
                    self.assertIsNone(app.track); self.assertIsNone(app.attempt)
                    self.assertEqual(len(app.breathing.samples),0)
                    self.assertEqual(app.breathing_text.get(),"")
                    self.assertFalse(app.verified)
                    self.assertIn("لا يظهر إنسان",app.status.get())
                    app.hand_count=1; app.last_hand=time.monotonic()
                    app.process_frame()
                    self.assertIn("يد بشرية",app.status.get())
                    self.assertFalse(app.verified)
                    self.assertTrue(any(r[1]=="ATTEMPT_INTERRUPTED" for r in app.audit.recent()))
                finally:
                    app.close()

if __name__=="__main__": unittest.main()
