import io
import time
import unittest
from unittest.mock import patch
import numpy as np
from modules.camera import LatestCamera
from modules.object_client import ObjectClient
from object_worker import read_exact

class FakeCamera:
    def isOpened(self): return True
    def set(self,*args): return True
    def read(self): return True,np.zeros((32,32,3),np.uint8)
    def release(self): pass

class CaptureTests(unittest.TestCase):
    def test_latest_frame_is_copied_and_closes(self):
        with patch("modules.camera.cv2.VideoCapture",return_value=FakeCamera()):
            camera=LatestCamera()
            try:
                ok,frame=camera.read(); self.assertTrue(ok)
                frame[:]=255
                self.assertEqual(camera.read()[1].max(),0)
            finally: camera.release()
            self.assertFalse(camera.thread.is_alive())
    def test_stale_object_result_not_shown(self):
        client=ObjectClient("."); client.result=["person"]; client.stamp=time.monotonic()-3
        self.assertEqual(client.snapshot(),[])
        client.stamp=time.monotonic(); self.assertEqual(client.snapshot(),["person"])
    def test_pipe_truncation(self):
        self.assertIsNone(read_exact(io.BytesIO(b"a"),4))
        self.assertEqual(read_exact(io.BytesIO(b"abcd"),4),b"abcd")
