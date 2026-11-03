import unittest,tempfile,json,time
from pathlib import Path
import numpy as np
from modules.pad import PAD,Evidence,crop_patch,secure_decision
from modules.evaluation import record_trial

class PADTests(unittest.TestCase):
    def test_no_single_frame_pass(self):
        e=Evidence();e.add(.99,0);self.assertEqual(e.state(0),'pending')
    def test_stable_live_and_stale(self):
        e=Evidence()
        for t in [0,.3,.6,.9]:e.add(.95,t)
        self.assertEqual(e.state(1),'live');self.assertEqual(e.state(3),'pending')
    def test_attack(self):
        e=Evidence()
        for t in [0,.3,.6,.9]:e.add(.05,t)
        self.assertEqual(e.state(1),'attack')
    def test_uncertain_and_gap_reset(self):
        e=Evidence()
        for t,v in [(0,.95),(.3,.95),(.6,.5),(.9,.95)]:e.add(v,t)
        self.assertEqual(e.state(1),'pending');e.add(.95,4);self.assertEqual(len(e.samples),1)
    def test_invalid(self):
        e=Evidence();e.add(float('nan'),0);self.assertEqual(e.state(0),'pending')
    def test_decision_requires_both(self):
        for state in ['unavailable','pending','attack']:self.assertFalse(secure_decision(True,state))
        self.assertFalse(secure_decision(False,'live'));self.assertTrue(secure_decision(True,'live'))
    def test_missing_models_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            p=PAD(d);self.assertEqual(p.update(None,None,0),'unavailable')
    def test_actual_onnx_load_and_inference(self):
        p=PAD(Path(__file__).resolve().parents[1]/'models/pad');self.assertFalse(p.error)
        frame=np.random.default_rng(1).integers(0,255,(240,320,3),dtype=np.uint8)
        self.assertTrue(0<=p.predict(frame,(100,60,90,100))<=1)
    def test_crop_edges(self):
        frame=np.zeros((240,320,3),np.uint8)
        self.assertEqual(crop_patch(frame,(0,0,100,100),4).shape,(80,80,3))
        with self.assertRaises(ValueError):crop_patch(frame,(-2,0,100,100),4)
    def test_rejection_record_without_images(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'trials.jsonl'
            record_trial(path,{'kind':'print','condition':'normal'},'PAD_REJECTED','id','standard',{'state':'attack'})
            row=json.loads(path.read_text('utf8'));self.assertEqual(row['outcome'],'rejected');self.assertEqual(len(list(Path(d).iterdir())),1)
