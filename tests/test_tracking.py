import unittest
import tempfile
from pathlib import Path
import csv
import numpy as np
from modules.tracking import MotionTracker
from modules.identity import IdentityStore, CandidateHistory
from modules.audit import AuditStore

class TrackingTests(unittest.TestCase):
    def test_small_motion_keeps_track(self):
        tracker=MotionTracker()
        first=tracker.update([(10,10,100,100)],0)[0]["id"]
        self.assertEqual(first,tracker.update([(15,12,100,100)],.1)[0]["id"])
    def test_absence_breaks_track(self):
        tracker=MotionTracker(); first=tracker.update([(10,10,100,100)],0)[0]["id"]
        tracker.update([],.1)
        self.assertNotEqual(first,tracker.update([(10,10,100,100)],.2)[0]["id"])
    def test_gap_breaks_track(self):
        tracker=MotionTracker(); first=tracker.update([(10,10,100,100)],0)[0]["id"]
        self.assertNotEqual(first,tracker.update([(10,10,100,100)],1)[0]["id"])
    def test_jump_breaks_track(self):
        tracker=MotionTracker(); first=tracker.update([(10,10,100,100)],0)[0]["id"]
        self.assertNotEqual(first,tracker.update([(400,10,100,100)],.1)[0]["id"])
    def test_multiple_faces_break_association(self):
        tracker=MotionTracker(); first=tracker.update([(10,10,100,100)],0)[0]["id"]
        results=tracker.update([(10,10,100,100),(200,10,100,100)],.1)
        self.assertNotIn(first,[x["id"] for x in results])
    def test_candidate_requires_stability(self):
        h=CandidateHistory()
        for _ in range(4): self.assertIsNone(h.update("a"))
        self.assertEqual(h.update("a"),"a")
        self.assertIsNone(h.update(None))
    def test_candidate_clear(self):
        h=CandidateHistory()
        for _ in range(5): h.update("a")
        h.clear(); self.assertIsNone(h.update("a"))

class IdentityTests(unittest.TestCase):
    def test_enroll_reload_delete(self):
        with tempfile.TemporaryDirectory() as folder:
            store=IdentityStore(folder)
            if not store.available: self.skipTest("OpenCV contrib unavailable")
            rng=np.random.default_rng(1)
            sample=rng.integers(0,256,(128,128),dtype=np.uint8)
            key=store.enroll("../name is data",[sample]*15)
            self.assertEqual(store.predict(sample)[0],key)
            reloaded=IdentityStore(folder)
            self.assertEqual(reloaded.name(key),"../name is data")
            self.assertEqual(reloaded.predict(sample)[0],key)
            reloaded.delete(key)
            self.assertFalse((Path(folder)/key).exists())
            self.assertIsNone(reloaded.predict(sample)[0])
            self.assertEqual(IdentityStore(folder).profiles,{})
    def test_unknown_without_profiles(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(IdentityStore(folder).predict(np.zeros((128,128),np.uint8)),(None,None))
    def test_insufficient_samples(self):
        with tempfile.TemporaryDirectory() as folder:
            store=IdentityStore(folder)
            if not store.available: self.skipTest("OpenCV contrib unavailable")
            with self.assertRaises(ValueError): store.enroll("a",[])
    def test_rejection_threshold(self):
        with tempfile.TemporaryDirectory() as folder:
            store=IdentityStore(folder,threshold=-1)
            if not store.available: self.skipTest("OpenCV contrib unavailable")
            sample=np.zeros((128,128),np.uint8)
            store.enroll("a",[sample]*15)
            self.assertIsNone(store.predict(sample)[0])

class AuditTests(unittest.TestCase):
    def test_attempt_link_persists_across_sessions(self):
        with tempfile.TemporaryDirectory() as folder:
            audit=AuditStore(folder); first=audit.session
            audit.add("ATTEMPT_STARTED",attempt="one",track="t1",candidate="p1")
            audit.add("CHALLENGE_TIMEOUT",attempt="one",track="t1",candidate="p1")
            audit.close(); second=AuditStore(folder)
            self.assertNotEqual(first,second.session)
            rows=second.recent(); self.assertEqual(len(rows),2)
            self.assertEqual(rows[0][2:5],("one","t1","p1")); second.close()
    def test_csv_export(self):
        with tempfile.TemporaryDirectory() as folder:
            audit=AuditStore(folder); audit.add("=bad",reason="اختبار")
            file=Path(folder)/"out.csv"; audit.export(file); audit.close()
            with file.open(encoding="utf-8-sig",newline="") as f: rows=list(csv.reader(f))
            self.assertEqual(rows[1][1],"'=bad")
            self.assertIn("اختبار",rows[1][-1])

if __name__=="__main__": unittest.main()
