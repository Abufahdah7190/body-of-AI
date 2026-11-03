import unittest
from modules.evaluation import summarize
class EvaluationTests(unittest.TestCase):
    def test_errors_and_nonresponses_are_separate(self):
        records=[dict(kind="print",condition="normal",profile="standard",outcome=o) for o in ["accepted","rejected","inconclusive"]]
        r=summarize(records)["print | normal | standard"]
        self.assertEqual(r["error_rate_decided"],.5)
        self.assertAlmostEqual(r["inconclusive_rate"],1/3)
    def test_live_rejection_is_error(self):
        r=summarize([dict(kind="live",condition="normal",profile="standard",outcome="rejected")])
        self.assertEqual(next(iter(r.values()))["error_rate_decided"],1)
    def test_no_decisions_not_zero_error(self):
        r=summarize([dict(kind="live",condition="dark",profile="no_head",outcome="inconclusive")])
        self.assertIsNone(next(iter(r.values()))["error_rate_decided"])
