import unittest,tempfile,json
from pathlib import Path
from unittest.mock import patch
from PIL import Image
import object_training as t
from modules.object_filter import display_objects
from modules.signals import SignalTracker
from types import SimpleNamespace as P
class TrainingTests(unittest.TestCase):
 def test_prepare_empty_reconfigure_and_portable_yaml(self):
  with tempfile.TemporaryDirectory() as d,patch.object(t,'DATA',Path(d)):
   t.prepare(['wallet']);t.prepare(['wallet','key'])
   self.assertEqual(t.names(),['wallet','key']);self.assertNotIn(d,(Path(d)/'data.yaml').read_text())
 def test_empty_training_rejected(self):
  with tempfile.TemporaryDirectory() as d,patch.object(t,'DATA',Path(d)):
   t.prepare(['wallet'])
   with self.assertRaises(ValueError):t.validate()
 def test_duplicate_split_and_invalid_box(self):
  with tempfile.TemporaryDirectory() as d,patch.object(t,'DATA',Path(d)):
   t.prepare(['wallet'])
   for split in ['train','val','test']:
    Image.new('RGB',(32,32),'red').save(Path(d)/'images'/split/'a.png')
    (Path(d)/'labels'/split/'a.txt').write_text('0 .5 .5 2 2')
   with self.assertRaises(ValueError):t.validate()
   r=json.loads((Path(d)/'validation.json').read_text());self.assertTrue(any('Duplicate' in e for e in r['errors']));self.assertTrue(any('Invalid' in e for e in r['errors']))
 def test_cannot_reindex_labeled_dataset(self):
  with tempfile.TemporaryDirectory() as d,patch.object(t,'DATA',Path(d)):
   t.prepare(['wallet']);(Path(d)/'labels/train/a.txt').write_text('')
   with self.assertRaises(ValueError):t.prepare(['key'])
 def test_valid_synthetic_structure(self):
  with tempfile.TemporaryDirectory() as d,patch.object(t,'DATA',Path(d)):
   t.prepare(['wallet']);n=0
   for split,count in [('train',10),('val',3),('test',3)]:
    for i in range(count):
     n+=1;Image.new('RGB',(32,32),(n,0,0)).save(Path(d)/'images'/split/f'{i}.png')
     (Path(d)/'labels'/split/f'{i}.txt').write_text('0 .5 .5 .5 .5')
   self.assertEqual(t.validate()['errors'],[])
 def test_custom_class_display(self):
  row={'name':'wallet','custom':True,'confidence':.8};self.assertEqual(display_objects([row]),[row])
class AddedExpressionTests(unittest.TestCase):
 def face(self):
  lm=[P(x=.5,y=.5) for _ in range(478)];lm[234].x=.2;lm[454].x=.8;lm[61].x=.35;lm[291].x=.65;return lm
 def test_down_corners_no_sadness_claim(self):
  lm=self.face();lm[13].y=.45;lm[14].y=.45;s=SignalTracker();s.update(lm,640,480,.3,0)
  self.assertIn('لا يثبت الحزن',s.update(lm,640,480,.3,.8)[0])
 def test_one_brow(self):
  lm=self.face();lm[105].y=.4;s=SignalTracker();s.update(lm,640,480,.3,0)
  self.assertIn('حاجب واحد',s.update(lm,640,480,.3,.8)[0])
 def test_long_gap_resets_expression_hold(self):
  lm=self.face();lm[105].y=.4;s=SignalTracker();s.update(lm,640,480,.3,0)
  self.assertNotIn('حاجب واحد',s.update(lm,640,480,.3,5)[0])
