from pathlib import Path
import io,json,zipfile,hashlib,time,copy
import numpy as np
from PIL import Image
import torch
from torch import nn
ROOT=Path(__file__).resolve().parent
OUT=Path('outputs/expression_training_v16');OUT.mkdir(exist_ok=True,parents=True)
CLASSES=['angry','disgust','fear','happy','neutral','sad','surprise']
torch.set_num_threads(4);torch.manual_seed(42);rng=np.random.default_rng(42)
started=time.time();groups={};bad=[]
with zipfile.ZipFile(ROOT/'dataset.zip') as z:
 for name in z.namelist():
  if not name.endswith('.jpg'):continue
  split,label,_=name.split('/')
  try:
   with Image.open(io.BytesIO(z.read(name))) as im:
    a=np.array(im.convert('L'),dtype=np.uint8)
   if a.shape!=(48,48):raise ValueError('shape')
   h=hashlib.sha256(a.tobytes()).hexdigest()
   groups.setdefault(h,[]).append((split,CLASSES.index(label),a,name))
  except Exception as e:bad.append([name,str(e)])
sets={'train':[],'val':[],'test':[]};conflicts=0;overlap=0
for h,g in groups.items():
 if len({r[1] for r in g})>1:conflicts+=len(g);continue
 if len({r[0] for r in g})>1:overlap+=len(g);continue
 sets['test' if g[0][0]=='test' else 'train'].append((h,*g[0][1:]))
train=sets['train'];sets['train']=[]
for c in range(7):
 rows=[r for r in train if r[1]==c];rng.shuffle(rows);cut=max(1,int(len(rows)*.15))
 sets['val']+=rows[:cut];sets['train']+=rows[cut:]
manifest={k:[{'sha256':r[0],'label':CLASSES[r[1]],'source':r[3]} for r in v] for k,v in sets.items()}
(OUT/'split_manifest.json').write_text(json.dumps(manifest),encoding='utf8')
audit={'counts':{k:len(v) for k,v in sets.items()},'conflicting_label_rows_removed':conflicts,'cross_original_split_rows_removed':overlap,'decode_errors':bad,'classes':CLASSES,'seed':42,'limits':'Exact pixel duplicates removed. Subject identities and near duplicates unavailable; this is not a subject-independent or camera benchmark.'}
(OUT/'data_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(json.dumps(audit),flush=True)
data={k:(torch.tensor(np.stack([r[2] for r in v])[:,None],dtype=torch.float32)/255,torch.tensor([r[1] for r in v])) for k,v in sets.items()}
model=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Flatten(),nn.Linear(64*6*6,128),nn.ReLU(),nn.Dropout(.35),nn.Linear(128,7))
model.load_state_dict(torch.load('outputs/expression_training_result/expression_weights.pt',map_location='cpu',weights_only=True))
counts=torch.bincount(data['train'][1],minlength=7).float();weights=(counts.sum()/counts).sqrt();weights/=weights.mean()
criterion=nn.CrossEntropyLoss(weight=weights);opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.001)
def evaluate(key):
 model.eval();x,y=data[key];p=[]
 with torch.no_grad():
  for i in range(0,len(y),256):p.append(model(x[i:i+256]).softmax(1))
 p=torch.cat(p);pred=p.argmax(1);cm=torch.bincount(y*7+pred,minlength=49).reshape(7,7)
 recall=cm.diag()/cm.sum(1).clamp(min=1)
 return {'accuracy':float((pred==y).float().mean()),'macro_recall':float(recall.mean()),'per_class_recall':recall.tolist(),'confusion_matrix':cm.tolist()},p
initial,_=evaluate('val');best=initial['macro_recall'];state=copy.deepcopy(model.state_dict());best_epoch=0;history=[];stale=0
torch.save(state,OUT/'expression_weights.pt')
for epoch in range(1,7):
 model.train();x,y=data['train'];order=torch.randperm(len(y));total=0
 for ids in order.split(128):
  xb=x[ids].clone();yb=y[ids];flip=torch.rand(len(ids))<.5;xb[flip]=xb[flip].flip(-1)
  xb=(xb*(.85+.3*torch.rand(len(ids),1,1,1))+.06*torch.randn(len(ids),1,1,1)).clamp(0,1)
  opt.zero_grad();loss=criterion(model(xb),yb);loss.backward();opt.step();total+=loss.item()*len(ids)
 metrics,_=evaluate('val');row={'epoch':epoch,'loss':total/len(y),'validation':metrics,'elapsed_seconds':round(time.time()-started)};history.append(row)
 print(json.dumps(row),flush=True);(OUT/'history.json').write_text(json.dumps(history,indent=2))
 if metrics['macro_recall']>best:
  best=metrics['macro_recall'];state=copy.deepcopy(model.state_dict());torch.save(state,OUT/'expression_weights.pt');stale=0;best_epoch=epoch
 else:stale+=1
 if stale>=4:break
model.load_state_dict(state);validation,vp=evaluate('val');test,tp=evaluate('test')
# Select abstention threshold on validation only; do not tune against test.
vy=data['val'][1];threshold=.95;target_met=False
for t in np.arange(.5,.951,.025):
 mask=vp.max(1).values>=t
 if int(mask.sum())>=100 and float((vp.argmax(1)[mask]==vy[mask]).float().mean())>=.8:
  threshold=float(t);target_met=True;break
mask=tp.max(1).values>=threshold
report={'continued_from':'expression_training_result best epoch 11','additional_epochs_run':len(history),'best_epoch':best_epoch,'validation':validation,'test':test,'threshold':threshold,'validation_80pct_target_met':target_met,'test_accepted_count':int(mask.sum()),'test_coverage':float(mask.float().mean()),'test_selective_accuracy':float((tp.argmax(1)[mask]==data['test'][1][mask]).float().mean()) if mask.any() else None,'elapsed_seconds':time.time()-started,'note':'Expression appearance labels only; not psychological diagnosis, liveness, identity. Confidence is not a calibrated probability.'}
(OUT/'metrics.json').write_text(json.dumps(report,indent=2));torch.jit.trace(model.eval(),torch.zeros(1,1,48,48)).save(str(OUT/'expression_model.ts'))
print('COMPLETE '+json.dumps(report),flush=True)
