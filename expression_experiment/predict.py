"""Run the locally trained appearance classifier on one image, not for authentication."""
from pathlib import Path
import argparse,json
import numpy as np
from PIL import Image,ImageOps
import torch
p=argparse.ArgumentParser();p.add_argument('image',help='A tightly cropped frontal face image');a=p.parse_args()
root=Path(__file__).resolve().parent
metrics=json.loads((root/'metrics.json').read_text())
classes=json.loads((root/'data_audit.json').read_text())['classes']
im=ImageOps.grayscale(Image.open(a.image)).resize((48,48),Image.Resampling.BILINEAR)
x=torch.from_numpy(np.asarray(im,dtype=np.float32).copy()[None,None]/255)
model=torch.jit.load(str(root/'expression_model.ts'),map_location='cpu');model.eval()
with torch.no_grad():probs=model(x).softmax(1)[0]
v,i=probs.max(0)
print(json.dumps({'appearance_label':classes[int(i)] if float(v)>=metrics['threshold'] else 'uncertain','raw_score':float(v),'note':'Not a psychological diagnosis or liveness result'},indent=2))
