"""Local object dataset preparation, annotation, training and explicit activation."""
from pathlib import Path
import argparse,json,hashlib,shutil,sys,os
ROOT=Path(__file__).resolve().parent
DATA=ROOT/'object_dataset'
SUFFIXES={'.jpg','.jpeg','.png','.bmp'}
def prepare(classes):
    if not 1<=len(classes)<=50 or len(set(classes))!=len(classes):raise ValueError('Use 1..50 unique class names')
    if any(not c or any(ch in c for ch in '\n\r/\\') for c in classes):raise ValueError('Invalid class name')
    meta=DATA/'classes.json'
    if meta.exists() and json.loads(meta.read_text('utf8'))!=classes and (any((DATA/'images').rglob('*.*')) or any((DATA/'labels').rglob('*.txt'))):
        raise ValueError('Existing class order must not change: it defines label IDs. Use a separate project copy for another dataset.')
    for split in ['train','val','test']:
        for part in ['images','labels']:(DATA/part/split).mkdir(parents=True,exist_ok=True)
    meta.write_text(json.dumps(classes,ensure_ascii=False,indent=2),'utf8')
    # JSON strings are valid YAML scalars; keep arbitrary class names quoted.
    text='train: images/train\nval: images/val\ntest: images/test\nnames:\n'
    text+=''.join(f'  {i}: {json.dumps(c,ensure_ascii=False)}\n' for i,c in enumerate(classes))
    (DATA/'data.yaml').write_text(text,'utf8')
    print('Dataset ready:',DATA);print('Class IDs:',dict(enumerate(classes)))
def names():
    p=DATA/'classes.json'
    if not p.exists():raise ValueError('Run prepare first')
    return json.loads(p.read_text('utf8'))
def images(split):return sorted(p for p in (DATA/'images'/split).glob('*') if p.suffix.lower() in SUFFIXES)
def validate():
    from PIL import Image
    classes=names();errors=[];seen={};counts={};manifest=[]
    for split in ['train','val','test']:
        counts[split]=[0]*len(classes);files=images(split)
        if not files:errors.append(f'{split}: no images')
        stems=[p.stem for p in files]
        if len(set(stems))!=len(stems):errors.append(f'{split}: duplicate filename stems')
        for p in files:
            try:
                with Image.open(p) as im:
                    im=im.convert('RGB');digest=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
                if digest in seen:errors.append(f'Duplicate image: {p.name}, already in {seen[digest]}')
                seen[digest]=split+'/'+p.name
            except Exception as exc:errors.append(f'Unreadable image {p.name}: {exc}');continue
            label=DATA/'labels'/split/(p.stem+'.txt')
            if not label.exists():errors.append(f'Missing label: {label}');continue
            present=set()
            for line in label.read_text('utf8').splitlines():
                try:
                    parts=line.split()
                    if len(parts)!=5:raise ValueError()
                    cls=int(parts[0]);x,y,w,h=map(float,parts[1:])
                    if not 0<=cls<len(classes) or not (w>0 and h>0 and 0<=x-w/2<=x+w/2<=1.000001 and 0<=y-h/2<=y+h/2<=1.000001):raise ValueError()
                    present.add(cls)
                except ValueError:errors.append(f'Invalid YOLO box in {label.name}: {line}')
            for cls in present:counts[split][cls]+=1
            manifest.append({'split':split,'file':p.name,'pixel_sha256':digest,'label_sha256':hashlib.sha256(label.read_bytes()).hexdigest()})
        for cls,n in enumerate(counts[split]):
            minimum=10 if split=='train' else 3
            if n<minimum:errors.append(f'{split}: {classes[cls]} has {n} labeled images; minimum check is {minimum}, not proof of sufficient diversity')
    report={'classes':classes,'counts':counts,'errors':errors,'files':manifest,'note':'Exact-pixel dedup only; near duplicates and same object instance/session must be split manually.'}
    (DATA/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf8')
    if errors:raise ValueError('\n'.join(errors[:30]))
    print('Data checks passed. This does not prove annotation quality or generalization.');return report
def label(split,relabel=False):
    import cv2
    classes=names()
    for p in images(split):
        out=DATA/'labels'/split/(p.stem+'.txt')
        if out.exists() and not relabel:continue
        im=cv2.imread(str(p))
        if im is None:print('Cannot read',p);continue
        h,w=im.shape[:2];scale=min(1,1000/w,700/h);view=cv2.resize(im,(round(w*scale),round(h*scale)));vh,vw=view.shape[:2]
        rows=[]
        while True:
            print('\nImage:',p.name,'Classes:',dict(enumerate(classes)))
            answer=input('Class ID to draw; Enter = finish this image; q = quit: ').strip()
            if answer.lower()=='q':cv2.destroyAllWindows();return
            if not answer:
                if not rows and input('No supported objects in this image? Type yes to save as negative: ').strip()!='yes':continue
                out.write_text('\n'.join(rows)+'\n' if rows else '',encoding='utf8');break
            try:
                cls=int(answer)
                if not 0<=cls<len(classes):raise ValueError()
            except ValueError:print('Invalid class ID');continue
            x,y,bw,bh=cv2.selectROI('Draw object box, Enter confirms, C cancels',view,False,False)
            cv2.destroyAllWindows()
            if bw and bh:rows.append(f'{cls} {(x+bw/2)/vw:.8f} {(y+bh/2)/vh:.8f} {bw/vw:.8f} {bh/vh:.8f}')
    print('Annotation finished:',split)
def configure():
    folder=ROOT/'.object_config';folder.mkdir(exist_ok=True);os.environ['YOLO_CONFIG_DIR']=str(folder)
def train(epochs):
    report=validate();base=ROOT/'yolov8n.pt'
    if not base.exists():raise ValueError('Run INSTALL_OBJECTS.cmd first')
    configure()
    from ultralytics import YOLO,settings
    settings.update({'sync':False})
    model=YOLO(str(base))
    model.train(data=str(DATA/'data.yaml'),epochs=epochs,imgsz=416,batch=2,device='cpu',workers=0,cache=False,amp=False,project=str(ROOT/'object_training_runs'),name='custom',exist_ok=False,seed=42,deterministic=True,plots=False)
    run=Path(model.trainer.save_dir);best=run/'weights/best.pt'
    (run/'dataset_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf8')
    # Independent test split, never used for choosing the best validation epoch.
    result=YOLO(str(best)).val(data=str(DATA/'data.yaml'),split='test',imgsz=416,batch=1,device='cpu',workers=0,plots=False,project=str(run),name='held_out_test')
    metrics={k:float(v) for k,v in result.results_dict.items()}
    metrics.update({'classes':report['classes'],'note':'Review per-class results and live-camera errors before activation. Not PAD training.'})
    (run/'test_metrics.json').write_text(json.dumps(metrics,indent=2),'utf8')
    print('Training complete. Not activated. Best weights:',best);print('Test results:',run/'test_metrics.json')
def activate(path):
    path=Path(path).resolve()
    if not path.is_file() or path.suffix!='.pt':raise ValueError('Select your trained best.pt')
    if not (path.parent.parent/'test_metrics.json').exists():raise ValueError('Training/test report missing; use a completed run from this tool')
    configure()
    from ultralytics import YOLO
    model=YOLO(str(path));class_names=[str(model.names[i]) for i in range(len(model.names))]
    if class_names!=names():raise ValueError('Model classes do not match dataset classes')
    dest=ROOT/'custom_objects';dest.mkdir(exist_ok=True)
    tmp=dest/'model.pending';shutil.copyfile(path,tmp);tmp.replace(dest/'model.pt')
    meta={'classes':class_names,'sha256':hashlib.sha256((dest/'model.pt').read_bytes()).hexdigest(),'source_run':str(path),'trained_by_user':True}
    tmp=dest/'manifest.pending';tmp.write_text(json.dumps(meta,ensure_ascii=False,indent=2),'utf8');tmp.replace(dest/'manifest.json')
    print('Custom model activated. Restart camera. Base detector remains active too.')
def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='cmd',required=True)
    p=sub.add_parser('prepare');p.add_argument('--classes',nargs='+',default=['wallet','key','cup','cell phone'])
    p=sub.add_parser('label');p.add_argument('--split',choices=['train','val','test'],required=True);p.add_argument('--relabel',action='store_true')
    sub.add_parser('check');p=sub.add_parser('train');p.add_argument('--epochs',type=int,default=30)
    p=sub.add_parser('activate');p.add_argument('--weights',required=True)
    args=parser.parse_args()
    try:
        if args.cmd=='prepare':prepare(args.classes)
        elif args.cmd=='label':label(args.split,args.relabel)
        elif args.cmd=='check':validate()
        elif args.cmd=='train':
            if not 1<=args.epochs<=300:raise ValueError('epochs must be 1..300')
            train(args.epochs)
        elif args.cmd=='activate':activate(args.weights)
    except Exception as exc:print('STOP:',str(exc));return 1
    return 0
if __name__=='__main__':sys.exit(main())
