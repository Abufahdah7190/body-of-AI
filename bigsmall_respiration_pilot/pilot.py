"""Standalone BigSmall pilot. Never used for identity, PAD, or medical decisions."""
from pathlib import Path
import argparse,hashlib,json,time,math
import cv2
import numpy as np
import torch
from scipy.signal import butter,sosfiltfilt,detrend,periodogram
from BigSmall import BigSmall
ROOT=Path(__file__).resolve().parent
SHA='8789820ed61519337f86e5ae44abc2e099c66a3992fd5c103a995dd397a8143a'
FS=25

def prepare(frames):
    a=np.asarray(frames,dtype=np.float32)
    if a.ndim!=4 or a.shape[1:]!=(144,144,3) or len(a)<3 or not np.isfinite(a).all():
        raise ValueError('Invalid RGB input')
    sd=float(a.std())
    big=(a-a.mean())/max(sd,1e-7)
    diff=(a[1:]-a[:-1])/(a[1:]+a[:-1]+1e-7)
    ds=float(diff.std())
    diff=diff/max(ds,1e-7)
    diff=np.concatenate([diff,np.zeros_like(diff[:1])])
    small=np.stack([cv2.resize(f,(9,9),interpolation=cv2.INTER_AREA) for f in diff])
    n=len(a)//3*3
    return np.ascontiguousarray(big[:n].transpose(0,3,1,2)),np.ascontiguousarray(small[:n].transpose(0,3,1,2)),ds

def load_model():
    path=ROOT/'BP4D_BigSmall_Multitask_Fold1.pth'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=SHA:raise ValueError('Weight checksum mismatch')
    state=torch.load(path,map_location='cpu',weights_only=True)
    model=BigSmall();model.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True)
    return model.eval()

def summarize(resp,fs=FS):
    unknown={'status':'insufficient_signal','experimental_rate_per_minute':None}
    a=np.asarray(resp,dtype=np.float64).reshape(-1)
    if len(a)<fs*25 or not np.isfinite(a).all() or a.std()<1e-7:return unknown
    # Upstream respiration label is DiffNormalized: reconstruct by cumulative sum.
    wave=detrend(np.cumsum(a))
    wave=sosfiltfilt(butter(2,[.1,.5],btype='bandpass',fs=fs,output='sos'),wave)
    if wave.std()<1e-7:return unknown
    f,p=periodogram(wave,fs,window='hann');band=(f>=.1)&(f<=.5)
    ids=np.flatnonzero(band);k=ids[np.argmax(p[band])]
    ratio=float(p[max(0,k-1):k+2].sum()/max(p[(f>=.05)&(f<=1)].sum(),1e-12))
    if ratio<.65 or k==ids[0] or k==ids[-1]:return unknown
    # Reject large disagreement between halves. These thresholds are engineering gates, not validated confidence.
    rates=[]
    for half in np.array_split(wave,2):
        ff,pp=periodogram(half,fs,window='hann');sel=(ff>=.1)&(ff<=.5)
        rates.append(float(ff[sel][np.argmax(pp[sel])]*60))
    if abs(rates[0]-rates[1])>6:return unknown
    return {'status':'experimental_unvalidated','experimental_rate_per_minute':round(float(f[k]*60),1),'spectral_concentration':round(ratio,3),'half_window_rates':rates}

def analyze(frames,model):
    big,small,variation=prepare(frames)
    if variation<1e-5:return {'status':'static_input','experimental_rate_per_minute':None}
    values=[]
    with torch.inference_mode():
        for i in range(0,len(big),30):
            _,_,resp=model([torch.from_numpy(big[i:i+30]),torch.from_numpy(small[i:i+30])])
            values.extend(resp[:,0].cpu().numpy().tolist())
    return summarize(values)


def timing_stats(times):
    a=np.asarray(times,dtype=float)
    if len(a)<2 or not np.isfinite(a).all() or np.any(np.diff(a)<=0):
        return 0.0,float('inf')
    return float((len(a)-1)/(a[-1]-a[0])),float(np.diff(a).max())

def timing_ok(times):
    rate,gap=timing_stats(times)
    return rate>=20 and gap<=.15

def preview(frame,line):
    view=cv2.resize(frame,(640,480))
    cv2.putText(view,line,(12,25),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
    cv2.imshow('BigSmall pilot - no video saved',view)
    if cv2.waitKey(1)&255==27:raise RuntimeError('Cancelled')

def choose_camera(camera):
    trials=[]
    # Try lower resolutions and another Windows backend; never lower the signal timing gate.
    for backend,w,h in [(cv2.CAP_DSHOW,640,480),(cv2.CAP_DSHOW,320,240),(cv2.CAP_MSMF,640,480)]:
        cap=cv2.VideoCapture(camera,backend)
        keep=False
        try:
            if not cap.isOpened():continue
            cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,w);cap.set(cv2.CAP_PROP_FRAME_HEIGHT,h);cap.set(cv2.CAP_PROP_FPS,30)
            began=time.perf_counter();times=[];last_preview=-1
            while time.perf_counter()-began<4:
                ok,frame=cap.read()
                if not ok:break
                now=time.perf_counter();elapsed=now-began
                if elapsed>=1:times.append(now)
                if now-last_preview>=.15:
                    rate,_=timing_stats(times)
                    preview(frame,f'Camera check {len(trials)+1}/3 | {w}x{h} | {rate:.1f} FPS | Esc cancels')
                    last_preview=now
            rate,gap=timing_stats(times);trials.append({'backend':backend,'requested_size':[w,h],'fps':round(rate,2),'max_gap':round(gap,3) if math.isfinite(gap) else None})
            print('Camera check: '+json.dumps(trials[-1]),flush=True)
            if len(times)>=40 and timing_ok(times):
                keep=True;return cap
        finally:
            if not keep:cap.release()
    raise RuntimeError('Camera check failed before recording. Need >=20 FPS with gaps <=0.15s. Attempts: '+json.dumps(trials))

def acquire(video=None,camera=0):
    cap=cv2.VideoCapture(str(video)) if video else choose_camera(camera)
    if not cap.isOpened():raise RuntimeError('Cannot open video/camera')
    fps=cap.get(cv2.CAP_PROP_FPS)
    if video and (not math.isfinite(fps) or fps<20):cap.release();raise ValueError('Video must have a known frame rate >=20 FPS')
    detector=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    if detector.empty():cap.release();raise RuntimeError('Face detector unavailable')
    frames=[];timestamps=[];valid=0;checks=0;index=0;start=None;previous=None;changes=[];next_keep=0.0;face_samples=[];last_preview=-1;raw_times=[]
    try:
        while True:
            ok,bgr=cap.read()
            if not ok:break
            stamp=time.perf_counter()
            if start is None:start=stamp
            t=index/fps if video else stamp-start
            if t>=30:break
            raw_times.append(t)
            if not video and t>=5 and not timing_ok(raw_times[-120:]):
                rate,gap=timing_stats(raw_times[-120:])
                raise RuntimeError(f'Capture stopped early: {rate:.1f} FPS, gap {gap:.3f}s. No breathing result.')
            if index%10==0:
                # Defer expensive face detection until capture ends.
                face_samples.append(cv2.cvtColor(cv2.resize(bgr,(320,240)),cv2.COLOR_BGR2GRAY))
            rgb=cv2.cvtColor(cv2.resize(bgr,(144,144),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
            # Retain at most 25 frames/sec to bound memory. Actual capture timestamps are preserved.
            if t+1e-6>=next_keep:
                next_keep=(math.floor((t+1e-6)*FS)+1)/FS
                frames.append(rgb);timestamps.append(t)
                if previous is not None:changes.append(float(np.abs(rgb.astype(float)-previous).mean()))
                previous=rgb.astype(float)
            if not video and t-last_preview>=.1:
                rate,_=timing_stats(raw_times)
                preview(bgr,f'{int(t)}/30s | Camera {rate:.1f} FPS | Esc cancels')
                last_preview=t
            index+=1
    finally:
        cap.release();cv2.destroyAllWindows()
    if len(timestamps)<2 or timestamps[-1]-timestamps[0]<28:raise ValueError('Need 30 seconds of uninterrupted video')
    dt=np.diff(timestamps)
    rate=(len(timestamps)-1)/(timestamps[-1]-timestamps[0])
    print(f'Capture: {rate:.1f} FPS, largest gap {dt.max():.3f}s',flush=True)
    if dt.max()>.15 or rate<20:raise ValueError(f'Frame timing too irregular or slow: {rate:.1f} FPS; largest gap {dt.max():.3f}s; requires >=20 FPS and gaps <=0.15s')
    print('Capture complete. Checking face visibility and analyzing; please wait...',flush=True)
    for gray in face_samples:
        faces=detector.detectMultiScale(gray,1.1,5,minSize=(35,35));checks+=1
        if len(faces)==1 and faces[0][2]>=gray.shape[1]*.15 and 35<float(gray.mean())<220:valid+=1
    if valid/max(checks,1)<.9:raise ValueError('One clear frontal face required throughout recording')
    if changes and np.percentile(changes,90)>12:raise ValueError('Excessive image changes: stay still with stable lighting')
    # Nearest-frame resampling is an experimental adaptation; variable-frame-rate files are not validated.
    ts=np.asarray(timestamps);target=np.arange(ts[0],ts[-1],1/FS)
    right=np.searchsorted(ts,target).clip(1,len(ts)-1);left=right-1
    ids=np.where(abs(ts[right]-target)<abs(ts[left]-target),right,left)
    return np.asarray(frames)[ids]

def main():
    p=argparse.ArgumentParser();p.add_argument('--video',type=Path);p.add_argument('--camera',type=int,default=0);p.add_argument('--report',type=Path);args=p.parse_args()
    torch.set_num_threads(2)
    print('Research pilot. Not a medical measurement or liveness proof. No images/audio/video saved.')
    print('Keep one frontal face large in the frame, remain still, use stable lighting. Capture: 30 seconds.')
    try:
        model=load_model();frames=acquire(args.video,args.camera);result=analyze(frames,model)
    except (ValueError,RuntimeError,OSError,AttributeError,cv2.error) as e:result={'status':'unavailable','experimental_rate_per_minute':None,'reason':str(e)}
    result['not_validated_against_reference_sensor']=True
    cv2.destroyAllWindows()
    print(json.dumps(result,indent=2))
    if not args.video:
        try:
            import tkinter as tk
            from tkinter import messagebox
            root=tk.Tk();root.withdraw()
            rate=result.get('experimental_rate_per_minute')
            message=(f"التقدير التجريبي: {rate} في الدقيقة\nغير متحقق منه طبيًا، ولا يثبت الحيوية." if rate is not None else "لم يُحسب معدل تنفس.\n"+result.get('reason',result.get('status','')))
            messagebox.showinfo('نتيجة تجربة التنفس',message,parent=root);root.destroy()
        except Exception:
            pass  # Console result remains available even without a GUI.

    if args.report:args.report.write_text(json.dumps(result,indent=2),encoding='utf8')
if __name__=='__main__':main()
