"""Experimental local LBPH candidates. Not identity proof or an access decision."""
from pathlib import Path
from collections import deque
import json
import uuid
import cv2
import numpy as np


def crop_face(frame,box):
    x,y,w,h=box
    if w<80 or h<80:
        return None
    roi=frame[y:y+h,x:x+w]
    if roi.size==0:
        return None
    gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(cv2.resize(gray,(128,128)))


class CandidateHistory:
    def __init__(self):
        self.items=deque(maxlen=5)
    def clear(self):
        self.items.clear()
    def update(self,candidate):
        self.items.append(candidate)
        if len(self.items)==5 and candidate and all(x==candidate for x in self.items):
            return candidate
        return None


class IdentityStore:
    def __init__(self,directory,threshold=45.0):
        self.directory=Path(directory)
        self.directory.mkdir(parents=True,exist_ok=True)
        self.index=self.directory/"profiles.json"
        self.profiles=json.loads(self.index.read_text(encoding="utf-8")) if self.index.exists() else {}
        self.threshold=threshold
        self.model=None
        self.labels={}
        self.available=hasattr(cv2,"face") and hasattr(cv2.face,"LBPHFaceRecognizer_create")
        self.train()

    def save(self):
        temp=self.index.with_suffix(".tmp")
        temp.write_text(json.dumps(self.profiles,ensure_ascii=False,indent=2),encoding="utf-8")
        temp.replace(self.index)

    def train(self):
        self.model=None; self.labels={}
        if not self.available:
            return
        images=[]; labels=[]
        for label,(profile_id,profile) in enumerate(self.profiles.items()):
            # IDs are generated internally; never use profile names as file paths.
            if len(profile_id)!=32 or any(c not in "0123456789abcdef" for c in profile_id):
                raise ValueError("Invalid profile identifier")
            for file in (self.directory/profile_id).glob("*.png"):
                img=cv2.imread(str(file),cv2.IMREAD_GRAYSCALE)
                if img is not None and img.shape==(128,128):
                    images.append(img); labels.append(label)
            self.labels[label]=profile_id
        if images:
            model=cv2.face.LBPHFaceRecognizer_create()
            model.train(images,np.asarray(labels,dtype=np.int32))
            self.model=model

    def enroll(self,name,images):
        if not self.available:
            raise RuntimeError("OpenCV contrib is required")
        name=name.strip()
        if not name or len(name)>60:
            raise ValueError("Name must contain 1 to 60 characters")
        if len(images)<15:
            raise ValueError("At least 15 samples required")
        profile_id=uuid.uuid4().hex
        folder=self.directory/profile_id; folder.mkdir()
        try:
            for n,img in enumerate(images):
                if img.shape!=(128,128):
                    raise ValueError("Invalid sample size")
                if not cv2.imwrite(str(folder/f"{n:02d}.png"),img):
                    raise OSError("Cannot save enrollment sample")
            self.profiles[profile_id]={"name":name,"consent":"local_camera_enrollment"}
            self.save(); self.train()
        except Exception:
            self.profiles.pop(profile_id,None)
            for file in folder.glob("*.png"): file.unlink()
            folder.rmdir()
            self.save(); self.train()
            raise
        return profile_id

    def predict(self,image):
        if self.model is None or image is None:
            return None,None
        label,distance=self.model.predict(image)
        return (self.labels.get(label) if distance < self.threshold else None),float(distance)

    def name(self,profile_id):
        return self.profiles.get(profile_id,{}).get("name","غير معروف")

    def delete(self,profile_id):
        if profile_id not in self.profiles:
            return
        folder=self.directory/profile_id
        for file in folder.glob("*.png"): file.unlink()
        if folder.exists(): folder.rmdir()
        del self.profiles[profile_id]
        self.save(); self.train()
