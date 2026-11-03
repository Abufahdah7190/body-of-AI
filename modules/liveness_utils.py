"""
liveness_utils.py
حساب Eye Aspect Ratio (EAR) الصحيح لكلتا العينين + معدل الرمش (BPM)
EAR أدق من مسافة البكسل الثابتة لأنه نسبة (مقاومة للمسافة عن الكاميرا ودقة الكاميرا)
"""

import numpy as np

# نقاط العين القياسية في MediaPipe FaceMesh (468 نقطة، refine_landmarks=True)
LEFT_EYE = [33, 160, 158, 133, 153, 144]   # [يسار, أعلى1, أعلى2, يمين, أسفل1, أسفل2]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

EAR_BLINK_THRESHOLD = 0.21   # تحت هذا الرقم = العين مغلقة (قيمة معيارية شائعة)
EAR_CONSEC_FRAMES = 2        # عدد الفريمات المتتالية لتأكيد الرمشة (يمنع false positives)


def _landmark_xy(landmark, w, h):
    return np.array([landmark.x * w, landmark.y * h])


def eye_aspect_ratio(landmarks, eye_indices, w, h):
    """
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    p1 = _landmark_xy(landmarks[eye_indices[0]], w, h)
    p2 = _landmark_xy(landmarks[eye_indices[1]], w, h)
    p3 = _landmark_xy(landmarks[eye_indices[2]], w, h)
    p4 = _landmark_xy(landmarks[eye_indices[3]], w, h)
    p5 = _landmark_xy(landmarks[eye_indices[4]], w, h)
    p6 = _landmark_xy(landmarks[eye_indices[5]], w, h)

    vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)

    if horizontal == 0:
        return 0.3  # قيمة افتراضية آمنة تمنع القسمة على صفر

    return vertical / (2.0 * horizontal)


def average_ear(landmarks, w, h):
    """متوسط EAR للعينين معاً — أكثر استقراراً من عين واحدة."""
    left = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
    right = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
    return (left + right) / 2.0


class BlinkTracker:
    """
    يتتبع الرمشات بعدد فريمات متتالية (يمنع اعتبار أي اهتزاز رمشة وهمية)
    ويحسب BPM (Blinks Per Minute) على نافذة زمنية متحركة.
    """

    def __init__(self, ear_threshold=EAR_BLINK_THRESHOLD, consec_frames=EAR_CONSEC_FRAMES):
        self.ear_threshold = ear_threshold
        self.consec_frames = consec_frames
        self._below_count = 0
        self.blink_count = 0
        self._blink_timestamps = []  # لحساب BPM على نافذة متحركة

    def update(self, ear, timestamp):
        blinked_now = False
        if ear < self.ear_threshold:
            self._below_count += 1
        else:
            if self._below_count >= self.consec_frames:
                self.blink_count += 1
                self._blink_timestamps.append(timestamp)
                blinked_now = True
            self._below_count = 0
        return blinked_now

    def blinks_per_minute(self, window_seconds=60, now=None):
        import time
        now = time.monotonic() if now is None else now
        if window_seconds <= 0:
            return 0
        self._blink_timestamps = [t for t in self._blink_timestamps if 0 <= now-t <= window_seconds]
        return round(len(self._blink_timestamps) * 60 / window_seconds)
