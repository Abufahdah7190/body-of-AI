"""Rectangle heuristic only; cannot identify a phone or prove spoofing."""
import cv2
import numpy as np


def detect_rectangular_device(frame):
    """يبحث عن حواف مستطيلة (جوال/تابلت/شاشة) في الفريم كاملاً، دائماً."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        area = cv2.contourArea(c)
        if 6000 < area < 120000:
            approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h) if h > 0 else 0
                # الجوالات والشاشات لها نسبة عرض/ارتفاع مميزة (طولية أو عريضة قليلاً)
                if 0.4 < aspect_ratio < 2.6:
                    return True, (x, y, w, h)
    return False, None

