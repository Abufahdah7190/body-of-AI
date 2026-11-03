"""Run general and charger-specific YOLO models in an isolated process."""

import contextlib
import hashlib
import json
import os
import struct
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
(root / ".object_config").mkdir(exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(root / ".object_config")


def read_exact(stream, size):
    result = bytearray()
    while len(result) < size:
        chunk = stream.read(size - len(result))
        if not chunk:
            return None
        result.extend(chunk)
    return bytes(result)


def box_iou(first, second):
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0


def charger_predictions(model, image):
    """Detect at full frame and on a magnified center crop for small cable ends."""
    height, width = image.shape[:2]
    crop_left = int(width * 0.10)
    crop_top = int(height * 0.05)
    crop_right = int(width * 0.90)
    crop_bottom = int(height * 0.95)
    crop = image[crop_top:crop_bottom, crop_left:crop_right]

    views = [
        (image, 0, 0, width, height),
        (crop, crop_left, crop_top, crop_right, crop_bottom),
    ]
    candidates = []
    for view, offset_x, offset_y, view_right, view_bottom in views:
        if view.size == 0:
            continue

        result = model.predict(
            view,
            imgsz=640,
            conf=0.20,
            iou=0.55,
            verbose=False,
            device="cpu",
        )[0]
        scale_x = (view_right - offset_x) / view.shape[1]
        scale_y = (view_bottom - offset_y) / view.shape[0]

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            mapped_box = [
                (x1 * scale_x + offset_x) / width,
                (y1 * scale_y + offset_y) / height,
                (x2 * scale_x + offset_x) / width,
                (y2 * scale_y + offset_y) / height,
            ]
            candidates.append({
                "name": str(result.names[int(box.cls.item())]),
                "confidence": float(box.conf.item()),
                "box": mapped_box,
                "custom": True,
            })

    # The same cable may be found in both views. Keep its strongest box only.
    candidates.sort(key=lambda row: row["confidence"], reverse=True)
    unique = []
    for candidate in candidates:
        duplicate = any(
            candidate["name"] == kept["name"]
            and box_iou(candidate["box"], kept["box"]) >= 0.45
            for kept in unique
        )
        if not duplicate:
            unique.append(candidate)
    return unique


def main():
    with contextlib.redirect_stdout(sys.stderr):
        import cv2
        import numpy as np
        import torch
        from ultralytics import YOLO, settings

        torch.set_num_threads(2)
        settings.update({"sync": False})
        general_model = YOLO(str(root / "yolov8n.pt"))
        charger_model = None
        manifest = root / "custom_objects" / "manifest.json"

        if manifest.exists():
            metadata = json.loads(manifest.read_text("utf8"))
            weights = root / "custom_objects" / "model.pt"
            digest = hashlib.sha256(weights.read_bytes()).hexdigest()
            if digest != metadata["sha256"]:
                raise ValueError("Custom model checksum mismatch")
            charger_model = YOLO(str(weights))

    while True:
        header = read_exact(sys.stdin.buffer, 4)
        if header is None:
            return

        size = struct.unpack("<I", header)[0]
        if size > 10_000_000:
            return

        payload = read_exact(sys.stdin.buffer, size)
        if payload is None:
            return

        image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            print(json.dumps({"objects": [], "detections": []}), flush=True)
            continue

        with contextlib.redirect_stdout(sys.stderr):
            general_result = general_model.predict(
                image,
                imgsz=640,
                conf=0.25,
                verbose=False,
                device="cpu",
            )[0]

        height, width = image.shape[:2]
        detections = []
        for box in general_result.boxes:
            coordinates = box.xyxy[0].tolist()
            normalized = [
                coordinates[0] / width,
                coordinates[1] / height,
                coordinates[2] / width,
                coordinates[3] / height,
            ]
            detections.append({
                "name": str(general_result.names[int(box.cls.item())]),
                "confidence": float(box.conf.item()),
                "box": normalized,
            })

        if charger_model is not None:
            with contextlib.redirect_stdout(sys.stderr):
                detections.extend(charger_predictions(charger_model, image))

        names = sorted({row["name"] for row in detections})
        print(json.dumps({"objects": names, "detections": detections}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"error": str(exc)}), flush=True)
