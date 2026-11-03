import os
from pathlib import Path
root=Path(__file__).resolve().parent
os.chdir(root)
os.environ["YOLO_CONFIG_DIR"]=str(root/".object_config")
from ultralytics import YOLO, settings
settings.update({"sync":False})
YOLO("yolov8n.pt")
print("Object detector is ready. Restart main.py.")
