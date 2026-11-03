@echo off
setlocal
cd /d "%~dp0"
echo [1/5] Checking Python 3.10...
py -3.10 -c "import sys; assert sys.version_info[:2] == (3,10)"
if errorlevel 1 (
 echo Install Python 3.10 x64 with Python Launcher, then run this file again.
 goto failed
)
echo [2/5] Preparing main camera environment...
if not exist ".venv\Scripts\python.exe" py -3.10 -m venv .venv
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo [3/5] Preparing object detector environment...
if not exist ".objects\Scripts\python.exe" py -3.10 -m venv .objects
if errorlevel 1 goto failed
".objects\Scripts\python.exe" -m pip install -r requirements-objects.txt
if errorlevel 1 goto failed
echo [4/5] Preparing object weights...
".objects\Scripts\python.exe" setup_objects.py
if errorlevel 1 goto failed
echo [5/5] Checking installed packages...
".venv\Scripts\python.exe" -c "import cv2,mediapipe,numpy,PIL,sounddevice,tkinter; assert hasattr(cv2,'face'); assert hasattr(mediapipe,'solutions')"
if errorlevel 1 goto failed
".objects\Scripts\python.exe" -c "import torch,scipy,cv2,numpy,ultralytics"
if errorlevel 1 goto failed
echo Setup complete. Run START_CAMERA.cmd.
pause
exit /b 0
:failed
echo Setup stopped. Keep the error text. You can run SETUP_ALL.cmd again to resume.
pause
exit /b 1
