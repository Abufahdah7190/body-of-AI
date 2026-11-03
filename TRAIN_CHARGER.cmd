@echo off
cd /d "%~dp0"
if not exist ".objects\Scripts\python.exe" (
  echo Run SETUP_ALL.cmd first to install the object detector.
  pause
  exit /b 1
)
".objects\Scripts\python.exe" train_charger_model.py
pause
