@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
 echo Create the main .venv first. See README.md.
 pause
 exit /b 1
)
".venv\Scripts\python.exe" -m venv .objects
if errorlevel 1 goto failed
".objects\Scripts\python.exe" -m pip install "ultralytics>=8.3,<9"
if errorlevel 1 goto failed
".objects\Scripts\python.exe" setup_objects.py
if errorlevel 1 goto failed
echo Ready. Run START_CAMERA.cmd.
pause
exit /b 0
:failed
echo Installation failed. Check internet and the error above.
pause
exit /b 1
