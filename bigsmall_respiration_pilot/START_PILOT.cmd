@echo off
setlocal
cd /d "%~dp0"
set "RESP_PY="
if exist ".respiration\Scripts\python.exe" set "RESP_PY=%~dp0.respiration\Scripts\python.exe"
if not defined RESP_PY if exist ".objects\Scripts\python.exe" set "RESP_PY=%~dp0.objects\Scripts\python.exe"
if not defined RESP_PY if exist "..\.objects\Scripts\python.exe" set "RESP_PY=%~dp0..\.objects\Scripts\python.exe"
if not defined RESP_PY (
 echo No compatible environment found. Run SETUP_PILOT.cmd once.
 pause
 exit /b 1
)
"%RESP_PY%" -c "import torch, cv2, scipy, numpy" >nul 2>&1
if errorlevel 1 (
 echo Missing pilot dependencies. Run SETUP_PILOT.cmd to create an isolated environment.
 pause
 exit /b 1
)
"%RESP_PY%" pilot.py %*
pause
