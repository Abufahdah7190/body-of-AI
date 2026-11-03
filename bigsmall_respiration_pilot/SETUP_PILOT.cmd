@echo off
setlocal
cd /d "%~dp0"
py -3.10 -m venv .respiration
if errorlevel 1 goto failed
".respiration\Scripts\python.exe" -m pip install -r requirements-pilot.txt
if errorlevel 1 goto failed
echo Ready. Run START_PILOT.cmd
pause
exit /b 0
:failed
echo Setup failed. Keep the error text for diagnosis. Existing project environments were not changed.
pause
exit /b 1
