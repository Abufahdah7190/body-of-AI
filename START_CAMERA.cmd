@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
 echo Run SETUP_ALL.cmd first.
 pause
 exit /b 1
)
".venv\Scripts\python.exe" main.py
if errorlevel 1 pause
