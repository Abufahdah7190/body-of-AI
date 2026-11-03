@echo off
cd /d "%~dp0"
if not exist ".objects\Scripts\python.exe" (
 echo Run INSTALL_OBJECTS.cmd first.
 exit /b 1
)
".objects\Scripts\python.exe" object_training.py %*
exit /b %errorlevel%
