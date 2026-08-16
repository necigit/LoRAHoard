@echo off
title "Coral LoRA & Checkpoint Puller"
cd /d "%~dp0"

echo ============================================
echo   Coral LoRA ^& Checkpoint Puller
echo   Desktop window will open (no browser needed)
echo   Close the window to quit
echo ============================================
echo.

rem find Python (prefer py launcher, else python)
set PY=python
where py >nul 2>nul && set PY=py -3

%PY% -X utf8 CoralLoRA_gui.py

pause
