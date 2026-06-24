@echo off
REM Daily PEAD push: scan for new top-quintile earnings beats and push BUYs to phone.
REM Quiet between earnings seasons; each beat alerts once (de-duped). Run weekday evenings.
cd /d "%~dp0"
set PY="C:\Users\xxdis\AppData\Local\Python\pythoncore-3.14-64\python.exe"
echo ===== PEAD daily run %DATE% %TIME% ===== >> "pead_notify.log"
%PY% pead_notify.py --universe full >> "pead_notify.log" 2>&1
echo ===== Done %TIME% ===== >> "pead_notify.log"
