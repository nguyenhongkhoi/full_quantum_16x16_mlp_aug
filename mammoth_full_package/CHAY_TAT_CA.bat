@echo off
rem Double-click or run from cmd: setup (first time only) + the whole Mammoth queue.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_all.ps1"
pause
