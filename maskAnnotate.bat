@echo off
REM Double-click this file to launch maskAnnotate

cd /d "%~dp0"

call conda activate fly2p

python run_gui.py
