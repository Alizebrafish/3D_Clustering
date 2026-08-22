@echo off
cd /d "%~dp0"
title F3DCA New Background Fixed

if not exist f3dca_env (
    python -m venv f3dca_env
)

call f3dca_env\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

python "F3DCA.py"
pause
