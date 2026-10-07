@echo off
setlocal enabledelayedexpansion

set TF_USE_LEGACY_KERAS=1
set TF_ENABLE_ONEDNN_OPTS=0

:: Check for script
set "PY_SCRIPT="
for %%F in (*.py) do (
    if not "%%F"=="setup.py" (
        set "PY_SCRIPT=%%F"
        goto :found_script
    )
)

:found_script
if "%PY_SCRIPT%"=="" (
    echo [ERROR] No Python file found in this directory!
    pause
    exit /b 1
)

:: If .venv doesn't exist or pandas isn't installed, do a full setup
if not exist ".venv\Scripts\python.exe" (
    goto :setup_venv
)

:: Check if pandas is really installed inside .venv
call .venv\Scripts\python.exe -c "import pandas" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Environment is incomplete or corrupted. Rebuilding...
    rmdir /s /q .venv
    goto :setup_venv
) else (
    goto :run_app
)

:setup_venv
echo [INFO] Creating clean virtual environment...
python -m venv .venv
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment. Ensure Python is installed and in PATH.
    pause
    exit /b 1
)

echo [INFO] Upgrading pip...
call .venv\Scripts\python.exe -m pip install --upgrade pip

echo [INFO] Installing required packages (this may take a few minutes)...
:: Notice the 'call' prefix - mandatory in Windows batch files
call .venv\Scripts\pip.exe install numpy pandas openpyxl scipy scikit-learn matplotlib pillow tf-keras tensorflow

if errorlevel 1 (
    echo [ERROR] Package installation failed!
    pause
    exit /b 1
)

:run_app
echo.
echo [INFO] Pre-flight check...
call .venv\Scripts\python.exe -c "import tkinter, pandas, sklearn, tensorflow; print('[OK] All required packages loaded successfully!')"
if errorlevel 1 (
    echo [ERROR] Libraries failed to load.
    pause
    exit /b 1
)

echo [INFO] Launching %PY_SCRIPT%...
call .venv\Scripts\python.exe "%PY_SCRIPT%" 2> crash_log.txt

if errorlevel 1 (
    echo.
    echo ===================================================
    echo [CRASH DETECTED] Error details:
    echo ===================================================
    type crash_log.txt
    echo ===================================================
)

pause