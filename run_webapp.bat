@echo off
REM Launch Face Anti-Spoofing Web Application

echo ============================================================
echo Face Anti-Spoofing Web Application
echo ============================================================
echo.

REM Activate virtual environment
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo ERROR: Virtual environment not found!
    echo Please run: python -m venv venv
    pause
    exit /b 1
)

echo.
echo Starting Flask server...
echo.
echo ============================================================
echo Web Interface will open at: http://localhost:5000
echo ============================================================
echo.
echo Press Ctrl+C to stop the server
echo.

REM Run Flask app
python app.py

pause
