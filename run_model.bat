@echo off
REM Quick launcher for trained model

echo ============================================================
echo Face Anti-Spoofing - Trained Model Launcher
echo ============================================================
echo.
echo Your model: 75.27%% accuracy
echo Location: models\best_model.pth
echo.

REM Activate virtual environment
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo ERROR: Virtual environment not found!
    pause
    exit /b 1
)

echo ============================================================
echo Select Mode:
echo ============================================================
echo 1. Webcam Test (Real-time)
echo 2. Test on Image
echo 3. Check Model Info
echo 4. Evaluate Model
echo 5. Exit
echo ============================================================
echo.

set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" (
    echo.
    echo Starting webcam test...
    echo Press 'q' to quit
    echo.
    python test_trained_model.py --model models/best_model.pth --mode webcam
    goto end
)

if "%choice%"=="2" (
    echo.
    set /p image="Enter image path: "
    echo.
    echo Testing image...
    python test_trained_model.py --model models/best_model.pth --mode image --input "%image%"
    goto end
)

if "%choice%"=="3" (
    echo.
    echo Checking model information...
    python check_model.py
    goto end
)

if "%choice%"=="4" (
    echo.
    echo Evaluating model...
    python src/evaluate.py --model models/best_model.pth --dataset custom1
    goto end
)

if "%choice%"=="5" (
    echo.
    echo Goodbye!
    goto end
)

echo.
echo Invalid choice!

:end
echo.
pause
