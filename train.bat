@echo off
REM Quick Training Launcher for Windows

echo ============================================================
echo Face Anti-Spoofing - GPU Training Launcher
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
echo ============================================================
echo Select Training Mode:
echo ============================================================
echo 1. Automated Setup (Recommended)
echo 2. Quick Training (30 epochs, ~30 min)
echo 3. Full Training (50 epochs, ~50 min)
echo 4. Test Training (5 epochs, ~5 min)
echo 5. Custom Settings
echo 6. Check GPU Only
echo ============================================================
echo.

set /p choice="Enter your choice (1-6): "

if "%choice%"=="1" (
    echo.
    echo Starting automated setup...
    python start_training.py
    goto end
)

if "%choice%"=="2" (
    echo.
    echo Starting quick training (30 epochs)...
    python train_gpu.py --dataset custom1 --epochs 30 --batch-size 32
    goto end
)

if "%choice%"=="3" (
    echo.
    echo Starting full training (50 epochs)...
    python train_gpu.py --dataset custom1 --epochs 50 --batch-size 32
    goto end
)

if "%choice%"=="4" (
    echo.
    echo Starting test training (5 epochs)...
    python train_gpu.py --dataset custom1 --epochs 5 --batch-size 32
    goto end
)

if "%choice%"=="5" (
    echo.
    echo Custom Training Settings
    echo ============================================================
    set /p dataset="Dataset (custom1/custom2/custom): "
    set /p epochs="Number of epochs: "
    set /p batch="Batch size: "
    echo.
    echo Starting training with custom settings...
    python train_gpu.py --dataset %dataset% --epochs %epochs% --batch-size %batch%
    goto end
)

if "%choice%"=="6" (
    echo.
    echo Checking GPU...
    python -c "import torch; print(f'\nGPU Available: {torch.cuda.is_available()}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
    echo.
    pause
    goto end
)

echo.
echo Invalid choice!
pause

:end
echo.
echo ============================================================
echo Training session ended
echo ============================================================
pause
