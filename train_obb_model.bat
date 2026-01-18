@echo off
title Train OBB Model
cd /d "%~dp0"
echo.
echo ============================================================
echo   OBB Model Training Tool
echo ============================================================
echo.
echo This will train a YOLOv8-OBB model on your collected data.
echo.
echo ============================================================

:: Check if training data exists
if not exist "training_data\obb\images" (
    echo.
    echo ERROR: No OBB training data found!
    echo.
    echo First, run bootstrap_obb.bat to create training data
    echo or save samples through the web app.
    echo.
    pause
    exit /b
)

:: Count training images
set count=0
for %%f in (training_data\obb\images\*.png training_data\obb\images\*.jpg) do set /a count+=1

echo Found %count% OBB training images.
echo.

if %count% LSS 5 (
    echo ERROR: Need at least 5 training images, currently have %count%.
    echo.
    echo Add more images to 'bootstrap_images' and run bootstrap_obb.bat
    echo.
    pause
    exit /b
)

echo Training Configuration:
echo   - Base model: yolov8n-obb.pt
echo   - Epochs: 50 (adjust in train_obb.py if needed)
echo   - Image size: 640
echo.
echo Press any key to start training...
echo (This may take a while depending on your hardware)
pause > nul

echo.
echo ============================================================
echo Starting training...
echo ============================================================
echo.

:: Run training script
python train_obb.py

echo.
echo ============================================================
if errorlevel 1 (
    echo Training finished with errors. Check messages above.
) else (
    echo Training finished successfully!
)
echo ============================================================
echo.
pause
