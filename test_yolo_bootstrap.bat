@echo off
title Test YOLO Bootstrap Detections
cd /d "%~dp0"
echo.
echo ============================================================
echo   Test YOLO Bootstrap Detections
echo ============================================================
echo.
echo This tests the STANDARD YOLO model (the same one used by
echo bootstrap_obb.bat) to verify that panel detections are correct.
echo.
echo This is useful to check BEFORE training the OBB model, so you
echo can verify the training data will be accurate.
echo.
echo ============================================================

:: Check if bootstrap_images exists
if not exist "bootstrap_images" (
    echo.
    echo ERROR: 'bootstrap_images' folder not found!
    echo.
    echo Create this folder and add some manga pages to test.
    echo.
    pause
    exit /b
)

:: Check if model exists
if not exist "models\manga109_yolo.pt" (
    if not exist "models\manga_panels.pt" (
        if not exist "models\manga_panels_finetuned.pt" (
            if not exist "models\yolov8n.pt" (
                echo.
                echo ERROR: No YOLO model found in 'models' folder!
                echo.
                echo Expected one of:
                echo   - manga109_yolo.pt
                echo   - manga_panels.pt
                echo   - manga_panels_finetuned.pt
                echo   - yolov8n.pt
                echo.
                pause
                exit /b
            )
        )
    )
)

echo Press any key to run detection test...
pause > nul

echo.
echo ============================================================
echo Running detection test...
echo ============================================================
echo.

python test_yolo_bootstrap.py

echo.
echo ============================================================
echo Test complete! Check the 'test_yolo_bootstrap_results' folder.
echo ============================================================
echo.
pause
