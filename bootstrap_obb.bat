@echo off
title Bootstrap OBB Training Data
cd /d "%~dp0"

echo.
echo ============================================================
echo   OBB Training Data Bootstrap Tool
echo ============================================================
echo.
echo This will process images in 'bootstrap_images' folder
echo (including all subfolders) and create OBB training data.
echo.
echo ============================================================

:: Check if bootstrap_images folder exists
if not exist "bootstrap_images" (
    echo Creating 'bootstrap_images' folder...
    mkdir bootstrap_images
    echo.
    echo Please place your manga page images in the 'bootstrap_images' folder
    echo then run this script again.
    echo.
    pause
    exit /b
)

echo.
echo Starting Python script...
echo (This window will stay open so you can see progress and any errors)
echo.
echo ============================================================

:: Run the Python script
python bootstrap_obb.py

:: Always pause at the end so user can see output
echo.
echo ============================================================
if errorlevel 1 (
    echo Script finished with errors. Check messages above.
) else (
    echo Script finished successfully!
)
echo ============================================================
echo.
pause
