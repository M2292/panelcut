@echo off
title Train Segmentation v2 Model (AGGRESSIVE)
cd /d "%~dp0"
echo.
echo ============================================================
echo   Train Segmentation v2 Model - AGGRESSIVE MODE
echo ============================================================
echo.
echo This trains a YOLOv8-seg model with HIGHER learning rate
echo for faster convergence and more aggressive exploration.
echo.
echo DIFFERENCES FROM STANDARD TRAINING:
echo   - 10x higher learning rate (0.01 vs 0.001)
echo   - More aggressive exploration of solution space
echo   - Better for datasets with 100+ images
echo   - May see drastic improvements even in later epochs
echo.
echo IMPORTANT: Unlike OBB, segmentation can detect arbitrary
echo quadrilaterals with each corner positioned independently
echo (not just rotated rectangles).
echo.
echo Training data comes from user corrections when downloading
echo panels with Seg v2 selected in the dropdown.
echo.
echo ============================================================

python train_seg_v2_aggressive.py
