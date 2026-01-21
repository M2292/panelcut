@echo off
title Train Segmentation v2 Model
cd /d "%~dp0"
echo.
echo ============================================================
echo   Train Segmentation v2 Model
echo ============================================================
echo.
echo This trains a YOLOv8-seg model for manga panel detection.
echo.
echo IMPORTANT: Unlike OBB, segmentation can detect arbitrary
echo quadrilaterals with each corner positioned independently
echo (not just rotated rectangles).
echo.
echo Training data comes from user corrections when downloading
echo panels with Seg v2 selected in the dropdown.
echo.
echo ============================================================

python train_seg_v2.py
