@echo off
title Student Performance Prediction System
color 0A
echo.
echo  ============================================================
echo    Student Performance Prediction System - FDS Project
echo  ============================================================
echo.
echo  [1] Run Full Pipeline  (main.py)
echo  [2] Generate HTML Report  (generate_report.py)
echo  [3] Interactive Predict Only  (src/predict.py)
echo  [4] Run EDA Only  (src/eda.py)
echo  [5] Exit
echo.
set /p choice=  Enter choice (1-5): 

if "%choice%"=="1" (
    echo.
    echo  Running full pipeline...
    python main.py
    echo.
    echo  Done! Run option 2 to generate an HTML report.
    pause
)
if "%choice%"=="2" (
    echo.
    echo  Generating HTML Report...
    python generate_report.py
    echo.
    echo  Opening report...
    start outputs\report.html
    pause
)
if "%choice%"=="3" (
    echo.
    python src\predict.py
    pause
)
if "%choice%"=="4" (
    echo.
    python src\eda.py
    pause
)
if "%choice%"=="5" exit

run.bat
