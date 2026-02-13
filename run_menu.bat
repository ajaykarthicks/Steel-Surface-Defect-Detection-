@echo off
setlocal

set "ROOT=%~dp0"
set "VENV_PY=%ROOT%.venv\Scripts\python.exe"

if exist "%VENV_PY%" (
    set "PY=%VENV_PY%"
) else (
    set "PY=python"
)

echo ===============================
echo Project neu - Launcher
echo ===============================
echo 1) Detect Defect GUI
echo 2) Train GUI
echo 3) Benchmark
echo 4) Setup
echo Q) Quit
echo.
set /p "CHOICE=Select an option: "

if /I "%CHOICE%"=="1" goto detect
if /I "%CHOICE%"=="2" goto train
if /I "%CHOICE%"=="3" goto benchmark
if /I "%CHOICE%"=="4" goto setup
if /I "%CHOICE%"=="Q" goto end

echo Invalid selection.
goto end

:detect
pushd "%ROOT%"
"%PY%" defect_detector_gui.py
popd
goto end

:train
pushd "%ROOT%"
"%PY%" train_gui.py
popd
goto end

:benchmark
pushd "%ROOT%"
"%PY%" benchmark_all.py
popd
goto end

:setup
pushd "%ROOT%"
"%PY%" setup_project.py
popd
goto end

:end
endlocal
