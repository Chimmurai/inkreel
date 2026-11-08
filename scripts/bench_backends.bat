@echo off
chcp 65001 >nul
cd /d %~dp0..
if not exist reports mkdir reports
rem Порівняння бекендів нейромоделі: DirectML ^(.venv_dml^) проти CUDA і TensorRT ^(.venv_trt, лише NVIDIA^).
rem Нічого в застосунку не змінює; окремі середовища, бо onnxruntime-directml і onnxruntime-gpu взаємовиключні.
:menu
echo.
echo   Порівняння бекендів ^(звіт: reports\backends.md^)
echo   1  Створити .venv_dml і .venv_trt ^(NVIDIA: onnxruntime-gpu + CUDA/cuDNN/TensorRT з pip, кілька ГБ^)
echo   2  Заміряти все ^(DirectML, потім CUDA/TensorRT; TensorRT при першому кадрі будує рушій - хвилини^)
echo   3  Показати таблицю
echo   0  Назад
choice /c 1230 /n /m "  Вибір: "
set K=%errorlevel%
if %K%==4 exit /b 0
if %K%==1 call :make
if %K%==2 call :run
if %K%==3 python tools\bench_backends.py --merge
pause
goto menu

:make
if not exist .venv_dml ( python -m venv .venv_dml || ( echo Потрібен Python 3.10-3.13 & exit /b 1 ) )
.venv_dml\Scripts\python -m pip install --upgrade pip
.venv_dml\Scripts\python -m pip install numpy opencv-python pillow onnxruntime-directml
if not exist .venv_trt ( python -m venv .venv_trt || ( echo Потрібен Python 3.10-3.13 & exit /b 1 ) )
.venv_trt\Scripts\python -m pip install --upgrade pip
.venv_trt\Scripts\python -m pip install numpy opencv-python pillow onnxruntime-gpu
.venv_trt\Scripts\python -m pip install nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cuda-nvrtc-cu12
.venv_trt\Scripts\python -m pip install tensorrt-cu12 tensorrt-cu12-libs
echo Якщо якийсь пакет не став - це не страшно, замір покаже, що з провайдерів доступне.
exit /b 0

:run
if exist .venv_dml\Scripts\python.exe ( .venv_dml\Scripts\python tools\bench_backends.py --label dml ) else ( echo Нема .venv_dml - пункт 1 )
if exist .venv_trt\Scripts\python.exe ( .venv_trt\Scripts\python tools\bench_backends.py --label trt ) else ( echo Нема .venv_trt - пункт 1 )
python tools\bench_backends.py --merge
exit /b 0
