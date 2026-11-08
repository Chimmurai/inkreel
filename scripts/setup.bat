@echo off
chcp 65001 >nul
cd /d %~dp0..
rem Створює середовища, яких ще немає. Повторний запуск безпечний, готове не чіпає.
rem   .venv        застосунок, тести, збірка
rem   .venv_train  PyTorch з CUDA для навчання (до 3 ГБ)
rem   .venv_dml    лише onnxruntime-directml для чесних замірів на відеокарті
:menu
echo.
echo   Яке середовище створити?
echo   1  .venv ^(застосунок^)       2  .venv_train ^(навчання^)       3  .venv_dml ^(заміри^)       4  усі три       0  назад
choice /c 12340 /n /m "  Вибір: "
set K=%errorlevel%
if %K%==5 exit /b 0
if %K%==1 call :app
if %K%==2 call :train
if %K%==3 call :dml
if %K%==4 ( call :app & call :train & call :dml )
pause
exit /b 0

:app
if not exist .venv ( python -m venv .venv || ( echo Потрібен Python 3.10-3.13 & exit /b 1 ) )
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
call deactivate
exit /b 0

:dml
if not exist .venv_dml ( python -m venv .venv_dml )
.venv_dml\Scripts\python -m pip install onnxruntime-directml numpy
exit /b 0

:train
echo.
echo === Відеокарта й драйвер NVIDIA
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>nul
if errorlevel 1 echo nvidia-smi не знайдено - встанови драйвер NVIDIA ^(nvidia.com/drivers^) і запусти знову.
if not exist .venv_train ( python -m venv .venv_train || ( echo Потрібен Python 3.10-3.13 & exit /b 1 ) )
call .venv_train\Scripts\activate.bat
python -m pip install --upgrade pip
pip install numpy opencv-python pillow onnx onnxscript onnxruntime onnxconverter-common
set OKCUDA=
python -c "import torch" 2>nul
if not errorlevel 1 python tools\train_ink.py cuda && set OKCUDA=ok
rem збірки PyTorch за спаданням: якщо нова CUDA не працює з драйвером, пробуємо старшу
for %%C in (cu128 cu126 cu124) do if not defined OKCUDA (
  echo.
  echo === Ставлю PyTorch для CUDA %%C ^(до 3 ГБ, кілька хвилин^)
  pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/%%C
  if not errorlevel 1 python tools\train_ink.py cuda && set OKCUDA=%%C
)
if not defined OKCUDA echo Не вдалося отримати PyTorch, що бачить відеокарту. Онови драйвер NVIDIA.
call deactivate
exit /b 0
