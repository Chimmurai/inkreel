@echo off
chcp 65001 >nul
cd /d %~dp0..
if not exist .venv_train ( echo Спершу створи .venv_train: InkReel.bat, пункт 2 & pause & exit /b 1 )
call .venv_train\Scripts\activate.bat
rem Одна модель: IRNM1 ^(пресет ir1^). Прогін лежить у weights\irnm1 ^(best.pt, last.pt^); датасет - E:\Inkreel_Dataset.
rem Старт: ваги weights\lite_fq_b ^(або weights\ink1^), якщо irnm1 ще нема; далі повторний запуск продовжує.
:menu
echo.
echo   Навчання IRNM1 ^(єдина модель, з акцентом на штрихи^)
echo   1  Пробне, ~6-10 хв ^(перевіряє відеокарту, дані й штрафи за штрихи^)
echo   2  Донавчити IRNM1 ^(3 год; повторний запуск продовжує^)
echo   3  Додати N хвилин до вже пройденого
echo   4  Метрики ^(штрихи: 0 = як Ultra, 1 = штрихів нема^)
echo   5  Встановити irnm1 у застосунок
echo   6  Швидкість пресетів на відеокарті
echo   0  Назад
choice /c 1234560 /n /m "  Вибір: "
set K=%errorlevel%
if %K%==7 exit /b 0
if %K%==1 python tools\train_ink.py train --smoke --preset ir1 --init-from weights\lite_fq_b --lr 5e-4 --warmup 100
if %K%==2 call :main
if %K%==3 call :more
if %K%==4 call :stats
if %K%==5 call :install
if %K%==6 python tools\train_ink.py bench --preset all --h 1080
pause
goto menu

:main
set SRC=weights\lite_fq_b
if not exist %SRC%\best.pt set SRC=weights\ink1
python tools\train_ink.py train --out weights\irnm1 --preset ir1 --init-from %SRC% --resume --hours 3 --lr 5e-4 --warmup 200 --eval-min 15 --stroke-w 2.0 --stroke-sel 0.5
call :stats
echo.
choice /c YN /n /m "  Встановити irnm1 у застосунок зараз? [Y/N] "
if errorlevel 2 exit /b 0
call :install
exit /b 0

:more
set M=
set /p M= Скільки хвилин додати? Наприклад 60: 
if not defined M exit /b 0
set /a N=%M% 2>nul
if %N% LEQ 0 ( echo Потрібне ціле число більше нуля & exit /b 0 )
python tools\train_ink.py train --out weights\irnm1 --resume --add-min %N%
exit /b 0

:stats
echo === було ^(lite_fq_b^)
python tools\train_ink.py eval --run weights\lite_fq_b
echo === стало ^(irnm1^)
python tools\train_ink.py eval --run weights\irnm1
exit /b 0

:install
if not exist weights\irnm1\best.pt ( echo Нема weights\irnm1\best.pt - спершу пункт 2 & exit /b 0 )
python tools\install_student.py weights\irnm1
echo Далі: меню перевірок, пункт 1.
exit /b 0
