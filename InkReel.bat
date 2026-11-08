@echo off
chcp 65001 >nul
cd /d %~dp0
title InkReel
:menu
cls
echo.
echo   ===== InkReel =====
echo.
echo   1  Запустити програму
echo   2  Перше налаштування ^(створити середовища .venv, .venv_train, .venv_dml^)
echo   3  Навчання моделі IRNM1 ^(меню^)
echo   4  Датасет для навчання ^(меню^)
echo   5  Перевірки й заміри швидкості ^(меню^)
echo   6  Тести ^(pytest^)
echo   7  Збірка: exe, інсталятор, пакет Store ^(меню^)
echo   0  Вихід
echo.
choice /c 12345670 /n /m "  Вибір: "
set K=%errorlevel%
if %K%==1 call scripts\run.bat
if %K%==2 call scripts\setup.bat
if %K%==3 call scripts\train.bat
if %K%==4 call scripts\dataset.bat
if %K%==5 call scripts\check.bat
if %K%==6 call scripts\tests.bat
if %K%==7 call scripts\build.bat
if %K%==8 exit /b 0
goto menu
