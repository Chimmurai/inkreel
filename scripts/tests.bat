@echo off
chcp 65001 >nul
cd /d %~dp0..
if not exist .venv ( echo Спершу створи .venv & pause & exit /b 1 )
call .venv\Scripts\activate.bat
pip install -q pytest
rem   без аргументів - меню; з аргументами - ці аргументи йдуть у pytest як є
if not "%~1"=="" ( python -m pytest tests -q %* & goto end )
:menu
echo.
echo   Тести ^(вікна на Qt перевіряються без екрана: нічого не з'являється^)
echo   1  Швидкі: без важких ^(експорти, пули, навчання^) - ~1 хв
echo   2  Усі підряд ^(довго^)
echo   0  Назад
choice /c 120 /n /m "  Вибір: "
set K=%errorlevel%
if %K%==3 goto end
if %K%==1 python -m pytest tests -q -m "not slow" -o faulthandler_timeout=120
if %K%==2 python -m pytest tests -q -o faulthandler_timeout=120
pause
goto menu
:end
call deactivate
pause
