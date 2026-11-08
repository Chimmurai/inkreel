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
echo   Тести
echo   1  Швидкі: без вікон і без важких ^(експорти, пули, навчання^) - ~1 хв
echo   2  Усе без вікон ^(можна працювати паралельно^)
echo   3  Лише з вікнами ^(GUI^): не торкайся миші й клавіатури
echo   4  Усі підряд ^(довго^)
echo   0  Назад
choice /c 12340 /n /m "  Вибір: "
set K=%errorlevel%
if %K%==5 goto end
if %K%==1 python -m pytest tests -q -m "not gui and not slow"
if %K%==2 python -m pytest tests -q -m "not gui" -o faulthandler_timeout=120
if %K%==3 python -m pytest tests -q -m gui -o faulthandler_timeout=120
if %K%==4 python -m pytest tests -q -o faulthandler_timeout=120
pause
goto menu
:end
call deactivate
pause
