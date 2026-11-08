@echo off
chcp 65001 >nul
cd /d %~dp0..
if not exist .venv ( echo Спершу перше налаштування: пункт 2 у InkReel.bat & pause & exit /b 1 )
call .venv\Scripts\activate.bat
python -m benvid
if errorlevel 1 pause
