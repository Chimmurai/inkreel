@echo off
chcp 65001 >nul
cd /d %~dp0..
if not exist .venv ( echo Спершу створи .venv: InkReel.bat, пункт 2 & pause & exit /b 1 )
call .venv\Scripts\activate.bat
rem Датасет для дистиляції Ultra: E:\Inkreel_Dataset. Зупинка - Ctrl+C, повторний запуск продовжує.
echo.
echo   1  Пробний: 36 зразків і аркуш preview.jpg
echo   2  Повний ^(параметри за замовчуванням^)
echo   3  Повний з обмеженнями: --max-gb і --count
echo   0  Назад
choice /c 1230 /n /m "  Вибір: "
set K=%errorlevel%
if %K%==4 exit /b 0
if %K%==1 ( python tools\make_dataset.py --smoke & if exist E:\Inkreel_Dataset\_smoke\preview.jpg start "" E:\Inkreel_Dataset\_smoke\preview.jpg )
if %K%==2 python tools\make_dataset.py
if %K%==3 call :limits
pause
exit /b 0

:limits
set /p G= Скільки ГБ найбільше ^(напр. 120^): 
set /p C= Скільки зразків ^(напр. 60000^): 
python tools\make_dataset.py --max-gb %G% --count %C%
exit /b 0
