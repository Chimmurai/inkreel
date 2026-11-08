@echo off
chcp 65001 >nul
cd /d %~dp0..
rem Фіксує версії бібліотек, на яких програму перевірено: requirements.lock.txt. build_exe.bat збирає саме з них, тож випуск не зміниться через нову версію
rem numpy, OpenCV чи numba (картинка залежить від їхньої арифметики). Запускати, коли середовище .venv перевірене: тести пройшли, програма працює.
rem Файл зберегти разом із вихідним кодом випуску; щоб оновити бібліотеки свідомо - видалити його, перебудувати .venv, перевірити й зафіксувати знову.
if not exist .venv (
  echo Спершу створи середовище: меню - 2 ^(Перше налаштування^)
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
pip freeze --exclude-editable > requirements.lock.txt
if errorlevel 1 ( echo НЕ ВДАЛОСЬ & pause & exit /b 1 )
echo Записано: requirements.lock.txt ^(build_exe.bat тепер збирає з нього^)
pause
