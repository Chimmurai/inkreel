@echo off
chcp 65001 >nul
cd /d %~dp0..
if not exist .venv (
  python -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
if exist requirements.lock.txt (
  echo Версії бібліотек - з requirements.lock.txt
  pip install -r requirements.lock.txt pyinstaller
) else (
  pip install -r requirements.txt pyinstaller
)
pyinstaller inkreel.spec --noconfirm
if errorlevel 1 ( echo ЗБІРКА НЕ ВДАЛАСЬ & pause & exit /b 1 )

rem Меню правої кнопки для збірки без інсталятора: menu_on.bat додає пункти для фото й GIF, menu_off.bat прибирає (лише HKCU, без прав адміністратора)
> "dist\InkReel\menu_on.bat" echo @start "" "%%~dp0InkReel.exe" --register-menu
> "dist\InkReel\menu_off.bat" echo @start "" "%%~dp0InkReel.exe" --unregister-menu

rem Ліцензії сторонніх компонентів їдуть разом із програмою - і в папці, і в інсталяторі, і в пакеті для Store (THIRD_PARTY_NOTICES.md, licenses\)
python tools\collect_licenses.py dist\InkReel
if errorlevel 1 ( echo ЛІЦЕНЗІЇ НЕ СКЛАДЕНО & pause & exit /b 1 )

echo.
echo Готово: dist\InkReel\InkReel.exe
echo Меню правої кнопки без інсталятора: dist\InkReel\menu_on.bat ^(вимкнути: menu_off.bat^)

rem Inno Setup після встановлення не додається в PATH, тож шукаємо ISCC.exe у звичних теках
set "ISCC="
where iscc >nul 2>nul && set "ISCC=iscc"
if not defined ISCC if exist "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if not defined ISCC if exist "%ProgramFiles%\Inno Setup 6\ISCC.exe" set "ISCC=%ProgramFiles%\Inno Setup 6\ISCC.exe"
if not defined ISCC if exist "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" set "ISCC=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
if not defined ISCC goto noiscc
python tools\sync_installer.py
"%ISCC%" installer.iss
if errorlevel 1 ( echo ІНСТАЛЯТОР НЕ ЗІБРАВСЬ & pause & exit /b 1 )
echo Інсталятор: dist\InkReel-Setup.exe
goto end
:noiscc
echo Щоб зробити інсталятор, встанови Inno Setup ^(https://jrsoftware.org/isinfo.php^) і запусти build_exe.bat ще раз.
:end
pause
