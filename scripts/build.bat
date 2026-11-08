@echo off
chcp 65001 >nul
cd /d %~dp0..
:menu
echo.
echo   Збірка ^(результат у dist\^)
echo   1  Програма + інсталятор: dist\InkReel\InkReel.exe, dist\InkReel-Setup.exe
echo   2  Пакет MSIX для Microsoft Store
echo   3  MSIX підписаний тестовим сертифікатом
echo   4  Зафіксувати версії бібліотек ^(requirements.lock.txt^): збірка буде відтворюваною
echo   0  Назад
choice /c 12340 /n /m "  Вибір: "
set K=%errorlevel%
if %K%==5 exit /b 0
if %K%==1 call scripts\build_exe.bat
if %K%==2 call scripts\build_msix.bat
if %K%==3 call scripts\build_msix.bat --sign store\InkReelDev.pfx
if %K%==4 call scripts\freeze_deps.bat
goto menu
