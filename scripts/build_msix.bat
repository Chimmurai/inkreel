@echo off
chcp 65001 >nul
cd /d %~dp0..
rem Пакет MSIX для Microsoft Store: dist\InkReel.msix. Спершу build_exe.bat. Дані з Partner Center - store\identity.json, кроки - store\STORE.md
rem   build_msix.bat                                   пакет для відправки в Store
rem   build_msix.bat --stage-only                      лише тека dist\msix (без Windows SDK)
rem   build_msix.bat --sign store\InkReelDev.pfx       пакет, підписаний тестовим сертифікатом (store\make_test_cert.ps1), щоб поставити його на цей ПК
if not exist dist\InkReel\InkReel.exe (
  echo Спершу збери програму: build_exe.bat
  pause
  exit /b 1
)
if not exist .venv (
  python -m venv .venv
)
call .venv\Scripts\activate.bat
pip install -q pillow numpy
python tools\make_msix.py %*
if errorlevel 1 (
  echo ПАКЕТ НЕ ЗІБРАВСЬ
  pause
  exit /b 1
)
echo.
echo Далі: store\STORE.md
pause
