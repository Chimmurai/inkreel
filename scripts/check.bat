@echo off
chcp 65001 >nul
cd /d %~dp0..
if not exist reports mkdir reports
:menu
echo.
echo   Перевірки й заміри ^(звіти пишуться в теку reports\^)
echo   1  Нейро ^(рівень 6^): бекенд, мс на кадр, стадії, паралельність -^> reports\student_check.txt
echo   2  Швидкість рівнів Ink/Symbols -^> reports\bench_report.md
echo   3  Заміри легких пресетів на DirectML -^> reports\dml_lite.txt
echo   4  DirectML проти CUDA/TensorRT ^(окремі середовища^)
echo   0  Назад
choice /c 12340 /n /m "  Вибір: "
set K=%errorlevel%
if %K%==5 exit /b 0
if %K%==4 call scripts\bench_backends.bat
if %K%==1 call :student
if %K%==2 call :bench
if %K%==3 call :lite
pause
goto menu

:student
if not exist .venv ( echo Спершу створи .venv & exit /b 1 )
call .venv\Scripts\activate.bat
pip install -q onnxruntime-directml
set INKREEL_STUDENT=auto
set F=reports\student_check.txt
echo === Нейро: вибір бекенду і замір на 720p > %F%
python -m benvid.student --probe --bench 720 1280 >> %F% 2>&1
echo === 1080p і 1440p на вибраному бекенді >> %F%
python -m benvid.student --bench 1080 1920 >> %F% 2>&1
python -m benvid.student --bench 1440 2560 >> %F% 2>&1
echo === стадії кадра: 1080p і 1440x3438 >> %F%
python -m benvid.student --stages 1080 1920 >> %F% 2>&1
python -m benvid.student --stages 1440 3438 >> %F% 2>&1
echo === паралельність: 1 сеанс проти кількох і відеокарта + процесор >> %F%
python -m benvid.student --par 1080 1920 >> %F% 2>&1
python -m benvid.student --par 1440 3438 >> %F% 2>&1
type %F%
call deactivate
exit /b 0

:bench
if not exist .venv ( echo Спершу створи .venv & exit /b 1 )
call .venv\Scripts\activate.bat
python -m benvid.bench --out reports
call deactivate
exit /b 0

:lite
if not exist .venv_train ( echo Потрібне .venv_train & exit /b 1 )
if not exist .venv_dml ( echo Потрібне .venv_dml & exit /b 1 )
if not exist reports\tmp mkdir reports\tmp
call .venv_train\Scripts\activate.bat
for %%P in (fastq fq_noh2 fq_nor1 fq_nb1 fq_a fq_b fq_c fq_d) do python tools\train_ink.py export --random --fp16 --parts --mask --preset %%P --out reports\tmp\lite_%%P.onnx
call deactivate
set F=reports\dml_lite.txt
echo === 1920x1072 > %F%
for %%P in (fastq fq_noh2 fq_nor1 fq_nb1 fq_a fq_b fq_c fq_d) do ( echo --- %%P >> %F% & .venv_dml\Scripts\python tools\bench_dml.py reports\tmp\lite_%%P.onnx >> %F% 2>&1 )
echo === 3440x1440 >> %F%
for %%P in (fastq fq_noh2 fq_nor1 fq_nb1 fq_a fq_b fq_c fq_d) do ( echo --- %%P >> %F% & .venv_dml\Scripts\python tools\bench_dml.py reports\tmp\lite_%%P.onnx --h 1440 --w 3440 >> %F% 2>&1 )
type %F%
exit /b 0
