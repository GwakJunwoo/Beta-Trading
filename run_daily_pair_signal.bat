@echo off
REM Beta Trading daily pair signal -> Telegram (scheduled task wrapper)
REM Runs at 09:10 daily.
cd /d "C:\Users\infomax\Beta Trading"

set LOGDIR=data\factor_trading\daily_pair_signal
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
set LOGFILE=%LOGDIR%\_run.log

echo. >> "%LOGFILE%"
echo ============================================================ >> "%LOGFILE%"
echo [%date% %time%] daily_pair_signal start >> "%LOGFILE%"
echo ============================================================ >> "%LOGFILE%"

set PYTHONIOENCODING=utf-8
"C:\Users\infomax\anaconda3\python.exe" "factor_trading\scripts\daily_pair_signal.py" 1>> "%LOGFILE%" 2>&1
set RC=%ERRORLEVEL%

echo. >> "%LOGFILE%"
echo [%date% %time%] daily_pair_signal done (exit=%RC%) >> "%LOGFILE%"
exit /b %RC%
